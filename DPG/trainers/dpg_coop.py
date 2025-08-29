import os
import numpy as np
from torch.cuda.amp import GradScaler, autocast

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import dassl
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.utils.tools import mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import listdir_nohidden

from .basedg import *
from utils.clip_part import *



_tokenizer = _Tokenizer()

class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DPG.N_CTX
        ctx_init = cfg.TRAINER.DPG.CTX_INIT
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        domainnames = cfg.DATASET.SOURCE_DOMAINS
        domainnames = [
            ", a {} image.".format(domain) for domain in domainnames
        ]

        n_dm = len(cfg.DATASET.SOURCE_DOMAINS)
        n_dmx = cfg.TRAINER.DPG.N_DMX  
        n = n_dmx + n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        
        if ctx_init: # use given words to initialize context vectors 
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:   # random initialization
            if cfg.TRAINER.DPG.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=self.dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n)

        domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(domain_vectors, std=0.02)
        self.domain_vectors = nn.Parameter(domain_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        self.ctx = nn.Parameter(ctx_vectors)  

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
       

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
       
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.DPG.CSC
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DPG.CLASS_TOKEN_POSITION
     
    def forward(self, cfg, domain):
        ctx = self.ctx
        ctx_dim = ctx.size(-1)
        dmx = self.domain_vectors  

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            if self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1,
                                              -1)  
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1,
                                          -1)  
       
        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  
        
        prompts = []
        for i in range(len(domain)):
            domain_name = cfg.ALL_DOMAINS[int(domain[i])]
            domain_index = cfg.SOURCE_DOMAINS.index(domain_name)

            ctxdmx = torch.cat([ctx, dmx[domain_index]],
                            dim=1).reshape(self.n_cls,
                                            self.n_ctx + self.n_dmx, ctx_dim)
            prefix = self.token_prefix
            suffix = self.token_suffix

            prompts.append(self.construct_prompts(ctxdmx, prefix, suffix))

        prompts = torch.stack(prompts).type(self.dtype)
   
        return prompts
    
    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompts = torch.tensor(0)

    def forward(self, cfg, image, domain):
        prompts = self.prompt_learner(cfg, domain)
        self.prompts = prompts
        
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.unsqueeze(1)
        
        logits = torch.empty(image.shape[0], text_features.shape[1])

        logit_scale = self.logit_scale.exp()
        for i in range(image.shape[0]):
            logits[i] = logit_scale * image_features[i] @ text_features[i].t()

        return logits
    
    
@TRAINER_REGISTRY.register()
class DPG_CoOp(BaseDG):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPG.PREC in ["fp16", "fp32", "amp"]


    def build_model(self):
        cfg = self.cfg
        print(cfg)
        classnames = self.dm.dataset.classnames
        if not cfg.TEST.NO_TEST:
            self.test_best_result = -np.inf
            
        if torch.cuda.is_available() and cfg.USE_CUDA:
            if len(cfg.GPU) == 1:
                self.device = torch.device("cuda:{}".format(cfg.GPU))
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})...")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.DPG.PREC == "fp32" or cfg.TRAINER.DPG.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.n_dm = self.model.prompt_learner.n_dm
        self.n_cls = self.model.prompt_learner.n_cls

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        len_train_loader_x = len(self.train_loader_x)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        else:
            raise ValueError
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.DPG.PREC == "amp" else None
    
    def forward_backward(self, batch):
        images, labels, domain = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.DPG.PREC
        if prec == "amp":
            with autocast():
                output = self.model(self.cfg, images, domain).to(self.device)
                loss = F.cross_entropy(output, labels)
                # cosine_sim = 0
                l2_distance = 0
                if "distance" in self.cfg.OUTPUT_DIR:
                    if self.model.prompt_learner.n_ctx == self.model.prompt_learner.n_dmx:
                        '''for i in range(len(self.model.prompt_learner.domain_vectors)):
                            cosine_sim = cosine_sim + F.cosine_similarity(self.model.prompt_learner.ctx.view(-1), 
                                                                        self.model.prompt_learner.domain_vectors[i].view(-1), dim = -1)
                            for j in range(i + 1, len(self.model.prompt_learner.domain_vectors)):
                                if j < len(self.model.prompt_learner.domain_vectors):
                                    cosine_sim = cosine_sim + F.cosine_similarity(self.model.prompt_learner.domain_vectors[i].view(-1), 
                                                                                self.model.prompt_learner.domain_vectors[j].view(-1), dim = -1)
                        cosine_sim = cosine_sim / 6'''
                        for i in range(len(self.model.prompt_learner.domain_vectors)):
                            l2_distance = l2_distance + torch.cdist(self.model.prompt_learner.ctx, 
                                                                        self.model.prompt_learner.domain_vectors[i], p = 2)
                            for j in range(i + 1, len(self.model.prompt_learner.domain_vectors)):
                                if j < len(self.model.prompt_learner.domain_vectors):
                                    l2_distance = l2_distance + torch.cdist(self.model.prompt_learner.domain_vectors[i], 
                                                                                self.model.prompt_learner.domain_vectors[j], p = 2)
                        l2_distance = l2_distance / 6
                    else:
                        for i in range(len(self.model.prompt_learner.domain_vectors)):
                            for j in range(i + 1, len(self.model.prompt_learner.domain_vectors)):
                                if j < len(self.model.prompt_learner.domain_vectors):
                                    '''cosine_sim = cosine_sim + F.cosine_similarity(self.model.prompt_learner.domain_vectors[i].view(-1), 
                                                                                self.model.prompt_learner.domain_vectors[j].view(-1), dim = -1)'''
                                    l2_distance = l2_distance + torch.cdist(self.model.prompt_learner.domain_vectors[i], 
                                                                                self.model.prompt_learner.domain_vectors[j], p = 2)
                        l2_distance = l2_distance / 3
                        # cosine_sim = cosine_sim / 3
                    '''cosine_distance = 1 - cosine_sim   
                    closs = -cosine_distance.mean()
                    total_loss = loss + closs'''
                    total_loss = loss - 0.5 * l2_distance.mean()
                else:
                    total_loss = loss
                self.optim.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
        else:
            output = self.model(images)
            loss = F.cross_entropy(output, labels)
            cosine_sim = 0
            if self.model.prompt_learner.n_ctx == self.model.prompt_learner.n_dmx:
                for i in range(len(self.model.prompt_learner.domain_vectors)):
                    cosine_sim = cosine_sim + F.cosine_similarity(self.model.prompt_learner.ctx.view(-1), 
                                                                self.model.prompt_learner.domain_vectors[i].view(-1), dim = -1)
                    for j in range(i + 1, len(self.model.prompt_learner.domain_vectors)):
                        if j < len(self.model.prompt_learner.domain_vectors):
                            cosine_sim = cosine_sim + F.cosine_similarity(self.model.prompt_learner.domain_vectors[i].view(-1), 
                                                                        self.model.prompt_learner.domain_vectors[j].view(-1), dim = -1)
                cosine_sim = cosine_sim / 6
            else:
                for i in range(len(self.model.prompt_learner.domain_vectors)):
                    for j in range(i + 1, len(self.model.prompt_learner.domain_vectors)):
                        if j < len(self.model.prompt_learner.domain_vectors):
                            cosine_sim = cosine_sim + F.cosine_similarity(self.model.prompt_learner.domain_vectors[i].view(-1), 
                                                                        self.model.prompt_learner.domain_vectors[j].view(-1), dim = -1)
                cosine_sim = cosine_sim / 3
            cosine_distance = 1 - cosine_sim  
            closs = -cosine_distance.mean(())
            total_loss = loss + closs
            self.model_backward_and_update(total_loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, labels)[0].item(),
        }
        
        if "distance" in self.cfg.OUTPUT_DIR:
            # loss_summary["distance"] = closs.item()
            loss_summary["distance"] = l2_distance.mean().item()
            loss_summary["total_loss"] = total_loss.item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        curr_result = self.test(split="val")
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.best_epoch = self.epoch
            if self.cfg.SAVE_MODEL:
                self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
                
                prompts_ctx_path = os.path.join(self.output_dir, 'prompt_learner', f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}_ctx.pt')
                torch.save(self.model.prompt_learner.ctx, prompts_ctx_path)
                prompts_dmx_path = os.path.join(self.output_dir, 'prompt_learner', f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}_dmx.pt')
                torch.save(self.model.prompt_learner.domain_vectors, prompts_dmx_path)
        
                self.n_ctx = self.cfg.TRAINER.DPG.N_CTX
                prompt_dir = 'prompt_labels' + '/' + self.cfg.DATASET.NAME.split('_')[1] + '/' + self.cfg.MODEL.BACKBONE.NAME.replace('/', '') + '/' + 'seed_' + str(self.cfg.SEED)
                mkdir_if_missing(prompt_dir)
                prompts_ctx_label_path = os.path.join(prompt_dir, f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}_ctx.pt')
                prompts_ctx = self.model.prompt_learner.ctx
                torch.save(prompts_ctx,prompts_ctx_label_path)
                prompts_dmx_label_path = os.path.join(prompt_dir, f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}_dmx.pt')
                prompts_dmx = self.model.prompt_learner.domain_vectors
                torch.save(prompts_dmx, prompts_dmx_label_path)

                print(f'Prompt saved to {prompt_dir}')
        print('Domain {} val best acc: {:.1f}%, best epoch: {}'.format(self.cfg.TARGET_DOMAIN, self.best_result, self.best_epoch+1))
        
        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)


