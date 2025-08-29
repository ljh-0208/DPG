DATA=    # ******* your data path *******
TRAINER=DPG_CoOp
CFG=b32_ep50    # config file
CTX=16
DMX=4

DATASET=$1
BACKBONE=$2     # backbone name
GPU=$3


# bash scripts/spg_coop/spg_coop.sh pacs ViT-B/16 0
# bash scripts/spg_coop/spg_coop.sh vlcs RN50 1
# bash scripts/spg_coop/spg_coop.sh office_home RN50 2 

# bash scripts/spg_coop/spg_coop_ctx16_dmx4.sh pacs ViT-B/16 0
# bash scripts/spg_coop/spg_coop.sh vlcs ViT-B/16 1
# bash scripts/spg_coop/spg_coop.sh office_home ViT-B/16 2

if [ "$DATASET" = "pacs" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 's')
elif [ "$DATASET" = "vlcs" ]; then
  ALL_DOMAIN=('c' 'l' 'p' 's')
elif [ "$DATASET" = "office_home" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 'r')
fi


for DOMAIN in "${ALL_DOMAIN[@]}"
do
  EXT_BACKBONE="${BACKBONE}_ctx${CTX}_dmx${DMX}"
  DIR=outputs/SPG/${TRAINER}/${DATASET}/seed_${SEED}/${CFG}/${EXT_BACKBONE//\//}/${DOMAIN}

  if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}, so skip this job"
  else
    echo "Run this job and save the output to ${DIR}"
    
    python train.py \
      --backbone ${BACKBONE} \
      --target-domains ${DOMAIN} \
      --root ${DATA} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}_coop.yaml \
      --config-file configs/trainers/DPG_CoOp/${CFG}.yaml \
      --output-dir ${DIR} \
      --gpu ${GPU} \
      --ctx ${CTX} \
      --dmx ${DMX}
  fi
done
