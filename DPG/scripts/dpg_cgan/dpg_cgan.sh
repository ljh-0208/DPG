DATA=   # ******* your data path *******
TRAINER=DPG_CGAN

DATASET=$1
CFG=$2          # config file
BACKBONE=$3     # backbone name
GPU=$4

# bash scripts/dpg_cgan/dpg_cgan.sh pacs dpg RN50 0
# bash scripts/dpg_cgan/dpg_cgan.sh vlcs dpg RN50 0
# bash scripts/dpg_cgan/dpg_cgan.sh office_home dpg RN50 1

# bash scripts/dpg_cgan/dpg_cgan.sh pacs dpg ViT-B/16 0
# bash scripts/dpg_cgan/dpg_cgan.sh vlcs dpg ViT-B/16 0
# bash scripts/dpg_cgan/dpg_cgan.sh office_home dpg ViT-B/16 1


if [ "$DATASET" = "pacs" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 's')
elif [ "$DATASET" = "vlcs" ]; then
  ALL_DOMAIN=('c' 'l' 'p' 's')
elif [ "$DATASET" = "office_home" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 'r')
fi


for SEED in 1 2 3 4 5
do
  for DOMAIN in "${ALL_DOMAIN[@]}"
  do
    DIR=outputs/dpg/multi_DG/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAIN}/seed_${SEED}

    if [ -d "$DIR" ]; then
      echo "Results are available in ${DIR}, so skip this job"
    else
      echo "Run this job and save the output to ${DIR}"
        
      python train.py \
        --backbone ${BACKBONE} \
        --target-domains ${DOMAIN} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/multi_source/${DATASET}.yaml \
        --config-file configs/trainers/DPG_CGAN/${CFG}.yaml \
        --output-dir ${DIR} \
        --seed ${SEED} \
        --gpu ${GPU}    
    fi
  done
done
