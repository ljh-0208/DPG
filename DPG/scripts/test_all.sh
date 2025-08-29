DATA=    # ******* your data path *******
TRAINER=DPG_CGAN

DATASET=$1
CFG=$2  # config file
BACKBONE=$3 # backbone name
GPU=$4

# bash scripts/test_all.sh pacs dpg RN50 0
# bash scripts/test_all.sh pacs dpg ViT-B/16 0

if [ "$DATASET" = "pacs" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 's')
elif [ "$DATASET" = "vlcs" ]; then
  ALL_DOMAIN=('c' 'l' 'p' 's')
elif [ "$DATASET" = "office_home" ]; then
  ALL_DOMAIN=('a' 'c' 'p' 'r')
fi

for TEST_BATCHSIZE in 1
do
  for DOMAIN in "${ALL_DOMAIN[@]}"
  do
    DIR=outputs/DPG/multi_DG/DPG_CGAN/pacs/dpg/${BACKBONE}/${DOMAIN}

    if [ -d "$DIR" ]; then
      echo "Results are available in ${DIR}, so skip this job"
    else
      echo "Run this job and save the output to ${DIR}"

      MODEL_DIR=outputs/DPG/multi_DG/DPG_CGAN/pacs/dpg/DPG_models/${DOMAIN}

      python train.py \
        --backbone ${BACKBONE} \
        --target-domains ${DOMAIN} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/multi_source/${DATASET}.yaml \
        --config-file configs/trainers/DPG_CGAN/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --gpu ${GPU} \
        --test-batchsize ${TEST_BATCHSIZE} \
        --eval-only
    fi
  done
done
