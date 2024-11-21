#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
CHECKPOINT_DIR="${WORK_DIR}latest.pth"

echo "$CHECKPOINT_DIR" # debugging line, can be removed later

torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --work_dir $WORK_DIR --seed 0 --launcher pytorch ${PY_ARGS}

# FOR RUNNING DENSECL #working with no checkpoints
#./tools/dist_train.sh configs/selfsup/densecl/densecl_coco_800ep.py 1
# TO ADD CHECKPOINTS EXAMPLE: --resume_from work_dirs/selfsup/densecl/densecl_coco_800ep/latest.pth

# FOR RUNNING BYOL # not working yet
# ./tools/dist_train.sh configs/selfsup/byol/r50_bs4096_ep200_coco.py 1

# FOR RUNNING MOCO # working with no checkpoints
# ./tools/dist_train.sh configs/selfsup/moco/r50_v2.py 1
#

# FOR RUNNING SIMCLR
# ./tools/dist_train.sh configs/selfsup/simclr/r50_bs256_ep200.py 1