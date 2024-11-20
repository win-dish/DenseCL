#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
CHECKPOINT_DIR="${WORK_DIR}latest.pth"

echo "$CHECKPOINT_DIR"

torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --work_dir $WORK_DIR --seed 0 --launcher pytorch ${PY_ARGS}

#./tools/dist_train.sh configs/selfsup/densecl/densecl_coco_800ep.py 1 