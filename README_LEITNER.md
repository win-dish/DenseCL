# Dense Contrastive Learning Algorithm

## How to run this algorithm

to start the Training execute the `dist_train.sh` file within the tools folder (example command `./tools/dist_train.sh`).

The shell script expects the followign command line arguments:
* configuration file (i.e. `configs/selfsup/densecl/densecl_coco_800ep.py`) at pos 1
* number of GPUs to be used at pos 2 (to use on local machine use 1 as value)

To restart the training from en existing checkpoint, add the checkpoint file name to the `CHECKPOINT_DIR` variable specified in the shell script and add `--resume from "$CHECKPOINT_DIR"` to the last line of the shell script right before `${PY_ARGS}`. IF you with to start training from scratch you can omit this.

## Metadata

Training on local machine requires approx. 12mins per epoch with the COCO2017 dataset.