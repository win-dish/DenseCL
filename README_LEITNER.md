# How to run different SSL algorithms

The algorithms in this repository are run with the `dist_train.sh` script located in the tools directory. There are 2 mandatory positional command line arguments:
* CONFIG_FILE: the respective algorithm's config file located in `configs/selfsup/<algoritm>`
* GPUS: the number of GPUs to be used (use 1 if training on local machine)
Other optional arguments are:
* `--resume_from ${CHECKPOINT_FILE}`: training is resumed from checkpoint file passed (**not working yet**)
* `--pretrained $ {PRETRAIN_WEIGHTS}`: load pretrained weights for the backbone (**not tested yet**)
* `--deterministic`: Switch on "deterministic" mode which slows down training but the results are reproducible (**not tested yet**)

```bash
./tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```

in `tools/dist_train.sh` example cli commands for starting the training with various algorithms can be found commented

The Learning rate needs to be adjusted based on the number of GPUs used according to the following formula: $newLearnRate = oldLearnRate \cdot \dfrac{newGPUs}{oldGPUs}$

## DenseCL

Checkpoints are not working yet -> appears to be a problem with mmcv, which identifies syntax error in python unknown file


## BYOL

Not working yet -> problem with unexpected argument `filter_size`. Checkpoints to be tested later
IF `filter_size` is commented out, algotithm outputs warnings, but neither training starts, nor algorithm is terminated with error code (maybe neet to wait more and see what happens, but more likely to not work with commented out kernel size)

**Problem:** Gaussian Blur implementation called in config file in line 49-59 with 
```python
dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0,
                kernel_size=23
                )
        ],
        p=1.),
```

accepts no kernel size in __init__() funciton (located in openselfsup/datasets/pipelines/transformers.py). This should be somehow fixed, since kernel-size is probably needed.

## MOCO

Checkpoints ae not working yet -> appears to be the same problem as for DenseCL

## SimCLR

AttributeError: 'SyncBatchNorm' object has no attribute '_specify_ddp_gpu_num' first produce this error.

**NOT SURE IF SOLUTION IS WORKING**
Error comes from SyncBatchNorm module within pytorch.  _specigy_ddp_gup_num was used in older versions of Pytorch, is now deprecated and shouldn't be necessary anymore.
To solve the problem and start training i inserted this code to check, whether the attribute `_specify_ddp_gpu_num` existed before calling it in "openselfsup/models/utils/norm.py" in line 47-48

replaced 
```python
layer._specify_ddp_gpu_num(1)
```

with
```python
if hasattr(layer, '_specify_ddp_gpu_num'):
    layer._specify_ddp_gpu_num(1)
```

This problem was probably due to a compatibility issue with my newer pytorch version.

*NOTE:* seems to converge faster initially than MOCO or DenseCL -> could be because only one global loss is calculated and not local + global loss

Checkpoints are not working yet -> same problem as other algorithms.