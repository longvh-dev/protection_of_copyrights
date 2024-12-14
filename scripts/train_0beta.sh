#!/bin/bash
nohup python -u train.py \
    --name dank-base \
    --num_epochs 500 \
    --batch_size 2 \
    --lr 0.001 \
    --alpha 1.0 \
    --beta 10.0 \
    --watermark_region 4.0 \
    --train_dir "data/imagenet" \
    --train_classes "data/imagenet/train_classes_1.csv" \
    --eval_dir "data/imagenet" \
    --eval_classes "data/imagenet/train_classes.csv" \
    > logs_dank.log &
    # --checkpoint "checkpoint_epoch_160.pth" \