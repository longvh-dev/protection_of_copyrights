#!/bin/bash
export PYTHONPATH=$(pwd)
python inference.py \
    --checkpoints checkpoint_epoch_160.pth \
    --prompt "A photo, A oil painting, A sketch" \
    --image "data/imagenet/IMAGENET_CAT/n02123045_3567_n02123045.JPEG" \
    --watermark "IMAGENET_CAT" \
    --strength 0.1 \