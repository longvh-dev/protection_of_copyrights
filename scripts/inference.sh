#!/bin/bash
export PYTHONPATH=$(pwd)
python inference.py \
    --checkpoints checkpoints/20241203-115147/checkpoint_epoch_200.pth \
    --prompt "A photo, A oil painting, A sketch" \
    --image "data/wikiart/Early_Renaissance/andrea-del-castagno_crucifixion-1.jpg" \
    --watermark "aaron siskin" \
    --strength 0.1 \