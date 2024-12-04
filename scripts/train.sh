#!/bin/bash
# nohup python -u tools/train.py > logs.log &
# nohup python -u train.py \
#     --num_epochs 100 \
#     --batch_size 4 \
#     --lr 0.001 \
#     --alpha 1.0 \
#     --beta 0.0 \
#     --watermark_region 4.0 \
#     --train_dir "data/wikiart" \
#     --train_classes "data/wikiart/train_classes.csv" \
#     --eval_dir "data/wikiart" \
#     --eval_classes "data/wikiart/eval_classes.csv" \
#     > logs.log &

sbatch train_model.slurm
