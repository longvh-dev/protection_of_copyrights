#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=logs_%j.out             # File log 
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1

python -u train.py \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 0.001 \
    --alpha 1.0 \
    --beta 0.0 \
    --watermark_region 4.0 \
    --train_dir "data/wikiart" \
    --train_classes "data/wikiart/train_classes.csv" \
    --eval_dir "data/wikiart" \
    --eval_classes "data/wikiart/eval_classes.csv"