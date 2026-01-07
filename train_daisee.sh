#!/bin/bash

# Script to train on DAiSEE dataset (Engagement only)
# Uses dummy bounding boxes (full frame) and custom text prompts.

python main.py \
    --mode train \
    --exper-name daisee_engagement_finetune \
    --dataset DAiSEE \
    --gpu mps \
    --epochs 20 \
    --batch-size 8 \
    --lr 0.0001 \
    --lr-image-encoder 1e-5 \
    --lr-prompt-learner 0.001 \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --milestones 10 15 \
    --gamma 0.1 \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --print-freq 10 \
    --root-dir /content/drive/MyDrive/khoaluan/Dataset/DAiSEE/DataSet \
    --train-annotation train.txt \
    --test-annotation val.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face /content/drive/MyDrive/khoaluan/Dataset/DAiSEE/face_bbox_dummy.json \
    --bounding-box-body /content/drive/MyDrive/khoaluan/Dataset/DAiSEE/body_bbox_dummy.json \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True
