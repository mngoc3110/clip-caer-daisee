#!/bin/bash

python main.py \
    --mode eval \
    --dataset DAiSEE \
    --gpu mps \
    --exper-name daisee_final_eval \
    --eval-checkpoint "outputs/daisee_engagement_finetune-[01-07]-[16:04]/model_best.pth" \
    --root-dir DAiSEE_data/DataSet \
    --train-annotation ../daisee_train.txt \
    --test-annotation ../daisee_test.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face DAiSEE_data/face_bbox_dummy.json \
    --bounding-box-body DAiSEE_data/body_bbox_dummy.json \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42