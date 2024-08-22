#!/bin/bash

#Superpod:
#DATA_PATH="/home/anhnd81/anhnd81/.cache/imagenet/val"

DATA_PATH="/home/ubuntu/workspace/dataset/imagenet/val"
BS=128
# checkpoint='/home/ubuntu/workspace/mambavision_1/mambavision/model_weights/mambavision_tiny_1k.pth.tar'
checkpoint='/home/ubuntu/workspace/mambavision_1/mambavision/model_weights/model_best.pth.tar'
CUDA_VISIBLE_DEVICES=3 python validate.py --model mamba_vision_T --checkpoint=$checkpoint --data-dir=$DATA_PATH --batch-size $BS --input-size 3 224 224

