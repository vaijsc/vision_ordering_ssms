#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,4,6,7
export CUDA_VISIBLE_DEVICES=1,2,3,4
export TORCH_DISTRIBUTED_DEBUG=DETAIL
DATA_PATH="/home/ubuntu/workspace/dataset/imagenet1K"
MODEL=TransMamba_T
BS=128
# EXP=mambaV_test1
# EXP=Test
EXP=transmam
LR=8e-4 # original 8e-4
WD=0.05
WR_LR=1e-6
DR=0.2
MESA=0.25
RUN_FILE="/home/ubuntu/workspace/mambavision_1/mambavision/train_2.py"
torchrun --master-port=21383 --nproc_per_node=4 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
 --data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR \
> >(tee -a /home/ubuntu/workspace/mambavision_1/mambavision/result/mambaV_transpool.txt) 2>&1
# softsort with order token is linear projection of input -> linear projection dim [49 -> 1] - mamba_vision_test.py
# like above but for each block - mamba_vision_test1.py 