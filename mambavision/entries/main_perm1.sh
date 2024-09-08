#!/bin/bash
#SBATCH --job-name=mv_perm1
#SBATCH --error=/home/anhnd81/anhnd81/workspace/mambavision_1/mambavision/result/mambaV_perm1_1.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=sdc2-hpc-dgx-a100-020
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=40
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/vinai/users/anhnd81/envs/mambavision
cd /home/anhnd81/anhnd81/workspace/mambavision_1/mambavision
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES

#Superpod:
#DATA_PATH="/home/anhnd81/anhnd81/.cache/imagenet/train"

#DATA_PATH="/home/ubuntu/workspace/dataset/imagenet"
DATA_PATH="/home/anhnd81/anhnd81/.cache/imagenet"
MODEL=mamba_vision_T
BS=128
EXP=perm1
LR=8e-4
WD=0.05
WR_LR=1e-6
DR=0.38
MESA=0.25
#RUN_FILE="/home/ubuntu/workspace/mambavision_1/mambavision/train_attn.py"
RUN_FILE="/home/anhnd81/anhnd81/workspace/mambavision_1/mambavision/train_perm1.py"
# checkpoint="/home/anhnd81/anhnd81/workspace/mambavision_1/output/train/Attn1/20240814-163841-mamba_vision_T-224/checkpoint-76.pth.tar"
torchrun --nproc_per_node=1 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
--data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR # \
# --resume=$checkpoint
