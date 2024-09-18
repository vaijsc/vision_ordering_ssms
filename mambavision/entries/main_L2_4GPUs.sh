#!/bin/bash
#SBATCH --job-name=mv_L2_perm1
#SBATCH --error=/lustre/scratch/client/vinai/users/ducna22/workspace/mambavision_1/mambavision/result/mambaV_L2_perm1_4GPUs.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=sdc2-hpc-dgx-a100-011
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

#module purge
#module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
#conda activate /lustre/scratch/client/vinai/users/anhnd81/envs/mambavision
#conda activate /lustre/scratch/client/vinai/users/phinh2/workspace/envs/mambav
conda activate vssm
#cd /home/anhnd81/anhnd81/workspace/mambavision_1/mambavision
#cd /lustre/scratch/client/vinai/users/phinh2/workspace/mambavision_1/mambavision
cd /lustre/scratch/client/vinai/users/ducna22/workspace/mambavision_1/mambavision
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES

#Superpod:
#DATA_PATH="/home/anhnd81/anhnd81/.cache/imagenet/train"
#DATA_PATH="/home/anhnd81/anhnd81/.cache/imagenet"
#DATA_PATH="/lustre/scratch/client/vinai/users/phinh2/workspace/dataset/imagenet"
DATA_PATH="/lustre/scratch/client/vinai/users/ducna22/data/imagenet"
MODEL=mamba_vision_L2
BS=128
EXP=perm1
#EXP=Test # annotate this after finish debugging
LR=8e-4
WD=0.05
WR_LR=1e-6
DR=0.38
MESA=0.25
RUN_FILE="/lustre/scratch/client/vinai/users/ducna22/workspace/mambavision_1/mambavision/train_perm1.py"
torchrun --master_port 12367 --nproc_per_node=4 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
--data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR \
--resume /lustre/scratch/client/vinai/users/ducna22/workspace/mambavision_1/output/train/perm1/20240916-103042-mamba_vision_L2-224/checkpoint-0.pth.tar
