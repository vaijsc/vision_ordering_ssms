#!/bin/bash
#SBATCH --job-name=perm5_mv
#SBATCH --error=/lustre/scratch/client/vinai/users/phinh2/workspace/mambavision_1/mambavision/result/mambaV_perm5.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=sdc2-hpc-dgx-a100-017
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=40
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

eval "$(conda shell.bash hook)"
#conda activate /lustre/scratch/client/vinai/users/anhnd81/envs/mambavision
conda activate /lustre/scratch/client/vinai/users/phinh2/workspace/envs/mambav
#cd /home/anhnd81/anhnd81/workspace/mambavision_1/mambavision
cd /lustre/scratch/client/vinai/users/phinh2/workspace/mambavision_1/mambavision
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES

#Superpod:
#DATA_PATH="/home/anhnd81/anhnd81/.cache/imagenet/train"

#DATA_PATH="/home/anhnd81/anhnd81/.cache/imagenet"
DATA_PATH="/lustre/scratch/client/vinai/users/phinh2/workspace/dataset/imagenet"
MODEL=mamba_vision_T
BS=128
EXP=perm1-5
#EXP=Test
LR=8e-4
WD=0.05
WR_LR=1e-6
# DR=0.2
MESA=0.25
RUN_FILE="/lustre/scratch/client/vinai/users/phinh2/workspace/mambavision_1/mambavision/train_perm1_4.py"
torchrun --master_port 12370 --nproc_per_node=4 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
 --data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD}  --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR # \
# --resume /lustre/scratch/client/vinai/users/phinh2/workspace/mambavision_1/output/train/Original/20240817-001048-mamba_vision_T-224/checkpoint-308.pth.tar
# --drop-path ${DR}s