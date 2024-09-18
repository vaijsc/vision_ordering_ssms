#!/bin/bash
#SBATCH --job-name=mv_ori
#SBATCH --error=/lustre/scratch/client/vinai/users/ducna22/workspace/mambavision_1/mambavision/result/mambaV_ori_4gpus.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=sdc2-hpc-dgx-a100-006
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

#module purge
#module load python/miniconda3/miniconda3
#eval "$(conda shell.bash hook)"
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
MODEL=mamba_vision_T
BS=128
EXP=original
#EXP=Test # annotate this after finish debugging
LR=8e-4
WD=0.05
WR_LR=1e-6
# DR=0.38
MESA=0.25
RUN_FILE="/lustre/scratch/client/vinai/users/ducna22/workspace/mambavision_1/mambavision/train.py"
<<<<<<< HEAD
torchrun --master_port 12368 --nproc_per_node=1 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
 --data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR 
=======
torchrun --master_port 12358 --nproc_per_node=1 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
 --data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR \
> '/lustre/scratch/client/vinai/users/ducna22/workspace/mambavision_1/mambavision/result/mambaV_ori_4gpus.txt' 2>&1
>>>>>>> 824b36cfd00b5d030f36561d33df3d646cdc1c7b
# --resume /lustre/scratch/client/vinai/users/phinh2/workspace/mambavision_1/output/train/Original/20240817-001048-mamba_vision_T-224/checkpoint-308.pth.tar
#  --drop-path ${DR}
