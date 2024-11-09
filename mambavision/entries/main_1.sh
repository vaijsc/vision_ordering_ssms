#!/bin/bash
#SBATCH --job-name=coc
#SBATCH --error=/home/anhnd81/anhnd81/workspace/mambavision_1/mambavision/result/mambaV_coc.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
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
# EXP=ord1
# EXP=ord1_1
# EXP=ord_zigzag
EXP=coc
# EXP=ord1_1ss
# EXP=Test
LR=8e-4 # original 8e-4
WD=0.05
WR_LR=1e-6
DR=0.2
MESA=0.25
#RUN_FILE="/home/ubuntu/workspace/mambavision_1/mambavision/train_attn.py"
RUN_FILE="/home/anhnd81/anhnd81/workspace/mambavision_1/mambavision/train.py"
# checkpoint="/home/anhnd81/anhnd81/workspace/mambavision_1/output/train/perm1/20240908-234939-mamba_vision_T-224/last.pth.tar"
torchrun --master-port=12380 --nproc_per_node=4 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
 --data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR #\
#  --resume /home/anhnd81/anhnd81/workspace/mambavision_1/output/train/ord_zigzag/20241021-110627-mamba_vision_T-224/checkpoint-180.pth.tar
#--drop-path ${DR} 

#ord ordering inside mambamixer block, use torch.topk with order token is mean()
#ord1-1 ordering inside mambamixer block, use torch.topk with order token is mean(), for each mambamixer block
#ord1ss change to use softsort
#ord-zigzag 