conda activate mambav
cd /root/workspace/mambavision_1/mambavision
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=3,4,5,6
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
DATA_PATH="/root/workspace/dataset/ImageNet1K/data"
MODEL=mamba_vision_T
BS=128
#EXP=perm1
EXP=Test
LR=8e-4 # original 8e-4
WD=0.05
WR_LR=1e-6
DR=0.2 # Tiny
MESA=0.25
#RUN_FILE="/home/ubuntu/workspace/mambavision_1/mambavision/train_attn.py"
RUN_FILE="/root/workspace/mambavision_1/mambavision/train_perm1.py"
# checkpoint="/home/anhnd81/anhnd81/workspace/mambavision_1/output/train/perm1/20240908-234939-mamba_vision_T-224/last.pth.tar"
torchrun --master-port=12365 --nproc_per_node=4 $RUN_FILE --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
--data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR # \
# --resume /home/anhnd81/anhnd81/workspace/mambavision_1/output/train/perm1/20240910-162332-mamba_vision_T-224/checkpoint-307.pth.tar
