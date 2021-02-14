#!/bin/bash


DATASETS=(
    'None'     # dummy
    'tinyimagenet'
    

)
NUM_CLASSES=(
    0
    200
)

INIT_LR=(
    0
    1e-3
    
)
GPU_ID=0
ARCH='resnet50'
FINETUNE_EPOCHS=200

# ResNet50 pretrained on ImageNet
echo {\"imagenet\": \"0.7616\"} > logs/baseline_imagenet_acc_${ARCH}.txt

for TASK_ID in `seq 1 1`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main-Copy1.py \
        --arch $ARCH \
        --dataset ${DATASETS[TASK_ID]} --num_classes 200 \
        --lr 1e-1 \
        --weight_decay 4e-5 \
        --save_folder checkpoints/baseline/experiment2/$ARCH/${DATASETS[TASK_ID]} \
        --epochs $FINETUNE_EPOCHS \
        --mode finetune \
        --logfile logs/baseline_imagenet_acc_${ARCH}.txt \
        --use_imagenet_pretrained \
        --adv_train
done
