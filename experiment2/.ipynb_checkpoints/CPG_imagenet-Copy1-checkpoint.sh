#!/bin/bash

TARGET_TASK_ID=1
dataset=(
    'None'     # dummy
    'tiny-imagenet-200'
    'tiny-imagenet-200-1'
)
num_classes=(
    0
    200
    200    
)

init_lr=(
    0
    1e-3
    1e-3
)

pruning_lr=(
    0
    1e-3
    1e-3
)

GPU_ID=0
arch=(
    'resnet50'
    'resnet101'
    )
finetune_epochs=1

network_width_multiplier=1.0
pruning_ratio_interval=0.1
lr_mask=1e-4


for task_id in `seq $TARGET_TASK_ID $TARGET_TASK_ID`; do
    state=2
    while [ $state -eq 2 ]; do
        if [ "$task_id" != "1" ]
        then
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main-Copy1.py \
               --arch $arch \
               --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
               --lr ${init_lr[task_id]} \
               --lr_mask $lr_mask \
               --weight_decay 4e-5 \
               --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/scratch \
               --load_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id-1]}/gradual_prune \
               --epochs $finetune_epochs \
               --mode finetune \
               --network_width_multiplier $network_width_multiplier \
               --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
               --jsonfile logs/baseline_imagenet_acc_$arch.txt \
               --log_path checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/train.log \
               --adv_train
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main-Copy1.py \
               --arch $arch \
               --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
               --lr ${init_lr[task_id]} \
               --weight_decay 4e-5 \
               --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/scratch \
               --epochs $finetune_epochs \
               --mode finetune \
               --network_width_multiplier $network_width_multiplier \
               --jsonfile logs/baseline_imagenet_acc_$arch.txt \
               --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
               --use_imagenet_pretrained \
               --adv_train
        fi

        state=$?
        if [ $state -eq 2 ]
        then
            network_width_multiplier=1.5
            echo "New network_width_multiplier: $network_width_multiplier"
            break
            continue
        elif [ $state -eq 3 ]
        then
            echo "You should provide the baseline_cifar100_acc.txt as criterion to decide whether the capacity of network is enough for new task"
            exit 0
        fi
    done

    nrof_epoch=0
    if [ "$task_id" == "1" ]
    then
        nrof_epoch_for_each_prune=10
        pruning_frequency=1000
    else
        nrof_epoch_for_each_prune=20
        pruning_frequency=50
    fi
    start_sparsity=0.0
    end_sparsity=0.1
    nrof_epoch=$nrof_epoch_for_each_prune

    if [ $state -ne 5 ]
    then
        # gradually pruning
        CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main-Copy1.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]}  \
            --lr ${pruning_lr[task_id]} \
            --lr_mask 0.0 \
            --weight_decay 4e-5 \
            --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune \
            --load_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/scratch \
            --epochs $nrof_epoch \
            --mode prune \
            --initial_sparsity=$start_sparsity \
            --target_sparsity=$end_sparsity \
            --pruning_frequency=$pruning_frequency \
            --pruning_interval=4 \
            --jsonfile logs/baseline_imagenet_acc_$arch.txt \
            --network_width_multiplier $network_width_multiplier \
            --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
            --log_path checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/train.log \
            --adv_train

        for RUN_ID in `seq 1 2`; do
            nrof_epoch=$nrof_epoch_for_each_prune
            start_sparsity=$end_sparsity
            if [ $RUN_ID -lt 2 ]
            then
                end_sparsity=0.2
            else
                end_sparsity=0.15
            fi

            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main-Copy1.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
                --lr ${pruning_lr[task_id]} \
                --lr_mask 0.0 \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune \
                --load_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune \
                --epochs $nrof_epoch \
                --mode prune \
                --initial_sparsity=$start_sparsity \
                --target_sparsity=$end_sparsity \
                --pruning_frequency=$pruning_frequency \
                --pruning_interval=4 \
                --jsonfile logs/baseline_imagenet_acc_$arch.txt \
                --network_width_multiplier $network_width_multiplier \
                --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                --log_path checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/train.log \
                --adv_train
        done
    fi
done
