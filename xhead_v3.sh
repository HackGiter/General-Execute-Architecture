#!/bin/bash
project=xhead
version=v3
timestamp=$(date +"%Y-%m-%d")
log="log/xhead-$timestamp-$project-$version.log"

exec &> >(tee $log)

accelerate launch --config_file gea/self/$project/$version/accelerate_config.yaml -m gea.self.$project.$version.xhead_executor \
    --do_train \
    --epochs 5 \
    --max_steps -1 \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 1000 \
    --save_strategy steps \
    --eval_strategy steps \
    --val_ratio 0.04 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --shuffle \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 8 \
    --project /data/lhz/manual/model/3E/$project/0819 \
    --tensorboard_project /data/lhz/manual/model/3E/TENSORBOARD/NEW/$project/0819 \
    --optim adamw \
    --lr_scheduler cosine_with_min_lr \
    --lr 2e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --mixed_precision bf16 \
    --save_total_limit 4 \
    --dataset /data/lhz/manual/datasets/dataset/dist/items-2048-v1 \
    --model llama2 \
    --with_sys_prompt false \
    --path /data/lhz/manual/model/3E/$project/0819/ \