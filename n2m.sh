#!/bin/bash

timestamp=$(date +"%Y-%m-%d")
log="log/n2m-$timestamp.log"

exec &> >(tee $log)

accelerate launch --config_file gea/self/accelerate_config.yaml -m gea.self.n2m_executor \
    --do_train \
    --epochs 1 \
    --max_steps -1 \
    --logging_steps 500 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --val_ratio 0.02 \
    --per_device_train_batch_size 4096 \
    --per_device_eval_batch_size 4096 \
    --gradient_accumulation_steps 1 \
    --shuffle \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --project /data/lhz/manual/model/3E/N2M/0819 \
    --tensorboard_project /data/lhz/manual/model/3E/TENSORBOARD/NEW/N2M/0819 \
    --optim adamw \
    --lr_scheduler cosine_with_min_lr \
    --lr 3e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --mixed_precision bf16 \
    --save_total_limit 4 \
    --dataset ultrachat_200k,Magicoder-Evol-Instruct-110K \
    --model llama2 \
    --with_sys_prompt false \
    --path /data/lhz/manual/model/3E/N2M/0819/ \