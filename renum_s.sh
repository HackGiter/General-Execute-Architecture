#!/bin/bash
project=gpt
version=s
date=0826_v7
timestamp=$(date +"%Y-%m-%d")
log="log/$project-$timestamp-$version-$date.log"

exec &> >(tee $log)

accelerate launch --config_file gea/self/$project/$version/accelerate_config.yaml -m gea.self.$project.$version.gpt_executor \
    --do_train \
    --epochs 1 \
    --max_steps -1 \
    --logging_steps 200 \
    --save_steps 16000 \
    --eval_steps 4000 \
    --save_strategy steps \
    --eval_strategy steps \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --shuffle \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 8 \
    --project /data/lhz/manual/model/3E/$project/$date \
    --tensorboard_project /data/lhz/manual/model/3E/TENSORBOARD/NEW/$project/$date \
    --optim adamw \
    --lr_scheduler cosine_with_min_lr \
    --lr 2e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --mixed_precision bf16 \
    --save_total_limit 4 \
    --dataset /data/lhz/manual/datasets/renum_v5/s/train/data.jsonl \
    --eval_dataset /data/lhz/manual/datasets/renum_v5/s/eval/data.jsonl \
    --with_sys_prompt false \
    --path /data/lhz/manual/model/3E/$project/$date/ \