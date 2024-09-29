#!/bin/bash
project=xAdaptive
version=v3
timestamp=$(date +"%Y-%m-%d")
log="log/x-v1-$timestamp-$project-$version.log"

exec &> >(tee $log)

accelerate launch --config_file gea/self/example/$project/$version/accelerate_ds_z3.yaml -m gea.self.example.$project.$version.xAdaptive_executor \
    --do_train \
    --do_eval \
    --epochs 8 \
    --max_steps 100 \
    --logging_steps 50 \
    --save_steps 50 \
    --eval_steps 200 \
    --save_strategy steps \
    --eval_strategy steps \
    --val_ratio 0.05 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --shuffle \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 8 \
    --path /data/lhz/manual/model/3E/7B/0911-v5 \
    --project /data/lhz/manual/model/3E/7B/0911-v5 \
    --tensorboard_project /data/lhz/manual/model/3E/TENSORBOARD/7b/latest/0911-v5 \
    --dataset /data/lhz/manual/datasets/dataset/dist/items-2048-v1 \
    --optim adamw \
    --lr 2.4e-4 \
    --lr_scheduler cosine_with_min_lr \
    --warmup_steps 500 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --mixed_precision bf16 \
    --save_total_limit 4 \
    --model llama2 \
    --ddp_timeout 7200 \