#!/bin/bash

# SeeSR Log-Scale Training Script
# 基于原始SeeSR训练指令，适配log-scale修改

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/seesr_logscale" \
--root_folders 'preset/datasets/train_datasets/training_for_seesr' \
--ram_ft_path 'preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=256 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--gradient_checkpointing \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=10000 \
--max_train_steps=100000 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--seed=42

echo "Training completed!"
