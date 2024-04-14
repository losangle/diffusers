#!/bin/bash

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="sddata/finetune/lora/fairness"

accelerate launch --mixed_precision="fp16"  train_fairness.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataloader_num_workers=8 \
  --resolution=128 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="One Single Melanoma on skin tissue" \
  --seed=1337 \
  --num_iters_per_epoch=10 \
  --guidance_scale=7
#  --report_to=wandb \
