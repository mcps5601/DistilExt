#!/bin/bash

python train.py \
-exp_name hard/xsum/bert_ext_accum4 \
-ext_layers 2 \
-ext_hidden_size 768 \
-ext_ff_size 2048 \
-mode train \
-ext_dropout 0.1 \
-lr 0.002 \
-visible_gpus 0 \
-report_every 50 \
-save_checkpoint_steps 1000 \
-batch_size 3000 \
-train_steps 50000 \
-accum_count 4 \
-use_interval true \
-warmup_steps 10000 \
-max_pos 512 \
-is_student false \
-use_soft_targets false \
-bert_data_path ../bert_data/bert_data_xsum/xsum \