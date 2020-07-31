python train.py \
-exp_name soft+hard/xsum/transformer6_linear1024_accum5 \
-ext_layers 6 \
-ext_hidden_size 768 \
-ext_ff_size 1024 \
-mode train \
-ext_dropout 0.1 \
-lr 0.002 \
-visible_gpus 0 \
-report_every 50 \
-save_checkpoint_steps 1000 \
-batch_size 3000 \
-train_steps 50000 \
-accum_count 5 \
-use_interval true \
-warmup_steps 10000 \
-max_pos 512 \
-is_student true \
-use_soft_targets true \
-bert_data_path ../bert_data/bert_data_xsum/xsum \