python train.py \
-exp_name get_xsum_soft \
-bert_data_path ../bert_data/bert_data_xsum/xsum \
-soft_targets_folder soft_targets_large \
-dump_mode train \
-ext_layers 2 \
-mode get_soft \
-visible_gpus 1 \
-gpu_ranks 0 \
-test_batch_size 1 \
-test_from  \
-use_soft_targets False \
-is_student False \

