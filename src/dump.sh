python train.py \
-exp_name get_xsum_soft \
-bert_data_path ../bert_data/bert_data_xsum/xsum \
-ext_layers 2 \
-mode get_soft \
-visible_gpus 0 \
-gpu_ranks 0 \
-test_batch_size 1 \
-test_from /data/PreSumm/src/MODEL_PATH/trained_cnndm_ext/model_step_18000.pt \
-use_soft_targets False \
-is_student False \

