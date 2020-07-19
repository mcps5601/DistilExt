python train.py \
-exp_name soft+hard/xsum/bert_ext \
-ext_layers 2 \
-ext_ff_size 2048 \
-mode validate \
-visible_gpus 1 \
-gpu_ranks 0 \
-batch_size 30000 \
-test_all \
-block_trigram true \
