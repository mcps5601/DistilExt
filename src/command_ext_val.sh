python train.py \
-exp_name soft+hard/bert_emb/no_alpha/bert_transformer5_linear1024 \
-ext_layers 5 \
-ext_ff_size 1024 \
-mode validate \
-visible_gpus 0 \
-gpu_ranks 0 \
-batch_size 30000 \
-test_all \
-block_trigram true \
