python train.py \
-exp_name soft+hard/bert_emb/no_alpha/bert_transformer6_linear1024_accum2 \
-ext_layers 6 \
-ext_ff_size 1024 \
-mode validate \
-visible_gpus 0 \
-gpu_ranks 0 \
-batch_size 30000 \
-test_all \
-block_trigram true \
