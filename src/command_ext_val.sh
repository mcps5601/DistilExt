python train.py \
-exp_name soft+hard/bert_emb/no_alpha/bert_transformer6/step50000 \
-mode validate \
-visible_gpus 0 \
-gpu_ranks 0 \
-batch_size 30000 \
-test_all \
-block_trigram true \