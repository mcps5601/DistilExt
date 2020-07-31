# DistilExt

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.5.0 pytorch_transformers tensorboardX multiprocess pyrouge


Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)


## Trained Teacher Models
[CNN/DM BertExt](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)
[XSum BertExt](https://drive.google.com/file/d/1sZ8OoUL_GDoiV-FjKlbXH5s19rEN5gbo/view?usp=sharing)


## Trained Student Models
[CNN/DM DistilExt (8-layer Transformer)]()

[XSum DistilExt (6-layer Transformer)](https://drive.google.com/file/d/1vU8HQeOyd8BXDvvUXaGzWQuX0HSBsQTk/view?usp=sharing)


## Data Preparation 
For the steps of data preprocessing, please visit [PreSumm](https://github.com/nlpyang/PreSumm) for more information.
We provide our pre-processed data here.

### CNN/DailyMail
[CNN/DM](https://drive.google.com/file/d/1b9CpjMM_qFZMxJS8rgpNSYTwv2tm6smc/view?usp=sharing)

[CNN/DM (soft_targets)](https://drive.google.com/file/d/1hA3QiJj3YNzGS9Bp3dAQaAs7zKAS9AY0/view?usp=sharing)

### XSum
[XSum](https://drive.google.com/file/d/1sVQEDfkl0VzjInXgF9DhWGZWNzjnu05c/view?usp=sharing)

[XSum (soft_targets)](https://drive.google.com/file/d/1rzjs0dUK2YXu3SnfUnjt-AHeHsVxynZR/view?usp=sharing)



## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Extractive Setting

```
python train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512
```

### Abstractive Setting

#### TransformerAbs (baseline)
```
python train.py -mode train -accum_count 5 -batch_size 300 -bert_data_path BERT_DATA_PATH -dec_dropout 0.1 -log_file ../../logs/cnndm_baseline -lr 0.05 -model_path MODEL_PATH -save_checkpoint_steps 2000 -seed 777 -sep_optim false -train_steps 200000 -use_bert_emb true -use_interval true -warmup_steps 8000  -visible_gpus 0,1,2,3 -max_pos 512 -report_every 50 -enc_hidden_size 512  -enc_layers 6 -enc_ff_size 2048 -enc_dropout 0.1 -dec_layers 6 -dec_hidden_size 512 -dec_ff_size 2048 -encoder baseline -task abs
```
#### BertAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../logs/abs_bert_cnndm
```
#### BertExtAbs
```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3 -log_file ../logs/abs_bert_cnndm  -load_from_extractive EXT_CKPT   
```
* `EXT_CKPT` is the saved `.pt` checkpoint of the extractive model.




## Model Evaluation
### CNN/DM
```
 python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm 
```
### XSum
```
 python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -min_length 20 -max_length 100 -alpha 0.9 -result_path ../logs/abs_bert_cnndm 
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)

