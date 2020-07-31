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
For the steps of data preprocessing, please visit [PreSumm](https://github.com/nlpyang/PreSumm) for more information. \
We provide our pre-processed data here.

### CNN/DailyMail
[CNN/DM](https://drive.google.com/file/d/1b9CpjMM_qFZMxJS8rgpNSYTwv2tm6smc/view?usp=sharing)

[CNN/DM (soft_targets)](https://drive.google.com/file/d/1hA3QiJj3YNzGS9Bp3dAQaAs7zKAS9AY0/view?usp=sharing)

### XSum
[XSum](https://drive.google.com/file/d/1sVQEDfkl0VzjInXgF9DhWGZWNzjnu05c/view?usp=sharing)

[XSum (soft_targets)](https://drive.google.com/file/d/1rzjs0dUK2YXu3SnfUnjt-AHeHsVxynZR/view?usp=sharing)



## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

#### CNN/DM
##### To train a student
```
bash cnndm_train.sh
```
#### XSum
##### To train a teacher
```
bash xsum_train.sh
```
##### To train a student
```
bash xsum_train_stu.sh
```

## Model Evaluation
### CNN/DM
```
# this shell script will validate all the saved model steps during training
bash cnndm_val.sh
```
### XSum
```
# this shell script will validate all the saved model steps during training
bash xsum_val.sh
```

* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)

