#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os, json
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext, extract_soft

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_name", type=str, default='soft+hard/bert_emb/no_alpha/', help='set the name of experiment at each time')
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'get_soft'])
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=True)
    # batch_size: max number of words when training
    parser.add_argument("-batch_size", default=3000, type=int)
    parser.add_argument("-test_batch_size", default=1, type=int)
    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-ngram_blocking", default=4, type=int)
    parser.add_argument("-label_smoothing", default=0.1, type=float) 


    ############ params for PATHs ############
    parser.add_argument("-bert_data_path", default='../bert_data/bert_data_cnndm/cnndm')
    # parser.add_argument("-bert_data_path", default='../bert_data/bert_data_xsum/xsum')
    ##### uncomment the line below to extract soft labels
    # parser.add_argument("-test_from", default='/data/PreSumm/src/MODEL_PATH/trained_cnndm_ext/model_step_18000.pt')
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-train_from", default='')
    ############ params for PATHs ############


    ############ params for the extractive summarization task ############
    parser.add_argument("-ext_dropout", default=0.1, type=float)
    parser.add_argument("-ext_layers", default=6, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)
    #parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'], help='use pretrained BERT or not')
    ############ params for the extractive summarization task ############


    ############ params for knowledge distillation ############
    parser.add_argument("-use_soft_targets", type=str2bool, nargs='?',const=True, default=True, help='use both hard target and soft target as objective')
    #parser.add_argument("-distill_alpha", default=0.6, type=float, help='the hyperparameter for adjusting the ratio between the hard and soft loss')
    parser.add_argument("-is_student", type=str2bool, nargs='?',const=True, default=True, help='to use the student model')
    ############ params for knowledge distillation ############


    ########## 這區的參數不用調 ##########
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)
    ##### params for reporter
    parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-train_steps", default=50000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    ########## 這區的參數不用調 ##########


    ########## params for optimization ##########
    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr", default=0.002, type=float)
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=10000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    ########## params for optimization ##########


    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    log_path = os.path.join(os.path.split(args.log_file)[0], args.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    init_logger(os.path.join(log_path, os.path.split(args.log_file)[1]))

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
        elif (args.mode == 'lead'):
            baseline(args, cal_lead=True)
        elif (args.mode == 'oracle'):
            baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            model_save_path = os.path.join(args.model_path, args.exp_name)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            with open(os.path.join(model_save_path, 'command_args.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            if (args.use_soft_targets):
                args.bert_data_path = os.path.join(os.path.split(args.bert_data_path)[0], 'soft_targets',
                                                    os.path.split(args.bert_data_path)[1])
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)
        elif (args.mode == 'get_soft'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            extract_soft(args, device_id, cp, step)
