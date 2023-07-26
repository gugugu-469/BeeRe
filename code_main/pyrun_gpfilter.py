import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('./codes')
import os
import argparse
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
import shutil

from models import GPFilterModel
from trainer import GPFilterTrainer
from data import GPFilterDataset, GPFilterDataProcessor
from utils import init_logger, seed_everything, get_devices, get_time

import torch.utils.data as Data
from torch import nn
from d2l import torch as d2l
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForMaskedLM, RobertaModel, AlbertModel, AlbertTokenizerFast
MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertModel),
    'roberta': (AutoTokenizer, RobertaModel),
    'mcbert': (AutoTokenizer, AutoModelForMaskedLM),
    'albert':(AlbertTokenizerFast, AlbertModel)
}


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--method_name", default='gpfilter', type=str,
                        help="The name of method.")
    
    parser.add_argument("--data_dir", default='./ACE05-DyGIE/processed_data', type=str,
                        help="The task data directory.")
    
    parser.add_argument("--model_dir", default='/root/nas/Models', type=str,
                        help="The directory of pretrained models.")
    
    parser.add_argument("--model_type", default='bert', type=str, 
                        help="The type of selected pretrained models.")
    
    parser.add_argument("--pretrained_model_name", default='RoBERTa_zh_Large_PyTorch', type=str,
                        help="The path or name of selected pretrained models.")
    
    parser.add_argument("--finetuned_model_name", default='gpfilter', type=str,
                        help="The name of finetuned model.")
    
    parser.add_argument("--output_dir", default='./checkpoint', type=str,
                        help="The path of result data and models to be saved.")
    
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run the models in inference mode on the test set.")
    
    parser.add_argument("--do_filter", action="store_true",
                        help="Whether to run the models to filter results from different methods.")
    
    parser.add_argument("--model_version", default='', type=str,
                        help="model's version when do predict.")
    
    parser.add_argument("--result_output_dir", default='./result_output', type=str,
                        help="the directory of predict result to be saved.")
    
    parser.add_argument("--devices", default='-1', type=str,
                        help="devices id, -1 means using cpu.")
    
    parser.add_argument("--loss_show_rate", default=200, type=int,
                        help="liminate loss to [0,1] where show on the train graph.")
    
    parser.add_argument("--max_length", default=256, type=int,
                        help="the max length of sentence.")
    
    parser.add_argument("--train_batch_size", default=20, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate.")
    
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for, "
                             "E.g., 0.1 = 10% of training.")
    
    parser.add_argument("--earlystop_patience", default=100, type=int,
                        help="The patience of early stop")
    
    parser.add_argument('--logging_steps', type=int, default=400,
                        help="Log every X updates steps.")
    
    parser.add_argument('--seed', type=int, default=2021,
                        help="random seed for initialization")
    
    parser.add_argument("--save_metric", default='f1', type=str,choices=['p','r','f1','step','epoch','loss'],
                        help="the metric determine which model to save.")
    
    parser.add_argument('--do_rdrop', action="store_true",
                        help="whether to do r-drop")
    
    parser.add_argument('--rdrop_alpha', type=int, default=4,
                        help="hyper-parameter in rdrop")
    
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="dropout rate")
    
    parser.add_argument('--filter_threshold', type=float, default=0,
                        help="the threshold to filer results from different methods")
    
    parser.add_argument('--use_efficient_global_pointer', action="store_true",
                        help="whether to use Efficient Global Pointer")
    
    parser.add_argument('--inner_dim', type=int, default=64,
                        help="inner dim of gplinker")
    

    parser.add_argument('--with_type', action="store_true",
                        help="whether to use entity type information")

    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="value used to clip global grad norm")
    
    parser.add_argument('--model_version_1', type=str, default='',
                        help="modle version gpner")
    
    parser.add_argument('--model_version_2', type=str, default='',
                        help="model version gpner9")

    parser.add_argument('--predict_threshold', type=float, default=10,
                        help="fliter model's predict_threshold")

    parser.add_argument('--filter_head_threshold', type=float, default=-9.5,
                        help="entity head filter threshold")

    parser.add_argument('--filter_tail_threshold', type=float, default=-9.5,
                        help="entity tail filter threshold")
        
    args = parser.parse_args()
    args.devices = get_devices(args.devices.split(','))
    args.device = args.devices[0]
    args.distributed = True if len(args.devices) > 1  else False 
    seed_everything(args.seed)
    args.time = get_time(fmt='%m-%d-%H-%M')
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.method_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.pretrained_model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args.output_dir)
    args.train_result_output_dir = os.path.join(args.result_output_dir, args.method_name)
    args.predict_result_output_dir = os.path.join(args.result_output_dir, args.method_name,'gpf-'+args.model_version+'__gpner-'+args.model_version_1+'__gpner9-'+args.model_version_2)
    if args.do_train and args.do_predict:
        args.model_version = args.time
    if args.do_predict == True and args.model_version == '':
        raise Exception('You must give model version if you want to predict')    
    return args

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main():
    args = get_args()
    logger = init_logger(os.path.join(args.output_dir, 'log.txt'))
    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    if args.do_train:
        args.result_output_dir = args.train_result_output_dir
        if not os.path.exists(args.result_output_dir):
            os.makedirs(args.result_output_dir)
        makedirs = os.path.join(args.output_dir, args.model_version)
        if not os.path.exists(makedirs):
            os.makedirs(makedirs)
        logger.info(f'Training {args.finetuned_model_name} model...')
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.pretrained_model_name), do_lower_case=True)

        data_processor = GPFilterDataProcessor(args)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()
        train_dataset =GPFilterDataset(train_samples, data_processor, tokenizer, args, mode='train')
        eval_dataset = GPFilterDataset(eval_samples, data_processor, tokenizer, args, mode='eval')

        model = GPFilterModel(model_class, args, data_processor.schema)
        print(model)
        print('get_paras:{}'.format(get_parameter_number(model)))
        trainer = GPFilterTrainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                            logger=logger)

        global_step, best_step = trainer.train()
        
    if args.do_filter:
        args.result_output_dir = args.predict_result_output_dir
        if not os.path.exists(args.result_output_dir):
            os.makedirs(args.result_output_dir)
        load_dir = os.path.join(args.output_dir, args.model_version)
        logger.info(f'load tokenizer from {load_dir}')
        tokenizer = tokenizer_class.from_pretrained(load_dir)

        data_processor = GPFilterDataProcessor(args)
        test_samples = data_processor.get_test_sample()
        test_dataset = GPFilterDataset(test_samples, data_processor, tokenizer=tokenizer, mode='test',
                                 args=args)
        model = GPFilterModel(model_class, args, data_processor.schema)
        # print(model)
        trainer = GPFilterTrainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, logger=logger,test_dataset = test_dataset)
        trainer.load_checkpoint()
        trainer.predict_filter()


if __name__ == '__main__':
    main()





