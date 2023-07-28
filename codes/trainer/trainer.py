import torch
import torch.nn as nn
import os
import json
import jsonlines
import shutil
import math
import numpy as np
import torch.nn.functional as F
from d2l import torch as d2l
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import ProgressBar, TokenRematch, get_time, save_args, SPO, ACESPO
from loss import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy
from optimizer import GPLinkerOptimizer
import time
from tqdm import tqdm
from copy import deepcopy

class Trainer(object):
    def __init__(
            self,
            args,
            data_processor,
            logger,
            model=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
            test_dataset=None
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        if test_dataset is not None and isinstance(test_dataset, Dataset):
            self.test_dataset = test_dataset

        self.logger = logger

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        epoch_best_f1 = 0
        self.output_dir = os.path.join(args.output_dir, args.model_version)

        
        if args.distributed == True:
            model = nn.DataParallel(model, device_ids=args.devices).to(args.device)
        else:
            model.to(args.device)
            
        
        
        train_dataloader = self.get_train_dataloader()
        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)

        optimizer = GPLinkerOptimizer(args, model, train_steps= len(train_dataloader)  * args.epochs)
        
        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = 0
        cnt_patience = 0
        
        animator = d2l.Animator(xlabel='epoch', xlim=[0, args.epochs], ylim=[0, 1], fmts=('k-', 'r--', 'y-.', 'm:', 'g--', 'b-.', 'c:'),
                                legend=[f'train loss/{args.loss_show_rate}', 'train_p', 'train_r', 'train_f1', 'val_p', 'val_r', 'val_f1'])
        # 统计指标
        metric = d2l.Accumulator(5)
        num_batches = len(train_dataloader)
        
        
        all_times = []
        for epoch in range(args.epochs):
            print('Now Epoch:{}'.format(epoch))
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            start_time = time.time()
            for step, item in enumerate(train_dataloader):
                loss, train_p, train_r, train_f1 = self.training_step(model, item)
                loss = loss.item()
                metric.add(loss, train_p, train_r, train_f1, 1)
                pbar(step, {'loss': loss})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    val_p, val_r, val_f1 = self.evaluate(model)
                    print('\nevaluate finish:p:{}\tr:{}\tf1:{}\n'.format(val_p,val_r,val_f1))
                    animator.add(
                        global_step / num_batches, 
                        (# metric[0] / metric[-1] / args.loss_show_rate, # loss太大，除以loss_show_rate才能在[0,1]范围内看到
                            loss / args.loss_show_rate,
                            train_p,  # metric[1] / metric[-1],
                            train_r,  # metric[2] / metric[-1],
                            train_f1, # metric[3] / metric[-1],
                            val_p,
                            val_r,
                            val_f1))
                    if not os.path.exists(self.output_dir):
                        os.makedirs(self.output_dir)
                    d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)

                    if args.save_metric == 'step':
                        save_metric = global_step
                    elif args.save_metric == 'epoch':
                        save_metric = epoch
                    elif args.save_metric == 'loss':
                        # e的700次方刚好大于0，不存在数值问题
                        # 除以10，避免loss太大，exp(-loss)次方由于数值问题会小于0，导致存不上，最大可以处理7000的loss
                        save_metric = math.exp(- loss / 10) # math.exp(- metric[0] / metric[-1] / 10)
                    elif args.save_metric == 'p':
                        save_metric = val_p
                    elif args.save_metric == 'r':
                        save_metric = val_r
                    elif args.save_metric == 'f1':
                        save_metric = val_f1

                    if save_metric > best_score:
                        best_score = save_metric
                        best_step = global_step
                        cnt_patience = 0
                        self.args.loss = loss # metric[0] / metric[-1]
                        self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                                            #  metric[1] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1]
                        self.args.val_p, self.args.var_r, self.args.val_f1 = val_p, val_r, val_f1
                        print('find best metric, save checkpoint')
                        self._save_checkpoint(model)
                    else:
                        cnt_patience += 1
                        self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                        if cnt_patience >= self.args.earlystop_patience:
                            break
            end_time = time.time()
            use_time = end_time-start_time
            all_times.append(use_time)
            logger.info('One epoch train finish ,use {} seconds'.format(use_time))
            if cnt_patience >= args.earlystop_patience:
                break
            self.args.loss = loss
            self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
            self._save_checkpoint(model)
        logger.info('all epochs finished , al times:{}'.format(all_times))
        logger.info(f"\n***** {args.finetuned_model_name} model training stop *****" )
        logger.info(f'finished time: {get_time()}')
        logger.info(f"best val_{args.save_metric}: {best_score}, best step: {best_step}\n" )
        return global_step, best_step

    def predict(self):
        raise NotImplementedError

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model):
        args = self.args
        
        if args.distributed:
            model=model.module
        # 防止91存到3卡，但是82没有3卡的情况
        model = model.to(torch.device('cpu'))
        torch.save(model.state_dict(), os.path.join(self.output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', self.output_dir)
        self.tokenizer.save_vocabulary(save_directory=self.output_dir)
        model = model.to(args.device)
        save_args(args, self.output_dir)
        shutil.copyfile(os.path.join(args.model_dir, args.pretrained_model_name, 'config.json'),
                        os.path.join(self.output_dir, 'config.json'))

    def _save_best_epoch_checkpoint(self, model):
        args = self.args
        out = os.path.join(self.output_dir,'best')
        if not os.path.exists(out):
            os.makedirs(out)
        if args.distributed:
            model=model.module
        # 防止91存到3卡，但是82没有3卡的情况
        model = model.to(torch.device('cpu'))
        torch.save(model.state_dict(), os.path.join(out, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', out)
        self.tokenizer.save_vocabulary(save_directory=out)
        model = model.to(args.device)
        save_args(args, out)
        shutil.copyfile(os.path.join(args.model_dir, args.pretrained_model_name, 'config.json'),
                        os.path.join(out, 'config.json'))
    
    
    def load_checkpoint(self):
        args = self.args
        load_dir = os.path.join(args.output_dir, args.model_version)
        self.logger.info(f'load model from {load_dir}')
        # 每次加载到cpu中，防止爆显存
        checkpoint = torch.load(os.path.join(load_dir, 'pytorch_model.pt'), map_location=torch.device('cpu'))
        if 'module' in list(checkpoint.keys())[0].split('.'):
            self.model = nn.DataParallel(self.model, device_ids=args.devices).to(args.device)
        self.model.load_state_dict(checkpoint)
    
    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        collate_fn = self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False if self.args.do_rdrop else True,
            collate_fn=collate_fn
        )

    def get_eval_dataloader(self):
        collate_fn = self.eval_dataset.collate_fn if hasattr(self.eval_dataset, 'collate_fn') else None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_test_dataloader(self, batch_size=None):
        collate_fn = self.test_dataset.collate_fn_test if hasattr(self.test_dataset, 'collate_fn_test') else None
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

class GPFilterTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            test_dataset = None,
            ngram_dict=None
    ):
        super(GPFilterTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset = test_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids)

        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits2, mask_zero=True)
        loss = sum([loss1, loss2]) / 2
        
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits1[::2],dim=-1), F.softmax(logits1[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits1[1::2],dim=-1), F.softmax(logits1[::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[::2],dim=-1), F.softmax(logits2[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[1::2],dim=-1), F.softmax(logits2[::2],dim=-1), reduction='sum')
            # ’/ 4 * self.args.rdrop_alpha‘三是公式里带的, '/ 2'是为了头尾求平均
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits1.shape[0] / 2
        
        loss.backward()

        p1, r1, f11 = self.cal_prf(logits1, batch_head_labels)
        p2, r2, f12 = self.cal_prf(logits2, batch_tail_labels)
        p = (p1 + p2) / 2 
        r = (r1 + r2) / 2
        f1 = (f11 + f12) / 2
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        all_correct = 0
        all_ypred = 0
        all_ytrue = 0
        with torch.no_grad():
            for step, item in enumerate(eval_dataloader):
                pbar(step)
                model.eval()
                batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels = item
                batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels = \
                        batch_token_ids.to(device), batch_mask_ids.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
                logits1, logits2 = model(batch_token_ids, batch_mask_ids)
                correct,ypred,ytrue = self.get_ytrue_ypred(logits2, batch_tail_labels)
                all_correct += correct
                all_ypred += ypred
                all_ytrue += ytrue
                correct,ypred,ytrue = self.get_ytrue_ypred(logits1, batch_head_labels)
                all_correct += correct
                all_ypred += ypred
                all_ytrue += ytrue
        p = all_correct / all_ypred if all_ypred != 0 else 0
        r = all_correct / all_ytrue if all_ytrue != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1

    def get_ytrue_ypred(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        y_pred = torch.greater(y_pred, 0)
        return torch.sum(y_true * y_pred).item(), torch.sum(y_pred).item(), torch.sum(y_true).item()

    def cal_prf(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1

    def predict_filter(self):
        args = self.args
        logger = self.logger
        model = self.model
        output_dir = os.path.join('./result_output', 'filter_ace','gpf-'+args.model_version+'__gpner-'+args.model_version_1+'__gpner9-'+args.model_version_2)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = output_dir + '/test.jsonl'
        self.max_length = args.max_length
        data_processor = self.data_processor
#         schema = data_processor.schema
        schema = data_processor.predicate2id
        tokenizer = self.tokenizer
        device = args.device
#         num_examples = 4482
        id2predicate = data_processor.id2predicate

        start_time = time.time()
        test_dataloader = self.get_test_dataloader()
        num_examples = len(test_dataloader.dataset)
        logger.info("***** Running Testing *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Testing')
        device = self.args.device
        model.to(device)
        model.eval()
        predict_datas = []
        with torch.no_grad():
            for step, item in enumerate(test_dataloader):
                pbar(step)
                
                text_list,spo_list,out_list = item
                encoder_text = self.tokenizer(text_list, max_length=self.max_length, truncation=True,padding=True,return_tensors='pt')
                input_ids = encoder_text['input_ids']
                attention_mask = encoder_text['attention_mask']
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                score = model(input_ids, attention_mask)
                
                sentence_index_list = []
                p_list = []
                sh_list = []
                oh_list = []
                st_list = []
                ot_list = []
                get_spo = []
                for i in range(len(spo_list)):
                    item_spo = spo_list[i]
                    for spo in item_spo:
                        relation_key = spo['subject_type'] + "_" + spo['predicate'] + '_' + spo['object_type']
                        if relation_key not in schema:
                            continue
                        p = schema[relation_key]
                        sh = spo['subject_h']
                        st = spo['subject_t']
                        oh = spo['object_h']
                        ot = spo['object_t']
                        sentence_index_list.append(i)
                        p_list.append(p)
                        sh_list.append(int(sh))
                        st_list.append(int(st))
                        oh_list.append(int(oh))
                        ot_list.append(int(ot))  
                        get_spo.append({'text_index':i,'spo':spo})
                
                p_sh_oh_list = [sentence_index_list,p_list,sh_list,oh_list]
                p_st_ot_list = [sentence_index_list,p_list,st_list,ot_list]
                sh_oh_output = score[0].cpu()
                st_ot_output = score[1].cpu()
                sh_oh_check = (sh_oh_output[p_sh_oh_list]>args.filter_head_threshold).int()
                st_ot_check = (st_ot_output[p_st_ot_list]>args.filter_tail_threshold).int()
                final_check = sh_oh_check*st_ot_check
                for select_index in torch.where(final_check>0)[0]:
                    select_spo = get_spo[select_index]
                    out_list[select_spo['text_index']]['spo_list'].append(select_spo['spo'])
                    
                        
                predict_datas.extend(out_list)
        with jsonlines.open(output_dir, mode='w') as f:
            for data in predict_datas:
                f.write(data)
        end_time = time.time()
        print('cost time:{} seconds'.format(end_time-start_time))
        self.get_res_prf(output_dir)

    def get_res_prf(self,read_path):
        gold_path = os.path.join(self.args.data_dir,'test.json')
        # 黄金数据
        all_gold_jsons=[]
        with open(gold_path, 'r') as f_1:
            lines = f_1.readlines()
            for line in lines:
                all_gold_jsons.append(json.loads(line))
        gold_spos=[]
        for i in range(len(all_gold_jsons)):
            gold_json=all_gold_jsons[i]
            spo_list=gold_json['spo_list']
            for spo in spo_list:
                # print('spo:{}'.format(spo))
                if self.args.with_type:
                    gold_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['subject_type'].strip(),spo['object'].strip(),spo['object_type'].strip()))
                else:
                    gold_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['object'].strip()))

        #获取预测数据
        all_predict_jsons=[]
        with open(read_path, 'r') as f_2:
            lines = f_2.readlines()
            for line in lines:
                all_predict_jsons.append(json.loads(line))
        predict_spos=[]
        for i in range(len(all_predict_jsons)):
            predict_json=all_predict_jsons[i]
            spo_list=predict_json['spo_list']
            for spo in spo_list:
                if self.args.with_type:
                    predict_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['subject_type'].strip(),spo['object'].strip(),spo['object_type'].strip()))
                else:
                    predict_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['object'].strip()))

        # 计算pre,rec,f1
        P = len(set(predict_spos) & set(gold_spos)) / len(set(predict_spos)) if len(set(predict_spos)) != 0 else 0
        R = len(set(predict_spos) & set(gold_spos)) / len(set(gold_spos)) if len(set(gold_spos)) != 0 else 0
        F = (2 * P * R) / (P + R) if P+R != 0 else 0
        print('\nRESULT PRF：\n')
        print(str(round(P*100,2))+'|'+str(round(R*100,2))+'|'+str(round(F*100,2))+'|'+'\n')
            

class GPNERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            test_dataset=None,
            ngram_dict=None
    ):
        super(GPNERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_entity_labels = item
        logits = model(batch_token_ids, batch_mask_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        loss.backward()
        p, r, f1 = self.cal_prf(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def cal_prf(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        all_correct = 0
        all_ypred = 0
        all_ytrue = 0
        with torch.no_grad():
            for step, item in enumerate(eval_dataloader):
                pbar(step)
                model.eval()
                batch_token_ids, batch_mask_ids, batch_entity_labels = item
                batch_token_ids, batch_mask_ids, batch_entity_labels = \
                        batch_token_ids.to(device), batch_mask_ids.to(device), batch_entity_labels.to(device)
                logits = model(batch_token_ids, batch_mask_ids)
                correct,ypred,ytrue = self.get_ytrue_ypred(logits, batch_entity_labels)
                all_correct += correct
                all_ypred += ypred
                all_ytrue += ytrue
        p = all_correct / all_ypred if all_ypred != 0 else 0
        r = all_correct / all_ytrue if all_ytrue != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def get_ytrue_ypred(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        y_pred = torch.greater(y_pred, 0)
        return torch.sum(y_true * y_pred).item(), torch.sum(y_pred).item(), torch.sum(y_true).item()

    def predict(self):
        start_time = time.time()
        args = self.args
        logger = self.logger
        model = self.model
        id2class = self.data_processor.id2class
        self.max_length = args.max_length
        test_dataloader = self.get_test_dataloader()
        num_examples = len(test_dataloader.dataset)
        logger.info("***** Running Entity Extraction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Testing')
        device = self.args.device
        model.to(device)
        model.eval()
        predict_datas = []
        with torch.no_grad():
            for step, item in enumerate(test_dataloader):
                pbar(step)
                text_list = item
                texts = [text_list_item['text'] for text_list_item in text_list]
                encoder_text = self.tokenizer(texts, return_offsets_mapping=True, max_length=self.max_length, truncation=True,padding=True,return_tensors='pt')
                input_ids = encoder_text['input_ids']
                attention_mask = encoder_text['attention_mask']
                # 计算要mask的
                valid_length = (torch.sum(attention_mask,dim=-1)-1).tolist()
                text_num = len(input_ids)
                valid_length = [[0]*text_num,valid_length]
                valid_index = list(range(text_num))
                offset_mapping = encoder_text['offset_mapping']
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                multi_mask = attention_mask.unsqueeze(dim=-1).unsqueeze(dim=1)
                # 拿到输出
                # 根据长度 ，第一个和最后一个都减inf, 乘上attention_mask
                score = model(input_ids, attention_mask)
                outputs = (score*multi_mask).data.cpu().numpy()
                outputs[valid_index,:,valid_length,:] -= np.inf
                outputs[valid_index,:,:,valid_length] -= np.inf
                # 获取大于0的
                for text_index,entity_type, h, t in zip(*np.where(outputs > 0)):
                    text = text_list[text_index]['text']
                    text_offset_mapping = offset_mapping[text_index]
                    #解码到text
                    text_list[text_index]['entity_list'].append({'entity':text[text_offset_mapping[h][0]:text_offset_mapping[t][-1]], 'entity_type':id2class[entity_type],'h':str(h), 't':str(t)})
                # 加入最终数据
                predict_datas.extend(text_list)
        # entity抽取完毕，输出
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            for data in predict_datas:
                f.write(data)
        end_time = time.time()
        print('cost time:{} seconds'.format(end_time-start_time))
        self.predict_XP2X(output_dir,from_entity=True)


    def collate_fn_test(self,examples):
        text_list = []
        for item in examples:
            text = item
            text_list.append(text)

        return text_list
    
    def predict_XP2X(self,read_path,from_entity=False):
        start_time = time.time()
        args = self.args
        logger = self.logger
        model = self.model
        id2class = self.data_processor.id2class
        self.max_length = args.max_length
        predict_datas = []
        with jsonlines.open(read_path, mode='r') as lines:
            for line in lines:
                if 'spo_list' not in line.keys():
                    line['spo_list'] = []
                predict_datas.append(line)  
        processed_samples = []
        task = 'SP2O' if args.finetuned_model_name=='gpner' else 'OP2S'
        pbar2 = ProgressBar(n_total=len(predict_datas), desc='GET DATA')
        for index,sample in enumerate(predict_datas):
            pbar2(index)
            text = sample['text']
            if from_entity:
                for entity_dic in sample['entity_list']:
                    entity = entity_dic['entity']
                    h = entity_dic['h']
                    t = entity_dic['t']
                    entity_type = entity_dic['entity_type'] if args.with_type else ''
                    for predicate in self.data_processor.predicates:
                        prefix = self.data_processor.add_prefix(text, entity, predicate)
                        prefix_encode_length = len(self.tokenizer(prefix,add_special_tokens=False)['input_ids'])
                        processed_samples.append({'text': prefix+text, 'entity': entity, 'entity_type': entity_type, 'predicate': predicate, 'index':index,'h':h, 't':t,'prefix_encode_length':prefix_encode_length})
            else:
                for spo in sample['spo_list']:
                    if task == 'SP2O':
                        entity = spo['subject']
                        h = spo['subject_h']
                        t = spo['subject_t']
                        entity_type = spo['subject_type'] if args.with_type else ''
                    else:
                        entity = spo['object']
                        h = spo['object_h']
                        t = spo['object_t']
                        entity_type = spo['object_type'] if args.with_type else ''
                    predicate = spo['predicate']
                    prefix = self.data_processor.add_prefix(text, entity, predicate)
                    prefix_encode_length = len(self.tokenizer(prefix,add_special_tokens=False)['input_ids'])
                    processed_samples.append({'text': prefix+text, 'entity': entity, 'entity_type': entity_type, 'predicate': predicate, 'index':index,'h':h, 't':t,'prefix_encode_length':prefix_encode_length})


        print('\n构造完毕,共:{}条'.format(len(processed_samples)))
        test_dataloader_2 =  DataLoader(
            processed_samples,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_test
        )
        num_examples = len(test_dataloader_2.dataset)
        logger.info("***** Running {} *****".format(task))
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader_2), desc='Testing')
        device = self.args.device
        model.to(device)
        model.eval()
        with torch.no_grad():
            for step, item in enumerate(test_dataloader_2):
                pbar(step)
                text_list = item
                texts = [text_list_item['text'] for text_list_item in text_list]
                encoder_text = self.tokenizer(texts, return_offsets_mapping=True, max_length=self.max_length, truncation=True,padding=True,return_tensors='pt')
                input_ids = encoder_text['input_ids']
                attention_mask = encoder_text['attention_mask']
                # 计算要mask的
                valid_length = (torch.sum(attention_mask,dim=-1)-1).tolist()
                text_num = len(input_ids)
                valid_length = [[0]*text_num,valid_length]
                valid_index = list(range(text_num))
                offset_mapping = encoder_text['offset_mapping']
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                multi_mask = attention_mask.unsqueeze(dim=-1).unsqueeze(dim=1)
                # 拿到输出
                # 根据长度 ，第一个和最后一个都减inf, 乘上attention_mask
                score = model(input_ids, attention_mask)
                outputs = (score*multi_mask).data.cpu().numpy()
                outputs[valid_index,:,valid_length,:] -= np.inf
                outputs[valid_index,:,:,valid_length] -= np.inf
                # 获取大于0的
                for text_index,entity_type, h, t in zip(*np.where(outputs > 0)):
                    data = text_list[text_index]
                    text = data['text']
                    text_offset_mapping = offset_mapping[text_index]
                    pre_entity = data['entity']
                    predicate = data['predicate']
                    pre_entity_type = data['entity_type']
                    ori_h = data['h']
                    ori_t = data['t']
                    prefix_encode_length = data['prefix_encode_length']
                    now_h = str(h-prefix_encode_length)
                    now_t = str(t-prefix_encode_length)
                    #解码到text
                    if args.finetuned_model_name == 'gpner':
                        predict_datas[data['index']]['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,'object': text[text_offset_mapping[h][0]:text_offset_mapping[t][-1]], 'object_type': id2class[entity_type],'subject_h':ori_h,'subject_t':ori_t,'object_h':now_h,'object_t':now_t})
                    else:
                        predict_datas[data['index']]['spo_list'].append({'predicate': predicate, 'object': pre_entity, 'object_type': pre_entity_type,'subject': text[text_offset_mapping[h][0]:text_offset_mapping[t][-1]], 'subject_type': id2class[entity_type],'subject_h':now_h,'subject_t':now_t,'object_h':ori_h,'object_t':ori_t})
        # entity抽取完毕，输出
        logger.info("***** Running {} prediction *****".format(task))
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            for data in predict_datas:
                if 'entity_list' in data.keys():
                    del data['entity_list']
                f.write(data)
        end_time = time.time()
        print('All cost time:{} seconds'.format(end_time-start_time))
        self.get_predict_prf(output_dir)  

    def get_predict_prf(self,read_path):
        gold_path = os.path.join(self.args.data_dir,'test.json')
        # 黄金数据
        all_gold_jsons=[]
        with open(gold_path, 'r') as f_1:
            lines = f_1.readlines()
            for line in lines:
                all_gold_jsons.append(json.loads(line))
        gold_spos=[]
        for i in range(len(all_gold_jsons)):
            gold_json=all_gold_jsons[i]
            spo_list=gold_json['spo_list']
            for spo in spo_list:
                if self.args.with_type:
                    gold_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['subject_type'].strip(),spo['object'].strip(),spo['object_type'].strip()))
                else:
                    gold_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['object'].strip()))

        #获取预测数据
        all_predict_jsons=[]
        with open(read_path, 'r') as f_2:
            lines = f_2.readlines()
            for line in lines:
                all_predict_jsons.append(json.loads(line))
        predict_spos=[]
        for i in range(len(all_predict_jsons)):
            predict_json=all_predict_jsons[i]
            spo_list=predict_json['spo_list']
            for spo in spo_list:
                if self.args.with_type:
                    predict_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['subject_type'].strip(),spo['object'].strip(),spo['object_type'].strip()))
                else:
                    predict_spos.append((i,spo['predicate'],spo['subject'].strip(),spo['object'].strip()))

        # 计算pre,rec,f1

        P = len(set(predict_spos) & set(gold_spos)) / len(set(predict_spos)) if len(set(predict_spos)) != 0 else 0
        R = len(set(predict_spos) & set(gold_spos)) / len(set(gold_spos)) if len(set(gold_spos)) != 0 else 0
        F = (2 * P * R) / (P + R) if P+R != 0 else 0
        print('\nRESULT PRF：\n')
        print(str(round(P*100,2))+'|'+str(round(R*100,2))+'|'+str(round(F*100,2))+'|'+'\n')