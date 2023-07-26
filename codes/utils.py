import torch
import json
from datetime import datetime
from torch import nn
import logging
import random
import os
import time
import numpy as np
import unicodedata
import pynvml

def get_time(fmt='%Y-%m-%d %H:%M:%S'):
    """
    获取当前时间
    """
    ts = time.time()
    ta = time.localtime(ts)
    t = time.strftime(fmt, ta)
    return t

def save_args(args, path):
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        f.writelines('------------------- start -------------------\n')
        for arg, value in args.__dict__.items():
            f.writelines(f'{arg}: {str(value)}\n')
        f.writelines('------------------- end -------------------')
        
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger


def get_devices(devices_id):
    devices = []
    for i in devices_id:
        if i == '-1':
            devices.append(torch.device('cpu'))
        else:
            devices.append(torch.device(f'cuda:{i}'))
#     return [torch.device(f'cuda:{i}') for i in devices_id]
    return devices

class ProgressBar(object):
    """
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step)
    """
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
    
class TokenRematch:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                # offset的作用是避免文本中有重复字符
                offset = end

        return token_mapping
    
class SPO():
    def __init__(self, spo):
        self.spo = spo
        
    def __str__(self):
        return self.spo.__str__()
        
    def __eq__(self, other):
        return self.spo['predicate'] == other.spo['predicate'] and \
               self.spo['subject'] == other.spo['subject'] and self.spo['subject_type'] == other.spo['subject_type'] and \
               self.spo['object']["@value"] == other.spo['object']["@value"] and self.spo['object_type']["@value"] == other.spo['object_type']["@value"]
    
    def __hash__(self):
        return hash(self.spo['predicate'] + self.spo['subject'] + self.spo['subject_type'] + self.spo['object']["@value"] + self.spo['object_type']["@value"])
    
class ACESPO():
    def __init__(self, spo, with_type=False):
        self.spo = spo
        self.with_type = with_type
        
    def __str__(self):
        return self.spo.__str__()
        
    def __eq__(self, other):
        if self.with_type:
            return self.spo['predicate'] == other.spo['predicate'] and \
               self.spo['subject'] == other.spo['subject'] and self.spo['subject_type'] == other.spo['subject_type'] and\
               self.spo['object'] == other.spo['object'] and self.spo['object_type'] == other.spo['object_type']
        else:
            return self.spo['predicate'] == other.spo['predicate'] and \
               self.spo['subject'] == other.spo['subject'] and \
               self.spo['object'] == other.spo['object']
    
    def __hash__(self):
        if self.with_type:
            return hash(self.spo['predicate'] + self.spo['subject'] + self.spo['subject_type'] + self.spo['object'] + self.spo['object_type'])
        else:
            return hash(self.spo['predicate'] + self.spo['subject'] + self.spo['object'])
        