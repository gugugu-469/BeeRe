import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

class GPFilterDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train'
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.predicate2id = data_processor.predicate2id #spo
        self.schema = data_processor.schema #spo
        self.args = args
        if mode=='test' or mode == 'dev':
            self.get_item_func = self.get_item_test
        else:
            self.get_item_func = self.get_item_train
        
    def __len__(self):
        return len(self.samples)

    def encoder(self, item):
        args = self.args
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            sub_tokens = self.tokenizer.encode(s, add_special_tokens=False)
            key = s_t + "_" + p + "_" +o_t if args.with_type else p
            p = self.predicate2id[key]
            obj_tokens = self.tokenizer.encode(o, add_special_tokens=False)
            sh = self.data_processor.search(sub_tokens, input_ids)
            oh = self.data_processor.search(obj_tokens, input_ids)
            
            if sh == -1:
                sub_tokens = self.tokenizer.encode(' '+s, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)
            if oh == -1:
                obj_tokens = self.tokenizer.encode(' '+o, add_special_tokens=False)
                oh = self.data_processor.search(obj_tokens, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(sub_tokens)-1, p, oh, oh+len(obj_tokens)-1))

        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return head_labels, tail_labels, input_ids, attention_mask


    def get_item_test(self, idx):
            return self.samples[idx]

    def get_item_train(self, idx):
        item = self.samples[idx]
        return self.encoder(item)


    def __getitem__(self, idx):
        return self.get_item_func(idx)


    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids = [], []
        batch_head_labels, batch_tail_labels = [], []
        for item in examples:
            head_labels, tail_labels, input_ids, attention_mask = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()

        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels

    @staticmethod
    def collate_fn_test(examples):
        batch_texts = []
        batch_spo_lists = []
        batch_out = []
        for item in examples:
            batch_texts.append(item['text'])
            batch_spo_lists.append(item['spo_list'])
            batch_out.append({'text':item['text'],'spo_list':[]})


        return batch_texts,batch_spo_lists,batch_out

class GPNERDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train'
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
        if mode=='test' or mode == 'dev':
            self.get_item_func = self.get_item_test
        else:
            self.get_item_func = self.get_item_train
        
    def __len__(self):
        return len(self.samples)
    
    def get_item_test(self, idx):
        return self.samples[idx]

    def get_item_train(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    def encoder(self, item):
        args = self.args
        num_labels = args.num_entities if args.with_type else 1
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        if args.with_type:
            for sub, sub_type in item["entity_list"]:
                sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)

                if sh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
                else:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    sh = self.data_processor.search(sub_tokens, input_ids)
                    if sh != -1:
                        spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
            head_labels = [set() for i in range(num_labels)]
            for sh, st, sub_type in spoes:
                head_labels[sub_type].add((sh, st)) 
        else:
            for sub in item["entity_list"]:
                sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)

                if sh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1))
                else:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    sh = self.data_processor.search(sub_tokens, input_ids)
                    if sh != -1:
                        spoes.add((sh, sh+len(sub_tokens)-1))
            head_labels = [set() for i in range(num_labels)]
            for sh, st in spoes:
                head_labels[0].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return head_labels, input_ids, attention_mask

    def __getitem__(self, idx):
        return self.get_item_func(idx)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids = [], []
        batch_head_labels = []
        for item in examples:
            head_labels, input_ids, attention_mask = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_head_labels

    @staticmethod
    def collate_fn_test(examples):
        text_list = []
        for item in examples:
            text = item
            text_list.append(text)

        return text_list
