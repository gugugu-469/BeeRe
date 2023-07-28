import os
import json
import jsonlines
from collections import defaultdict
from utils import random


class GPFilterDataProcessor(object):
    def __init__(self, args):
        self.args = args
        root = args.data_dir
        self.train_path = os.path.join(root, 'train.json')
        self.dev_path = os.path.join(root, 'dev.json')
        self.schema_path = os.path.join(root, 'schemas.json')
        if args.do_filter:
            self.test_path = os.path.join('./result_output', 'merge','gpner-'+args.model_version_1+'__gpner9-'+args.model_version_2, 'test.jsonl')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self):
        with jsonlines.open(self.test_path, 'r') as f:
            data_list = []
            for line in f:
                data_list.append(line)
        return data_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        schema = set()
        if self.args.with_type:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema = []
                for idx, item in enumerate(f):
                    item = json.loads(item.rstrip())
                    schema.append(item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"])
        else:
            if self.args.method_name == 'ace05':
                schema = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE']
            elif self.args.method_name == 'scierc':
                schema = ['Conjunction', 'Part-of', 'Hyponym-of', 'Evaluate-for', 'Used-for', 'Compare', 'Feature-of']
            else:
                for idx, item in enumerate(f):
                        item = json.loads(item.rstrip())
                        schema.append(item["predicate"])


        schema = list(schema)
        print('self.schema:{}'.format(schema))
        self.schema = schema
        self.num_predicates = len(schema)
        self.args.num_predicates = self.num_predicates
        self.predicate2id = {v: i for i, v in enumerate(schema)}
        self.id2predicate = {i: v for i, v in enumerate(schema)}
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                line = json.loads(line)
                for _ in range(iter_num):
                    new_data.append({
                        "text":line["text"],
                        "spo_list":[(spo["subject"], spo["predicate"], spo["object"], spo["subject_type"], spo["object_type"]) for spo in line["spo_list"]] if args.with_type\
                                    else [(spo["subject"], spo["predicate"], spo["object"], '', '') for spo in line["spo_list"]]
                    })
        return new_data
    
class GPNERDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, 'train.json')
        self.dev_path = os.path.join(root, 'dev.json')
        self.test_path = os.path.join(root, 'test.json')
        self.schema_path = os.path.join(root, 'schemas.json')
        self._load_schema()
        
    def get_train_sample(self):
        return self._pre_process(self.train_path)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path)

    def get_test_sample(self):
        with jsonlines.open(self.test_path, 'r') as f:
            data_list = []
            for line in f:
                line['entity_list'] = []
                line['spo_list'] = []
                data_list.append(line)
        return data_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        if self.args.method_name == 'scierc':
            labels = ['Method', 'Task', 'Material', 'OtherScientificTerm', 'Generic', 'Metric']
            predicates = ['Conjunction', 'Part-of', 'Hyponym-of', 'Evaluate-for', 'Used-for', 'Compare', 'Feature-of']
        elif self.args.method_name == 'ace05':
            labels = ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
            predicates = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE']
        else:
            labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
            predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        self.labels = labels
        self.predicates = predicates
        self.num_predicates = len(predicates)
        self.num_entities = len(labels)
        self.args.num_predicates = len(predicates)
        self.args.num_entities = len(labels)
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        self.id2predicate = {i: v for i, v in enumerate(predicates)}
        
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate):
        prefix = f"entity: {entity}, relation: {predicate}, "
        return prefix

    def build_data(self, text, spo_list, entity2predicate_dic, data_type=1, keep_rate=1):
        args = self.args
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                # SP2O
                temp = (spo['object'], spo['object_type']) if args.with_type else spo['object']
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append(temp)
                input_entity_types.append(spo["subject"])
            else:
                # OP2S
                temp = (spo['subject'], spo['subject_type']) if args.with_type else spo['subject']
                positive_dic[f"{spo['object']}{spo['predicate']}"].append(temp)
                input_entity_types.append(spo['object'])
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        
        for input_entity in input_entity_types:
            predicates = self.predicates if args.with_type else entity2predicate_dic[input_entity]
            for predicate in predicates:
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type,
                    "text": self.add_prefix(text, input_entity, predicate), 
                    "entity_list":[] 
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                new_data.append(data)
        return new_data
    
    def _pre_process(self, path):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                new_data.append({
                    "type": 0,
                    "text": text,
                    "entity_list":[(spo["subject"], spo['subject_type']) if args.with_type else spo["subject"] for spo in spo_list] if self.args.finetuned_model_name == 'gpner' \
                                  else [(spo["object"], spo["object_type"]) if args.with_type else spo["object"] for spo in spo_list]
                })
                if self.args.finetuned_model_name == 'gpner':
                    new_data.extend(self.build_data(text, spo_list, self.subject_predicate_dic, 1, self.args.negative_samples_rate))
                else:
                    new_data.extend(self.build_data(text, spo_list, self.object_predicate_dic, 2, self.args.negative_samples_rate))
        return new_data
