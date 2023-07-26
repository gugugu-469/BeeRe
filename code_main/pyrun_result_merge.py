
import jsonlines
import os
from codes.utils import SPO, ACESPO
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_version_1', type=str, default='', required=True,
                        help="one model version")
    
    parser.add_argument('--model_version_2', type=str, default='', required=True,
                        help="another model version")

    parser.add_argument('--type', type=str, default='type1', choices=['type1','type2'],
                        help="type1:UNION; type2:INTERSECTION")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.type=='type1':
        # UNION
        read1 = os.path.join('./result_output', 'gpner9', args.model_version_2 ,'test.jsonl')
        read2 = os.path.join('./result_output', 'gpner', args.model_version_1 ,'test.jsonl')
        output_dir = os.path.join('./result_output', 'merge' ,'gpner-'+args.model_version_1+'__gpner9-'+args.model_version_2)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with jsonlines.open(f'{output_dir}/test.jsonl', mode='w') as w:
            with jsonlines.open(read1, mode='r') as r1, jsonlines.open(read2, mode='r') as r2:
                for data1, data2 in zip(r1, r2):
                    dic = {'text': data1['text'], 'spo_list': data1['spo_list'] + data2['spo_list']}
                    w.write(dic)
    else: 
        # INTERSECTION
        read1 = os.path.join('./result_output', 'gpner9', args.model_version_2 ,'test.jsonl')
        read2 = os.path.join('./result_output', 'gpner', args.model_version_1 ,'test.jsonl')
        output_dir = os.path.join('./result_output', 'merge' ,'gpner-'+args.model_version_1+'__gpner9-'+args.model_version_2)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with jsonlines.open(f'{output_dir}/test.jsonl', mode='w') as w:
            with jsonlines.open(read1, mode='r') as r1, jsonlines.open(read2, mode='r') as r2:
                for data1, data2 in zip(r1, r2):
                    dic = {'text': data1['text'], 'spo_list': []}
                    for spo in (set(ACESPO(spo) for spo in data1['spo_list']) & set(ACESPO(spo) for spo in data2['spo_list'])):
                        dic['spo_list'].append(spo.spo)
                    w.write(dic)


if __name__ == '__main__':
    main()
