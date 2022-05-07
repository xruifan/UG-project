"""
Author: Xuan-Rui Fan
Date: 30 Apr 2022
"""

import jsonlines
import json
import csv
from sklearn.metrics import f1_score

import os

if __name__ == '__main__':
    model_path = '/home/acb19lh/adapet/generic/bert-base-uncased/'
    eval_path = '/home/acb19lh/diss21/ADAPET-master/magpie-corpus-master/19280/'
    
    p0 = '[TEXT1], The sentence is [LBL].'
    p1 = '[TEXT1] : [LBL]'
    p2 = 'The following sentence is [LBL]. [TEXT1]'
    p3 = '([LBL]) [TEXT1]'
    
    for subdir in os.listdir(model_path):
        # read predicted labels
        pred_labels = []
        with jsonlines.open(os.path.join(model_path, subdir,'test.json'), mode='r') as reader:
            for r in reader:
                pred_labels.append(r['label'])
        
        # read true labels
        labels = []
        with jsonlines.open(os.path.join(eval_path,'test.jsonl'), mode='r') as reader:
            for r in reader:
                labels.append(r['LBL'])
        
        # calculate f1 score
        f1_macro = f1_score(labels, pred_labels, average='macro')
        assert len(labels) == len(pred_labels)
        
            
        # write f1 scores
        with open(os.path.join(model_path, subdir,'config.json'), mode='r') as reader:
                data = json.load(reader)
                pattern = data['pattern']
                data_num = data['data_dir'].split('/')[-2]
                
        if pattern == "[TEXT1], The sentence is [LBL].":
            pattern_num = 0
        elif pattern == "[TEXT1] : [LBL]":
            pattern_num = 1
        elif pattern == "The following sentence is [LBL]. [TEXT1]":
            pattern_num = 2
        elif pattern == "([LBL]) [TEXT1]":
            pattern_num = 3
        else:
            pattern_num = 999
        
        with open(os.path.join(model_path, subdir, 'f1_{}_p{}.txt'.format(data_num, pattern_num)), 'w') as f:
            f.write('final f1-macro score: {}'.format(f1_macro))
