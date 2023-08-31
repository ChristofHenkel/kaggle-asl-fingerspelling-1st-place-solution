#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import glob
from rapidfuzz.distance.DamerauLevenshtein_py import distance
import json


#TODO add argsparse

train = pd.read_csv('datamount/train_folded.csv')
train


# In[11]:


val_data_fns = [glob.glob(f'datamount/weights/cfg_1/fold{i}/val_data_seed*.pth') for i in [0,1,2,3]]
print('getting oofs from')
print(val_data_fns)




def get_score(phrase_gt, phrase_preds):
    N = np.array([len(p) for p in phrase_gt])
    D = np.array([distance(p1,p2) for p1,p2 in zip(phrase_gt,phrase_preds)])
    score = (N - D) / N   
    
    return score

with open('datamount/character_to_prediction_index.json', "r") as f:
    char_to_num = json.load(f)

rev_character_map = {j:i for i,j in char_to_num.items()}

def decode(generated_ids):
    return ''.join([rev_character_map.get(id_,'') for id_ in generated_ids])



dfs = []
for fold,val_fns in enumerate(val_data_fns):
    val_df = train[train['fold']==fold].copy()
    val_scores = []
    for val_fn in val_fns:
        val_data = torch.load(val_fn)['generated_ids']

        val_preds = np.array([decode(generated_ids) for generated_ids in tqdm(val_data.cpu().numpy())])
        val_scores += [get_score(val_df['phrase'].values, val_preds)]
    val_scores = np.stack(val_scores).mean(0)
    val_df['score'] = val_scores
    dfs += [val_df]


df = pd.concat(dfs)
df = df.loc[train.index]


df['phrase_len'] = df['phrase'].str.len()


#add supplemental data

train_supp = pd.read_csv('datamount/supplemental_metadata_folded.csv')


train_supp = train_supp[train_supp['phrase_len'] < 33].copy()
train_supp['score'] = 0.5
train_supp['is_sup'] = 1
df['is_sup'] = 0

df = pd.concat([df,train_supp])

df.to_csv('datamount/train_folded_oof_supp.csv', index=False)


