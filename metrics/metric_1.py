import numpy as np
from sklearn.metrics import f1_score
import torch
import scipy as sp
from rapidfuzz.distance.DamerauLevenshtein_py import distance


def get_score(phrase_gt, phrase_preds):
    N = np.array([len(p) for p in phrase_gt])
    D = np.array([distance(p1,p2) for p1,p2 in zip(phrase_gt,phrase_preds)])
    score = (N.sum() - D.sum()) / N.sum()    
    
    return score

def calc_metric(cfg, pp_out, val_df, pre="val"):
    
    
    phrase_gt = val_df['phrase'].values
    phrase_preds = pp_out['phrase_preds']
    phrase_preds_pp = pp_out['phrase_preds_pp']
    
    score = get_score(phrase_gt, phrase_preds)
    score_pp = get_score(phrase_gt, phrase_preds_pp)
    
    
    return {'score':score,
           'score_pp':score_pp,
           }