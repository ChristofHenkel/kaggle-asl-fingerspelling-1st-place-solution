import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

def post_process_pipeline(cfg, val_data, val_df):
    
    generated_ids = val_data['generated_ids'].cpu()
    seq_len = val_data['seq_len'].cpu()
    phrase_preds = ["".join([cfg.rev_character_map.get(s, "") for s in generated_id]) for generated_id in generated_ids.numpy()]
    
    #repplace short sequence predictions with dummy phrase
    mask = seq_len < cfg.max_len_for_dummy
    
    dummy_seq = torch.tensor(cfg.dummy_phrase_ids)
    dummy_seq = dummy_seq[None,].repeat(mask.sum(),1)
    generated_ids_pp = generated_ids.contiguous()
    generated_ids_pp[mask] = 59
    generated_ids_pp[mask,:dummy_seq.shape[1]] = dummy_seq
    generated_ids_pp = generated_ids_pp.numpy()
    
    #ids to tokens
    phrase_preds_pp = ["".join([cfg.rev_character_map.get(s, "") for s in generated_id]) for generated_id in generated_ids_pp]
    
    
    return {'phrase_preds':phrase_preds,'phrase_preds_pp':phrase_preds_pp}
