import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import torch
import pandas as pd
from transformers.models.speech_to_text import Speech2TextConfig
import augmentations as A
from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg.debug = True

#paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"datamount/weights/{os.path.basename(__file__).split('.')[0]}"
cfg.data_folder = f"datamount/train_landmarks_npy/"
cfg.train_df = f'datamount/train_folded.csv'
cfg.symmetry_fp = 'datamount/symmetry.csv'

# stages
cfg.test = False
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1
cfg.seed = -1

#logging
cfg.neptune_project = "common/quickstarts"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

# DATASET
cfg.dataset = "ds_1"
cfg.min_seq_len = 15

cfg.max_len = 384
cfg.max_phrase = 31 + 2 #max of train data + SOS + EOS
with open('datamount/character_to_prediction_index.json', "r") as f:
    char_to_num = json.load(f)

cfg.rev_character_map = {j:i for i,j in char_to_num.items()}
n= len(char_to_num)
cfg.pad_token = 'P'
cfg.start_token = 'S'
cfg.end_token = 'E'
char_to_num[cfg.pad_token] = n
char_to_num[cfg.start_token] = n+1
char_to_num[cfg.end_token] = n+2
num_to_char = {j:i for i,j in char_to_num.items()}
chars = np.array([num_to_char[i] for i in range(len(num_to_char))])

cfg.tokenizer = [char_to_num,num_to_char,chars]

#model

cfg.model = "mdl_1_pt"
cfg.ce_ignore_index = -100
cfg.label_smoothing = 0.
cfg.n_landmarks = 130
cfg.return_logits = False
cfg.pretrained = True
cfg.val_mode = 'padded'

config = Speech2TextConfig.from_pretrained("facebook/s2t-small-librispeech-asr")
config.encoder_layers = 0
config.decoder_layers = 2
config.d_model = 144
config.max_target_positions = 1024 #?
config.num_hidden_layers = 1
config.vocab_size = 63
config.bos_token_id = char_to_num[cfg.start_token]
config.eos_token_id = char_to_num[cfg.end_token]
config.decoder_start_token_id = char_to_num[cfg.start_token]
config.pad_token_id = char_to_num[cfg.pad_token]
config.num_conv_layers = 0
config.conv_kernel_sizes = []
config.max_length = 144
config.input_feat_per_channel = 144
config.num_beams = 1
config.attention_dropout = 0.2
# config.dropout = 0.2
config.decoder_ffn_dim = 512
config.init_std = 0.02
cfg.transformer_config = config


encoder_config = SimpleNamespace(**{})
encoder_config.input_dim=144
encoder_config.encoder_dim=144
encoder_config.num_layers=6
encoder_config.num_attention_heads= 4
encoder_config.feed_forward_expansion_factor=1
encoder_config.conv_expansion_factor= 2
encoder_config.input_dropout_p= 0.1
encoder_config.feed_forward_dropout_p= 0.1
encoder_config.attention_dropout_p= 0.1
encoder_config.conv_dropout_p= 0.1
encoder_config.conv_kernel_size= 51

cfg.encoder_config = encoder_config


# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 300
cfg.lr = 5e-4 * 9
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.05
cfg.clip_grad = 4.
cfg.warmup = 10
cfg.batch_size = 64
cfg.batch_size_val = 128
cfg.mixed_precision = False # True
cfg.pin_memory = False
cfg.grad_accumulation = 8.
cfg.num_workers = 8


#EVAL
cfg.calc_metric = True
cfg.eval_epochs = 10
cfg.save_val_data = True
# augs & tta

# Postprocess
cfg.post_process_pipeline =  "pp_1"
cfg.dummy_phrase_ids = [char_to_num[c] for c in '2 a-e -aroe']
cfg.max_len_for_dummy = 15
cfg.metric = "metric_1"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

cfg.decoder_mask_aug = 0.2
cfg.flip_aug = 0.5
cfg.outer_cutmix_aug = 0.5
cfg.train_aug = A.Compose([A.Resample(sample_rate=(0.5,1.5), p=0.8),
                           A.SpatialAffine(scale=(0.8,1.2),shear=(-0.15,0.15),shift=(-0.1,0.1),degree=(-30,30),p=0.75),  
                           A.TemporalMask(size=(0.2,0.4),mask_value=0.,p=0.5), #mask with 0 as it is post-normalization
                           A.SpatialMask(size=(0.05,0.1),mask_value=0.,mode='relative',p=0.5), #mask with 0 as it is post-normalization
                          ])
cfg.train_aug._disable_check_args() #disable otherwise input must be numpy/ int8

cfg.val_aug = None

