#!/usr/bin/env python
# coding: utf-8
# %%

import importlib, os, glob, copy, sys
import numpy as np
import pandas as pd
import copy
import json
import platform
import torch
from torch import nn
import tempfile
from torch.utils.data import Dataset, DataLoader
from transformers import Speech2TextForConditionalGeneration
from transformers import TFSpeech2TextForConditionalGeneration, TFAutoModel
from pathlib import Path
import shutil
from transformers import LogitsProcessorList

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Input
from transformers import TFLogitsProcessorList
from collections import OrderedDict



sys.path.append("./configs")
sys.path.append("./data")
sys.path.append("./models")
sys.path.append("./")

from rapidfuzz.distance.DamerauLevenshtein_py import distance


def get_score(phrase_gt, phrase_preds):
    N = np.array([len(p) for p in phrase_gt])
    D = np.array([distance(p1,p2) for p1,p2 in zip(phrase_gt,phrase_preds)])
    score = (N.sum() - D.sum()) / N.sum()    
    
    return score


def port_weights_to_tf_model(pt_model, tf_model):
    pt_model.eval() 
    pt_weights = pt_model.state_dict()
    lyrs_tf = [(i.name, i.shape.as_list()) for i in tf_model.weights]
    lyrs_pt = [(k,v.shape) for k,v in pt_weights.items() if len(v.shape)>0]

    assert len(lyrs_tf) == len(lyrs_pt)
    
    layer_weights = []
    for t, ((k_tf, sh_tf), (k_pt, sh_pt)) in enumerate(zip(lyrs_tf, lyrs_pt)):
        wts = pt_weights[k_pt].clone()
        if '/kernel' in k_tf:
            if len(wts.shape)==2:
                wts = wts.permute(1,0)
            elif len(wts.shape)==3:
                wts = wts.permute(2, 1,0)
        if '_dwconv' in k_tf:
            wts = wts.permute(2,0,1)
        layer_weights .append(wts.numpy())
    tf_model.set_weights(layer_weights)
    
    return tf_model


CACHE_PATH = 'cache'
config_name = 'cfg_2'
FOLD = -1
cfg = copy.copy(importlib.import_module(config_name).cfg)
df = pd.read_csv(cfg.train_df)

CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
val_df = df[df['fold']==0].copy() 

val_ds = CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")

model = importlib.import_module(cfg.model)
importlib.reload(model)
model_tf = importlib.import_module(cfg.model_tf)
importlib.reload(model_tf)
cfg.device = 'cpu'

net = model.Net(cfg).eval().to(cfg.device)
net.to(cfg.device)


# Load the weights. 
sd_fps = sorted(glob.glob(f'datamount/weights/{config_name}/fold{FOLD}/checkpoint_*'))
vals_fps = sorted(glob.glob(f'datamount/weights/{config_name}/fold{FOLD}/val*'))

sdnm = sd_fps[0]
valdnm = vals_fps[0]
print('SEED0:', sdnm)

sd = torch.load(sdnm, map_location=torch.device('cpu'))
val_data = torch.load(valdnm, map_location=torch.device('cpu'))
sd['model'] = {k.replace('module.', ''):v for k,v in sd['model'].items()} #for models trained with DDP
net.load_state_dict(sd['model'], strict=True)

net2 = model.Net(cfg).eval().to(cfg.device)
net2.to(cfg.device)
print()

sdnm2 = sd_fps[1]
valdnm2 = vals_fps[1]
print('SEED1:', sdnm2)

sd2 = torch.load(sdnm2, map_location=torch.device('cpu'))
val_data2 = torch.load(valdnm2, map_location=torch.device('cpu'))
sd2['model'] = {k.replace('module.', ''):v for k,v in sd2['model'].items()} #for models trained with DDP
net2.load_state_dict(sd2['model'], strict=True)


pad_token_id, bos_token_id, eos_token_id = [cfg.tokenizer[0][i] for i in 'P S E'.split()]
def decode_pred(pred_ids, eos_token = 'E'):
    pred_str = ''.join([cfg.tokenizer[1][i.item()] for i in pred_ids if i not in [pad_token_id, bos_token_id]])
    pred_str = pred_str.split(eos_token)[0]
    return pred_str




# Definde the preprocessor



class PreprocessingTF(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessingTF, self).__init__()
                
    def normalize(self,x):
        
        x2 = tf.reshape(x, [-1, x.shape[-1]])
        idx = tf.where(~tf.math.is_nan(x2[:,0]))[:,0]
        nonan = tf.gather(x2, idx)
        
        x = x - tf.reduce_mean(nonan, 0)[None, None, :]
        x = x / tf.math.reduce_std(nonan, axis = 0)[None, None, :]
        
        return x
    
    def fill_nans(self,x):
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        return x
        
    def call(self, x):
        
        # x = x[None]
        x = tf.transpose(tf.reshape(x, [len(x), 3,-1]), [0,2,1])   
        
        # Normalize & fill nans
        x = self.normalize(x)
        x = self.fill_nans(x)
        
        return x

tf_processor = PreprocessingTF()



class tf_FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, out_dim, n_landmarks):
        super().__init__()
        
        self.conv_out_dim = 32 * math.ceil(n_landmarks / 2)
        self.stem_linear = tf.keras.layers.Dense(out_dim, use_bias=False,name='stem_linear')
        self.stem_bn = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn', epsilon=1e-5)

        self.pad_dim = tf.keras.layers.ZeroPadding2D((1, 1))
        self.conv_stem = tf.keras.layers.Conv2D(32, (3, 3), padding = 'valid', strides = (1, 2), use_bias = False, name='conv_stem')
        self.bn_conv  = tf.keras.layers.BatchNormalization(momentum=0.1 ,name='bn_conv', epsilon=1e-05 , center=True, scale=True)
        self.act = tf.nn.silu

    def call(self, data):
        
        '''
        data = tf.convert_to_tensor(batch['input'][0])
        self.orig_input_dim = 2 * cfg.n_landmarks
        conv_stem = tf.keras.layers.Conv2D(32, (3, 3), padding = "same", strides = (1, 2), use_bias = False, name='conv_stem')
        
        '''
        
   
        xc = tf.expand_dims(data, axis = 0)

        
        xc = self.pad_dim(xc)
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = self.act(xc)
        xc = tf.reshape(xc, [1, -1, self.conv_out_dim])        
    
        x = self.stem_linear(xc)
        x = self.stem_bn(x)
        return x

fe_mapper = {"stem_linear/kernel:0": "stem_linear.weight",
            "stem_bn/gamma:0": "stem_bn.weight",
            "stem_bn/beta:0": "stem_bn.bias",
            "conv_stem/kernel:0": "conv_stem.weight",
            "bn_conv/gamma:0": "bn_conv.weight",
            "bn_conv/beta:0": "bn_conv.bias",
            "stem_bn/moving_mean:0": "stem_bn.running_mean", 
            "stem_bn/moving_variance:0": "stem_bn.running_var", 
            "bn_conv/moving_mean:0": "bn_conv.running_mean",
            "bn_conv/moving_variance:0": "bn_conv.running_var",}




# Convert the feature extractor



import math

def port_weights_to_tf_model2(pt_model, tf_model, mapper):
    lyrs_tf = [(i.name, i.shape.as_list()) for i in tf_model.weights]
    tf_keys = [w[0].split('/', maxsplit=1)[-1] for w in  lyrs_tf]
    pt_weights = pt_model.state_dict()
    lyrs_pt = [(mapper[k], pt_weights[mapper[k]].shape) for k in tf_keys]
    
    layer_weights = []
    for t, ((k_tf, sh_tf), (k_pt, sh_pt)) in enumerate(zip(lyrs_tf, lyrs_pt)):
        wts = pt_weights[k_pt].clone()
        if '/kernel' in k_tf:
            if len(wts.shape)==2:
                wts = wts.permute(1,0)
            elif len(wts.shape)==3:
                wts = wts.permute(2, 1,0)
            elif len(wts.shape)==4:
                wts = wts.permute(2, 3, 1,0)
        if '_dwconv' in k_tf:
            wts = wts.permute(2,0,1)
        layer_weights .append(wts.numpy())
    tf_model.set_weights(layer_weights)
    return tf_model


tf_indexes = [list(range(130))]+\
              [np.where(net.landmark_types == i)[0].tolist() for i in range(1,5)]
fe_names = ['feature_extractor',
            'feature_extractor_lhand',
            'feature_extractor_rhand',
             'feature_extractor_face',
             'feature_extractor_pose']


fetf_out_dict = {}
for tf_indx, fe_name in zip(tf_indexes, fe_names):
#     print(f'Map {fe_name}')
    pt_fe = getattr(net, fe_name)
    n_landmarks_fe = pt_fe.stem_linear.in_features // 16
    fe_dim = pt_fe.stem_linear.out_features
    fetf_out_dict[fe_name] = tf_FeatureExtractor(fe_dim, n_landmarks_fe)
    inp = tf.keras.Input((len(tf_indx), 3))
    x = fetf_out_dict[fe_name](inp)
    fetf_out_dict[fe_name] = tf.keras.Model(inp, x)
    fetf_out_dict[fe_name] = port_weights_to_tf_model2(pt_fe, fetf_out_dict[fe_name], model_tf.fe_mapper)
    

fetf_out_dict2 = {}
for tf_indx, fe_name in zip(tf_indexes, fe_names):
    pt_fe = getattr(net2, fe_name)
    n_landmarks_fe = pt_fe.stem_linear.in_features // 16
    fe_dim = pt_fe.stem_linear.out_features
    fetf_out_dict2[fe_name] = tf_FeatureExtractor(fe_dim, n_landmarks_fe)
    inp = tf.keras.Input((len(tf_indx), 3))
    x = fetf_out_dict2[fe_name](inp)
    fetf_out_dict2[fe_name] = tf.keras.Model(inp, x)
    fetf_out_dict2[fe_name] = port_weights_to_tf_model2(pt_fe, fetf_out_dict2[fe_name], model_tf.fe_mapper)


print('transfering feature extractor done')

'''
Convert the encoder
'''
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding # LlamaAttention, 
rotary_emb = LlamaRotaryEmbedding(cfg.encoder_config.encoder_dim//cfg.encoder_config.num_attention_heads, max_position_embeddings=cfg.max_len)

# Feed these into each layer, so they are not kept as extra paramters multiple times within each layer
cos = rotary_emb.cos_cached
sin = rotary_emb.sin_cached

encoder_pt = model.SqueezeformerEncoder(
                      input_dim=cfg.encoder_config.input_dim,
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      input_dropout_p= cfg.encoder_config.input_dropout_p,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
encoder_pt.eval()
encoder_pt.load_state_dict( net.encoder.state_dict() )

x = model_tf.SqueezeformerEncoderTF(
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
inp = tf.keras.Input((None,cfg.encoder_config.encoder_dim))
cos_inp = tf.keras.Input(tuple(cos.shape[1:]))
sin_inp = tf.keras.Input(tuple(sin.shape[1:]))
out = x(inp)#, cos_inp, sin_inp)
encoder_tf =  tf.keras.Model(inp, out)



encoder_pt2 = model.SqueezeformerEncoder(
                      input_dim=cfg.encoder_config.input_dim,
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      input_dropout_p= cfg.encoder_config.input_dropout_p,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
encoder_pt2.eval()
encoder_pt2.load_state_dict( net2.encoder.state_dict() )

x2 = model_tf.SqueezeformerEncoderTF(
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
inp = tf.keras.Input((None,cfg.encoder_config.encoder_dim))
cos_inp = tf.keras.Input(tuple(cos.shape[1:]))
sin_inp = tf.keras.Input(tuple(sin.shape[1:]))
out = x2(inp)
encoder_tf2 =  tf.keras.Model(inp, out)


## port weights
from collections import OrderedDict


pt_weights = encoder_pt.state_dict()
lyrs_tf = [(i.name, i.shape.as_list()) for i in encoder_tf.weights]
lyrs_pt = [(k,v.shape) for k,v in pt_weights.items() if len(v.shape)>0]

lyrs_tf2 = OrderedDict(lyrs_tf)
lyrs_pt2 = OrderedDict(lyrs_pt)

weight_map_rev0 = {'sequeeze_former_encoder/squeezeformer_block_0/mhsa/attention/pos_proj:0':'blocks.0.mhsa.attention.pos_proj.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/tf_llama_attention/q_proj/kernel:0':'blocks.0.mhsa_llama.q_proj.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/tf_llama_attention/k_proj/kernel:0':'blocks.0.mhsa_llama.k_proj.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/tf_llama_attention/v_proj/kernel:0':'blocks.0.mhsa_llama.v_proj.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/tf_llama_attention/o_proj/kernel:0':'blocks.0.mhsa_llama.o_proj.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_mhsa/gamma:0':'blocks.0.ln_mhsa.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_mhsa/beta:0':'blocks.0.ln_mhsa.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_mhsa/ffn1/kernel:0':'blocks.0.ff_mhsa.ffn1.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_mhsa/ffn1/bias:0':'blocks.0.ff_mhsa.ffn1.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_mhsa/ffn2/kernel:0':'blocks.0.ff_mhsa.ffn2.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_mhsa/ffn2/bias:0':'blocks.0.ff_mhsa.ffn2.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_ff_mhsa/gamma:0':'blocks.0.ln_ff_mhsa.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_ff_mhsa/beta:0':'blocks.0.ln_ff_mhsa.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_pw_conv_1/kernel:0':'blocks.0.conv.pw_conv_1.conv.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_pw_conv_1/bias:0':'blocks.0.conv.pw_conv_1.conv.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_dw_conv/depthwise_kernel:0':'blocks.0.conv.dw_conv.conv.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_dw_conv/bias:0':'none',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_bn/gamma:0':'blocks.0.conv.bn.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_bn/beta:0':'blocks.0.conv.bn.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_pw_conv_2/kernel:0':'blocks.0.conv.pw_conv_2.conv.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_pw_conv_2/bias:0':'blocks.0.conv.pw_conv_2.conv.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_conv/gamma:0':'blocks.0.ln_conv.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_conv/beta:0':'blocks.0.ln_conv.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_conv/ffn1/kernel:0':'blocks.0.ff_conv.ffn1.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_conv/ffn1/bias:0':'blocks.0.ff_conv.ffn1.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_conv/ffn2/kernel:0':'blocks.0.ff_conv.ffn2.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ff_conv/ffn2/bias:0':'blocks.0.ff_conv.ffn2.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_ff_conv/gamma:0':'blocks.0.ln_ff_conv.weight',
 'sequeeze_former_encoder/squeezeformer_block_0/ln_ff_conv/beta:0':'blocks.0.ln_ff_conv.bias',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_bn/moving_mean:0':'blocks.0.conv.bn.running_mean',
 'sequeeze_former_encoder/squeezeformer_block_0/conv/conv_bn/moving_variance:0':'blocks.0.conv.bn.running_var',
 'squeezeformer_block_0/scale_mhsa:0': 'blocks.0.scale_mhsa',
 'squeezeformer_block_0/bias_mhsa:0': 'blocks.0.bias_mhsa',
 'squeezeformer_block_0/scale_ff_mhsa:0': 'blocks.0.scale_ff_mhsa',
 'squeezeformer_block_0/bias_ff_mhsa:0': 'blocks.0.bias_ff_mhsa',
 'squeezeformer_block_0/scale_conv:0': 'blocks.0.scale_conv',
 'squeezeformer_block_0/bias_conv:0':  'blocks.0.bias_conv',
 'squeezeformer_block_0/scale_ff_conv:0': 'blocks.0.scale_ff_conv',
 'squeezeformer_block_0/bias_ff_conv:0': 'blocks.0.bias_ff_conv',}



weight_map_rev = weight_map_rev0
for block in range(1, cfg.encoder_config.num_layers):
    key_replace1 = ['squeezeformer_block_0',f'squeezeformer_block_{block}']
    key_replace2 = ['/tf_llama_attention/',f'/tf_llama_attention_{block}/']
    weight_map_rev.update({key.replace(*key_replace1).replace(*key_replace2):val.replace('blocks.0',f'blocks.{block}') for key,val in weight_map_rev0.items()})
    
    
lyrs_pt3 = []
for k, _ in lyrs_tf:
    pt_k = weight_map_rev[k]
    if pt_k == 'none':
        lyrs_pt3 += [(None,0)]
    else:
        lyrs_pt3 += [(pt_k,lyrs_pt2[pt_k])]
lyrs_pt3

encoder_dim = cfg.encoder_config.encoder_dim
layer_weights = []
for t, ((k_tf, sh_tf), (k_pt, sh_pt)) in enumerate(zip(lyrs_tf, lyrs_pt3)):
    
    if k_pt == None:
        wts = torch.zeros(sh_tf)
    else:
        wts = pt_weights[k_pt].clone() 
        
    if 'pos_proj:0' in k_tf:

        wts = wts.reshape(4,encoder_dim//4,encoder_dim).permute(0,2,1)
    if 'out_proj_kernel:0' in k_tf:
        wts = wts.permute(1,0)
        wts = wts.reshape(4,encoder_dim//4,encoder_dim)   
    
    if ('/kernel' in k_tf) or ('/depthwise_kernel' in k_tf):
        if len(wts.shape)==2:
            wts = wts.permute(1,0)
        elif len(wts.shape)==3:
            wts = wts.permute(2, 1,0)
        if 'pw_conv' in k_tf:
            wts = wts[None]
        if 'dw_conv' in k_tf:
            wts = wts[...,None]  
        
    layer_weights.append(wts.cpu().numpy())
    
encoder_tf.set_weights(layer_weights)



pt_weights2 = encoder_pt2.state_dict()

layer_weights2 = []
for t, ((k_tf, sh_tf), (k_pt, sh_pt)) in enumerate(zip(lyrs_tf, lyrs_pt3)):
    if k_pt == None:
        wts = torch.zeros(sh_tf)
    else:
        wts = pt_weights2[k_pt].clone() 
        
    if 'pos_proj:0' in k_tf:

        wts = wts.reshape(4,encoder_dim//4,encoder_dim).permute(0,2,1)
    if 'out_proj_kernel:0' in k_tf:
        wts = wts.permute(1,0)
        wts = wts.reshape(4,encoder_dim//4,encoder_dim)   
    
    if ('/kernel' in k_tf) or ('/depthwise_kernel' in k_tf):
        if len(wts.shape)==2:
            wts = wts.permute(1,0)
        elif len(wts.shape)==3:
            wts = wts.permute(2, 1,0)
        if 'pw_conv' in k_tf:
            wts = wts[None]
        if 'dw_conv' in k_tf:
            wts = wts[...,None]  
        
    layer_weights2.append(wts.cpu().numpy())



encoder_tf2.set_weights(layer_weights2)

print('transfering encoder done')

# '''
# Convert the decoder
# '''
sys.path.append('scripts')
from modeling_tf_speech_to_text2_cache import TFSpeech2TextDecoder2 as TFSpeech2TextDecoder
from transformers.modeling_tf_pytorch_utils import load_pytorch_state_dict_in_tf2_model


tf_s2t_decoder = TFSpeech2TextDecoder(cfg.transformer_config)
tf_s2t_decoder.base_model_prefix='tf_speech2_text_decoder'
tf_s2t_decoder._keys_to_ignore_on_load_missing = None
tf_s2t_decoder._keys_to_ignore_on_load_unexpected = None


input_ids = 60 +tf.zeros(2 + 33, dtype = tf.int64)#[None,:]
pos = tf.constant([[2]], dtype=tf.int32)
tf_input_ids = input_ids[None,:pos[0,0]]
encoder_hidden_states = tf.keras.Input((None,cfg.encoder_config.encoder_dim))[0][None]
tf_inputs = {'input_ids':tf_input_ids,'encoder_hidden_states':encoder_hidden_states}
_ = tf_s2t_decoder(tf_inputs)


## manually port weights

pt_weights = net.decoder.decoder.state_dict()
lyrs_tf = [(i.name, i.shape.as_list()) for i in tf_s2t_decoder.weights]
lyrs_pt = [(k,v.shape) for k,v in pt_weights.items() if len(v.shape)>0]

lyrs_tf2 = OrderedDict(lyrs_tf)
lyrs_pt2 = OrderedDict(lyrs_pt)

weight_map_rev = {'tf_speech2_text_decoder2/embed_tokens/weight:0': 'embed_tokens.weight',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/k_proj/kernel:0': 'layers.0.self_attn.k_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/k_proj/bias:0': 'layers.0.self_attn.k_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/q_proj/kernel:0': 'layers.0.self_attn.q_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/q_proj/bias:0': 'layers.0.self_attn.q_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/v_proj/kernel:0': 'layers.0.self_attn.v_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/v_proj/bias:0': 'layers.0.self_attn.v_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/out_proj/kernel:0': 'layers.0.self_attn.out_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn/out_proj/bias:0': 'layers.0.self_attn.out_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn_layer_norm/gamma:0': 'layers.0.self_attn_layer_norm.weight',
 'tf_speech2_text_decoder2/cond/layers.0/self_attn_layer_norm/beta:0': 'layers.0.self_attn_layer_norm.bias',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/k_proj/kernel:0': 'layers.0.encoder_attn.k_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/k_proj/bias:0': 'layers.0.encoder_attn.k_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/q_proj/kernel:0': 'layers.0.encoder_attn.q_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/q_proj/bias:0': 'layers.0.encoder_attn.q_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/v_proj/kernel:0': 'layers.0.encoder_attn.v_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/v_proj/bias:0': 'layers.0.encoder_attn.v_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/out_proj/kernel:0': 'layers.0.encoder_attn.out_proj.weight',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn/out_proj/bias:0': 'layers.0.encoder_attn.out_proj.bias',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn_layer_norm/gamma:0': 'layers.0.encoder_attn_layer_norm.weight',
 'tf_speech2_text_decoder2/cond/layers.0/encoder_attn_layer_norm/beta:0': 'layers.0.encoder_attn_layer_norm.bias',
 'tf_speech2_text_decoder2/cond/layers.0/fc1/kernel:0': 'layers.0.fc1.weight',
 'tf_speech2_text_decoder2/cond/layers.0/fc1/bias:0': 'layers.0.fc1.bias',
 'tf_speech2_text_decoder2/cond/layers.0/fc2/kernel:0': 'layers.0.fc2.weight',
 'tf_speech2_text_decoder2/cond/layers.0/fc2/bias:0': 'layers.0.fc2.bias',
 'tf_speech2_text_decoder2/cond/layers.0/final_layer_norm/gamma:0': 'layers.0.final_layer_norm.weight',
 'tf_speech2_text_decoder2/cond/layers.0/final_layer_norm/beta:0': 'layers.0.final_layer_norm.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/k_proj/kernel:0': 'layers.1.self_attn.k_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/k_proj/bias:0': 'layers.1.self_attn.k_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/q_proj/kernel:0': 'layers.1.self_attn.q_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/q_proj/bias:0': 'layers.1.self_attn.q_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/v_proj/kernel:0': 'layers.1.self_attn.v_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/v_proj/bias:0': 'layers.1.self_attn.v_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/out_proj/kernel:0': 'layers.1.self_attn.out_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn/out_proj/bias:0': 'layers.1.self_attn.out_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn_layer_norm/gamma:0': 'layers.1.self_attn_layer_norm.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/self_attn_layer_norm/beta:0': 'layers.1.self_attn_layer_norm.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/k_proj/kernel:0': 'layers.1.encoder_attn.k_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/k_proj/bias:0': 'layers.1.encoder_attn.k_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/q_proj/kernel:0': 'layers.1.encoder_attn.q_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/q_proj/bias:0': 'layers.1.encoder_attn.q_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/v_proj/kernel:0': 'layers.1.encoder_attn.v_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/v_proj/bias:0': 'layers.1.encoder_attn.v_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/out_proj/kernel:0': 'layers.1.encoder_attn.out_proj.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn/out_proj/bias:0': 'layers.1.encoder_attn.out_proj.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn_layer_norm/gamma:0': 'layers.1.encoder_attn_layer_norm.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/encoder_attn_layer_norm/beta:0': 'layers.1.encoder_attn_layer_norm.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/fc1/kernel:0': 'layers.1.fc1.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/fc1/bias:0': 'layers.1.fc1.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/fc2/kernel:0': 'layers.1.fc2.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/fc2/bias:0': 'layers.1.fc2.bias',
 'tf_speech2_text_decoder2/cond_1/layers.1/final_layer_norm/gamma:0': 'layers.1.final_layer_norm.weight',
 'tf_speech2_text_decoder2/cond_1/layers.1/final_layer_norm/beta:0': 'layers.1.final_layer_norm.bias',
 'tf_speech2_text_decoder2/cond_2/layer_norm/gamma:0': 'layer_norm.weight',
 'tf_speech2_text_decoder2/cond_2/layer_norm/beta:0': 'layer_norm.bias',
 'tf_speech2_text_decoder2/embed_positions/weights:0': 'embed_positions.weights'}

lyrs_pt3 = []
for k, _ in lyrs_tf:
    pt_k = weight_map_rev[k]
    if pt_k == 'none':
        lyrs_pt3 += [(None,0)]
    else:
        lyrs_pt3 += [(pt_k,lyrs_pt2[pt_k])]
lyrs_pt3

layer_weights = []
for t, ((k_tf, sh_tf), (k_pt, sh_pt)) in enumerate(zip(lyrs_tf, lyrs_pt3)):
    if k_pt == None:
        wts = torch.zeros(sh_tf)
    else:
        wts = pt_weights[k_pt].clone() 
        
    if 'pos_proj:0' in k_tf:
        #target_shape (4,128,32)
#         wts = wts.permute(1,0)
        wts = wts.reshape(4,encoder_dim//4,encoder_dim).permute(0,2,1)
    if 'out_proj_kernel:0' in k_tf:
        #target_shape (4,32,128)
        wts = wts.permute(1,0)
        wts = wts.reshape(4,encoder_dim//4,encoder_dim)   
    
    if ('/kernel' in k_tf) or ('/depthwise_kernel' in k_tf):
        if len(wts.shape)==2:
            wts = wts.permute(1,0)
        elif len(wts.shape)==3:
            wts = wts.permute(2, 1,0)
        if 'pw_conv' in k_tf:
            wts = wts[None]
        if 'dw_conv' in k_tf:
            wts = wts[...,None]  
        
    layer_weights.append(wts.cpu().numpy())
    
tf_s2t_decoder.set_weights(layer_weights)


#convert decoder2

tf_s2t_decoder2 = TFSpeech2TextDecoder(cfg.transformer_config)
tf_s2t_decoder2.base_model_prefix='tf_speech2_text_decoder'
tf_s2t_decoder2._keys_to_ignore_on_load_missing = None
tf_s2t_decoder2._keys_to_ignore_on_load_unexpected = None


input_ids = 60 +tf.zeros(2 + 33, dtype = tf.int64)#[None,:]
pos = tf.constant([[2]], dtype=tf.int32)
tf_input_ids = input_ids[None,:pos[0,0]]
encoder_hidden_states = tf.keras.Input((None,cfg.encoder_config.encoder_dim))[0][None]
tf_inputs = {'input_ids':tf_input_ids,'encoder_hidden_states':encoder_hidden_states}
_ = tf_s2t_decoder2(tf_inputs)


pt_weights2 = net2.decoder.decoder.state_dict()

layer_weights2 = []
for t, ((k_tf, sh_tf), (k_pt, sh_pt)) in enumerate(zip(lyrs_tf, lyrs_pt3)):
    if k_pt == None:
        wts = torch.zeros(sh_tf)
    else:
        wts = pt_weights2[k_pt].clone() 
        
    if 'pos_proj:0' in k_tf:
        wts = wts.reshape(4,encoder_dim//4,encoder_dim).permute(0,2,1)
    if 'out_proj_kernel:0' in k_tf:
        wts = wts.permute(1,0)
        wts = wts.reshape(4,encoder_dim//4,encoder_dim)   
    
    if ('/kernel' in k_tf) or ('/depthwise_kernel' in k_tf):
        if len(wts.shape)==2:
            wts = wts.permute(1,0)
        elif len(wts.shape)==3:
            wts = wts.permute(2, 1,0)
        if 'pw_conv' in k_tf:
            wts = wts[None]
        if 'dw_conv' in k_tf:
            wts = wts[...,None]  
        
    layer_weights2.append(wts.cpu().numpy())
    
tf_s2t_decoder2.set_weights(layer_weights2)

print('transfering decoder blocks done')



class DecoderTF(tf.keras.layers.Layer):
    def __init__(
        self,
        decoder_config,
        name="decoder",
        **kwargs,
    ):
        super(DecoderTF, self).__init__(name=name, **kwargs)

        self.config = decoder_config
        self.decoder = tf_s2t_decoder
        self.lm_head = tf.keras.layers.Dense(decoder_config.vocab_size, use_bias=False, name=f"lm_head")
        
        self.bos_token_id = decoder_config.decoder_start_token_id
        self.decoder_pad_token_id = decoder_config.pad_token_id #used for early stopping
        self.eos_token_id= decoder_config.eos_token_id
        self.max_phrase = 33
        self.vocab_size = decoder_config.vocab_size

    def call(self, x, decoder_input_ids=None, training=False, return_dict=False,**kwargs):
                    
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,encoder_hidden_states=x,return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        return lm_logits
    
    def generate(self,enc_hidden, training=False):
    
        input_ids = self.bos_token_id +tf.zeros(2 + self.max_phrase, dtype = tf.int64)[None,]
        
        pos = tf.constant([[2]], dtype=tf.int32)
        decoder_input_ids =  input_ids[:,:pos[0,0]]
        token_scores = []
        stop = tf.constant(False)
        for i in range(self.max_phrase):            
            next_token_scores = self.call(x = enc_hidden, 
                                        decoder_input_ids = decoder_input_ids,
                                        return_dict=False, training=False)[:, -1, :]

            token_scores.append( next_token_scores )
            next_tokens = tf.math.argmax(next_token_scores, -1)
            decoder_input_ids = tf.concat([decoder_input_ids,next_tokens[:,None]],axis=-1)
            pos = pos + 1
        
        # Add an eos token in case there is none
        last_token = tf.one_hot(tf.cast(self.eos_token_id, tf.int32), self.vocab_size)
        token_scores.append( tf.expand_dims(last_token, 0) )
        token_scores = tf.concat(token_scores, axis = 0)
        stop_pos = tf.where(tf.argmax(token_scores, -1)==self.eos_token_id)[0,0] 
        token_scores = token_scores[:stop_pos ]
        predicted_tokens = tf.math.argmax(token_scores, -1)
        return predicted_tokens
    
tf_decoder = DecoderTF(cfg.transformer_config)
_ = tf_decoder.lm_head(encoder_hidden_states) #build
tf_decoder.lm_head = port_weights_to_tf_model(net.decoder.lm_head, tf_decoder.lm_head)




class DecoderTF2(tf.keras.layers.Layer):
    def __init__(
        self,
        decoder_config,
        name="decoder",
        **kwargs,
    ):
        super(DecoderTF2, self).__init__(name=name, **kwargs)

        self.config = decoder_config
        self.decoder = tf_s2t_decoder2
        self.lm_head = tf.keras.layers.Dense(decoder_config.vocab_size, use_bias=False, name=f"lm_head")
        
        self.bos_token_id = decoder_config.decoder_start_token_id
        self.decoder_pad_token_id = decoder_config.pad_token_id #used for early stopping
        self.eos_token_id= decoder_config.eos_token_id
        self.max_phrase = 33
        self.vocab_size = decoder_config.vocab_size

    def call(self, x, decoder_input_ids=None, training=False, return_dict=False,**kwargs):
                    
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,encoder_hidden_states=x,return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        return lm_logits
    
    def generate(self,enc_hidden, training=False):
    
        input_ids = self.bos_token_id +tf.zeros(2 + self.max_phrase, dtype = tf.int64)[None,]
        
        pos = tf.constant([[2]], dtype=tf.int32)
        decoder_input_ids =  input_ids[:,:pos[0,0]]
        token_scores = []
        stop = tf.constant(False)
        for i in range(self.max_phrase):            
            # next_tokens_scores = self.decoder2(enc_hidden, input_ids[None,:pos[0,0]])
            next_token_scores = self.call(x = enc_hidden, 
                                        decoder_input_ids = decoder_input_ids,
                                        return_dict=False, training=False)[:, -1, :]
            # print(input_ids[None,:pos[0,0]])
#             next_token_scores = self.lm_head(next_token_logits)
            token_scores.append( next_token_scores )
            next_tokens = tf.math.argmax(next_token_scores, -1)
#             input_ids = tf.tensor_scatter_nd_update(input_ids, pos, next_tokens)
            decoder_input_ids = tf.concat([decoder_input_ids,next_tokens[:,None]],axis=-1)
            pos = pos + 1
        
        # Add an eos token in case there is none
        last_token = tf.one_hot(tf.cast(self.eos_token_id, tf.int32), self.vocab_size)
        token_scores.append( tf.expand_dims(last_token, 0) )
        token_scores = tf.concat(token_scores, axis = 0)
        stop_pos = tf.where(tf.argmax(token_scores, -1)==self.eos_token_id)[0,0] 
        token_scores = token_scores[:stop_pos ]
        predicted_tokens = tf.math.argmax(token_scores, -1)
        return predicted_tokens
    
    
tf_decoder2 = DecoderTF2(cfg.transformer_config)
_ = tf_decoder2.lm_head(encoder_hidden_states) #build
tf_decoder2.lm_head = port_weights_to_tf_model(net2.decoder.lm_head, tf_decoder2.lm_head)

print('transfering decoder done')


#convert aux stuff
x = tf.keras.layers.Dense(1, use_bias=True, name=f"aux_fc")

inp = tf.keras.Input((None,cfg.encoder_config.encoder_dim))
out = x(inp)
aux_fc_tf =  tf.keras.Model(inp, out)

x2 = tf.keras.layers.Dense(1, use_bias=True, name=f"aux_fc")

inp = tf.keras.Input((None,cfg.encoder_config.encoder_dim))
out = x2(inp)
aux_fc_tf2 =  tf.keras.Model(inp, out)

aux_fc_tf = port_weights_to_tf_model(net.aux_fc, aux_fc_tf)
aux_fc_tf2 = port_weights_to_tf_model(net2.aux_fc, aux_fc_tf2)

print('transfering aux weights done')

ENS_WEIGHTS = [0.5,0.5]

class TFModel(tf.Module):
    def __init__(self):
        super(TFModel, self).__init__()
        
        self.tf_processor = PreprocessingTF()
        self.fea_ext1 = fetf_out_dict['feature_extractor'] # fetf_out
        self.fea_ext2 = fetf_out_dict2['feature_extractor'] # fetf_out2
        
        parts = 'lhand rhand face pose'.split()
        self.parts_tf_indexes = tf_indexes[1:]
        self.fea_ext_parts1 = [fetf_out_dict[f'feature_extractor_{p}'] for p in parts]
        self.fea_ext_parts2 = [fetf_out_dict2[f'feature_extractor_{p}'] for p in parts]
        
        self.encoder = encoder_tf
        self.encoder_b = encoder_tf2
        self.decoder = tf_decoder.decoder
        self.lm_head = tf_decoder.lm_head
        self.decoder2 = tf_decoder2.decoder
        self.lm_head2 = tf_decoder2.lm_head
        #self.cos = tf.convert_to_tensor(net.cos.numpy())
        #self.sin = tf.convert_to_tensor(net.sin.numpy())
        
        self.aux_fc = aux_fc_tf
        self.aux_fc2 = aux_fc_tf2
        self.aux_fc.trainable = False
        self.aux_fc2.trainable = False
        
        self.fea_ext1.trainable = False
        self.fea_ext2.trainable = False
        for i in range(len(self.fea_ext_parts1)):
            self.fea_ext_parts1[i].trainable = False
            self.fea_ext_parts2[i].trainable = False
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.decoder.config.use_cache = False
        self.lm_head.trainable = False
        
        self.encoder_b.trainable = False
        self.decoder2.trainable = False
        self.decoder2.config.use_cache = False
        self.lm_head2.trainable = False
        
        self.max_len = tf.constant(cfg.max_len)
        self.max_phrase = cfg.max_phrase
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.logits_processor = TFLogitsProcessorList()
        self.vocab_size = tf_decoder.config.vocab_size
        self.mode = ''
        self.stop_check = tf.constant(self.eos_token_id, dtype = tf.int64)
        
        dummy = np.zeros((1, self.vocab_size), dtype=np.float32)
        dummy[0,tf_decoder.config.pad_token_id] = 1.
        self.dummy_hidden_states = tf.convert_to_tensor(dummy)
        self.max_len_for_dummy = cfg.max_len_for_dummy
        self.dummy_scores = tf.one_hot(tf.cast(cfg.dummy_phrase_ids, tf.int32), self.vocab_size)
        self.dummy_empty = tf.one_hot(tf.cast([60], tf.int32), self.vocab_size)
        self.ens_weights = ENS_WEIGHTS
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, cfg.n_landmarks, 3], dtype=tf.float32, name='data'),
                                  tf.TensorSpec(shape=[], dtype=tf.int32, name='max_len'),])
    def interpolate_tf(self, data, max_len):        
        diff = max_len - len(data)
        
        if diff <= 0:  # Crop
            data = tf.image.resize(data, (max_len,tf.shape(data)[1]),method=tf.image.ResizeMethod.BILINEAR)
            return data    
        return data
    
    @staticmethod
    def normalize_part(x_in_part_):
        a_mean = tf.expand_dims(tf.reduce_mean(x_in_part_,axis=1),axis=1)
        a_std = tf.expand_dims(tf.math.reduce_std(x_in_part_,axis=1),axis=1)
        x_new = (x_in_part_ - a_mean) / a_std
        x_new = tf.where(tf.math.is_nan(x_new), tf.zeros_like(x_new), x_new)
        x_new = tf.where(tf.math.is_inf(x_new), tf.zeros_like(x_new), x_new)
        x_new = tf.where(tf.reduce_sum(x_in_part_[...,:2],axis=-1)[...,None] == 0, tf.zeros_like(x_new), x_new)
        return x_new
        
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, cfg.max_len, cfg.encoder_config.encoder_dim], dtype=tf.float32, name='x_pre'),
    ])
    def encoder2(self, x_pre):
        return self.encoder(x_pre)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, cfg.max_len, cfg.encoder_config.encoder_dim], dtype=tf.float32, name='x_pre'),
    ])
    def encoder2b(self, x_pre):
        return self.encoder_b(x_pre)
    
    def tf_merge_left_right(self, x):
        x_in_lhand = tf.boolean_mask(x, self.landmark_types==1, axis = 1)# x[:,landmark_types==1]
        x_in_rhand = tf.boolean_mask(x, self.landmark_types==2, axis = 1)#x[:,landmark_types==2]
        x_in_face = tf.boolean_mask(x, self.landmark_types==3, axis = 1)#x[:,landmark_types==3]
        x_in_pose = tf.boolean_mask(x, self.landmark_types==4, axis = 1)#x[:,landmark_types==4]
        
        x_in_hand2 = tf.concat([x_in_lhand, x_in_rhand], axis=-1)
        x_in_face2 = tf.concat([x_in_face[:,0::2,:], x_in_face[:,1::2,:]], axis=-1)
        x_in_pose2 = tf.concat([x_in_pose[:,0::2,:], x_in_pose[:,1::2,:]], axis=-1)
        x_in2 = tf.concat([x_in_hand2, x_in_face2, x_in_pose2], axis=1)
        return x_in2
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, cfg.n_landmarks * 3], dtype=tf.float32, name='inputs'),
    ])
    def call(self, x):

        
        # Preprocess Data
        
        x = tf.cast(x, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, cfg.max_len)), lambda: tf.identity(x))
        x = x[0]
        if tf.shape(x)[0] < self.max_len_for_dummy:
        
            token_scores = self.dummy_scores
        else:
            x_proc = self.tf_processor(x)
            #x_in = self.interpolate_or_pad_tf(x_proc, self.max_len)
            x_in = self.interpolate_tf(x_proc, self.max_len)
            
            x_in_part = tf.transpose(x_in, (1, 0, 2))
            x_in_part = [tf.transpose(tf.gather(x_in_part, iddx), (1, 0, 2)) for iddx in self.parts_tf_indexes ]
            x_in_part = [self.normalize_part(x_in_part_) for x_in_part_ in x_in_part]
            x_ext_part1 = [m(p) for p,m in zip(x_in_part, self.fea_ext_parts1)]
            x_ext_part2 = [m(p) for p,m in zip(x_in_part, self.fea_ext_parts2)]
            x_ext_part1 = tf.concat(x_ext_part1,axis=-1)
            x_ext_part2 = tf.concat(x_ext_part2,axis=-1)
            
            #x_in = tf.expand_dims(x_in, axis = 0)
            xfe_2  = self.fea_ext1(x_in) + x_ext_part1
            xfe_2b = self.fea_ext2(x_in) + x_ext_part2
            
            enc_hidden = self.encoder(xfe_2) # ([xfe_2, self.cos, self.sin])
            enc_hidden2 = self.encoder_b(xfe_2b) # ([xfe_2b, self.cos, self.sin])

            aux_logits = self.aux_fc(enc_hidden[:,0])
            aux_logits2 = self.aux_fc2(enc_hidden2[:,0])
            conf = tf.math.sigmoid(self.ens_weights[0] * aux_logits + self.ens_weights[1] * aux_logits2)
            
            if conf < 0.15:
                token_scores = self.dummy_scores
            else:
                input_ids = self.bos_token_id +tf.zeros(2 + self.max_phrase, dtype = tf.int64)#[None,:]
                pos = tf.constant([[2]], dtype=tf.int32)

                token_scores = []
                stop = tf.constant(False)
                past_key_values=None
                past_key_values2=None
                for i in range(self.max_phrase):            
                    out   = self.decoder(encoder_hidden_states = enc_hidden, 
                                                past_key_values=past_key_values,
                                                input_ids =  input_ids[None,:pos[0,0]][:,-1:],
                                                use_cache =  True,
                                                return_dict=False)
                    next_token_logits = out[0]
                    past_key_values = out[1]
                    next_token_logits = next_token_logits[:, -1, :]

                    out2   = self.decoder2(encoder_hidden_states = enc_hidden2, 
                                                past_key_values=past_key_values2,
                                                input_ids =  input_ids[None,:pos[0,0]][:,-1:],
                                                use_cache =  True,
                                                return_dict=False)


                    next_token_logits2 = out2[0]
                    past_key_values2 = out2[1]
                    next_token_logits2 = next_token_logits2[:, -1, :]
                    skip = tf.reduce_any(self.stop_check==input_ids)
                    next_token_scores = tf.cond(skip, 
                                                lambda: self.dummy_hidden_states,
                                                lambda: self.ens_weights[0] * self.lm_head(next_token_logits) + self.ens_weights[1] * self.lm_head2(next_token_logits2))
                    '''
                    next_token_scores = tf.cond(skip, 
                                                lambda: self.dummy_hidden_states,
                                                lambda: self.lm_head(next_token_logits))
                    '''


                    token_scores.append( next_token_scores )
                    next_tokens = tf.math.argmax(next_token_scores, -1)
                    input_ids = tf.tensor_scatter_nd_update(input_ids, pos, next_tokens)
                    pos = pos + 1

                # Add an eos token in case there is none
                last_token = tf.one_hot(tf.cast(self.eos_token_id, tf.int32), self.vocab_size)
                token_scores.append( tf.expand_dims(last_token, 0) )
                token_scores = tf.concat(token_scores, axis = 0)
                stop_pos = tf.where(tf.argmax(token_scores, -1)==self.eos_token_id)[0,0] 
                token_scores = token_scores[:stop_pos ]

        

        out = {'outputs': token_scores}  

        
        return out


mod3tf = TFModel()
mod3tf.trainable = False

print('building of final ensemble model  done')

# # %%


# # idx = 0

# # row = val_df.iloc[idx]
# # file_id, sequence_id, phrase = row[['file_id','sequence_id','phrase']]
# # try:
# #     data = np.load(f'/raid/asl-fingerspelling/train_landmarks_v3/{file_id}/{sequence_id}.npy')
# # except:
# #     data = np.load(f'datamount/train_landmarks_v3/{file_id}/{sequence_id}.npy')
# # # data = tf.convert_to_tensor(data_np)
# # n_frames = len(data)
# # x = tf.convert_to_tensor(data)


# # %%





# # %%


# from tqdm import tqdm
# val_strs = []
# pred_strs = []
# gt_strs = []
# # for idx in tqdm(range(len(val_df))):
# for idx in tqdm(range(500)):
#     #idx = 12
#     row = val_df.iloc[idx]
#     file_id, sequence_id, phrase = row[['file_id','sequence_id','phrase']]
#     try:
#         data = np.load(f'/raid/asl-fingerspelling/train_landmarks_v3/{file_id}/{sequence_id}.npy')
#     except:
#         data = np.load(f'datamount/train_landmarks_v3/{file_id}/{sequence_id}.npy')
#     data = tf.convert_to_tensor(data)
#     n_frames = len(data)
#     x = tf.convert_to_tensor(data)
#     pred_scores = mod3tf.call(x)['outputs']
#     pred = tf.argmax(pred_scores, -1)
#     pred_str = decode_pred(pred.numpy())
#     if pred_str==' ': pred_str=''
#     #print(f'Actual    [{n_frames}]: {phrase}')
#     #print(f'Predicted [{n_frames}]: {pred_str}')
#     val_str = decode_pred(val_data['generated_ids'][idx])
#     #print(f'Val data  [{n_frames}]: {val_str}')
#     val_strs += [val_str]
#     pred_strs += [pred_str]
#     gt_strs += [phrase]

# # %%



# get_score(gt_strs, val_strs), get_score(gt_strs, pred_strs), get_score(val_strs, pred_strs)



prefix = 'datamount/weights/'
tf_file = f'{prefix}{config_name}/fold{FOLD}/tflite_model.tfl.path'

tf.saved_model.save(mod3tf, tf_file, signatures={'serving_default': mod3tf.call})



# # %%


tflite_model_path  = f'{prefix}{config_name}/fold{FOLD}/model.tflite'



# # %%
    

def save_tflite_model(tf_file, tflite_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_file)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.allow_custom_ops=True
    tf_lite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tf_lite_model)

save_tflite_model(tf_file , tflite_model_path)



# %%


inference_args_fn = f'{prefix}{config_name}/fold{FOLD}/inference_args.json'


# %%


with open(cfg.data_folder + 'inference_args.json', "r") as f:
    columns = json.load(f)['selected_columns']
with open(inference_args_fn, 'w') as f:
     json.dump({ 'selected_columns': columns }, f)



print('final model saved under')
print(tflite_model_path, inference_args_fn)
