#Several tensorflow SqueezeFormer components where copied/ adapted from https://github.com/kssteven418/Squeezeformer

import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
import typing
from torch import Tensor
import math
import numpy as np
import tensorflow as tf
import json


def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


from transformers.models.speech_to_text import Speech2TextConfig, Speech2TextForConditionalGeneration
from transformers.models.speech_to_text.modeling_speech_to_text import shift_tokens_right, Speech2TextDecoder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FeedForwardModuleTF(tf.keras.layers.Layer):
    def __init__(
        self,
        encoder_dim=512,
        expansion_factor=4,
        dropout_p=0.1,
        name="ff_module",
        **kwargs,
    ):
        super(FeedForwardModuleTF, self).__init__(name=name, **kwargs)

        self.ffn1 = tf.keras.layers.Dense(
            expansion_factor * encoder_dim, name=f"ffn1",

        )
        self.act = tf.keras.layers.Activation(tf.nn.swish, name=f"act")
        self.do1 = tf.keras.layers.Dropout(dropout_p, name=f"do1")
        self.ffn2 = tf.keras.layers.Dense(
            encoder_dim, name=f"ffn2",
        )
        self.do2 = tf.keras.layers.Dropout(dropout_p, name=f"do2")

    def call(self, x, training=False, **kwargs):
        x = self.ffn1(x, training=training)
        x = self.act(x)
        x = self.do1(x, training=training)
        x = self.ffn2(x, training=training)
        x = self.do2(x, training=training)
        return x



class RelPositionalEncodingTF(tf.keras.layers.Layer):
    '''
    Same positional encoding method as NeMo library
    '''
    def __init__(self, d_model, max_len=5000, name="positional_encoding_nemo", **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.max_len = max_len
        positions = tf.expand_dims(tf.range(self.max_len - 1, -max_len, -1.0, dtype=tf.float32), axis=1)
        pos_length = tf.shape(positions)[0]
        pe = np.zeros([pos_length, d_model], 'float32')
        div_term = np.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        pe = tf.convert_to_tensor(pe)
        self.pe = tf.expand_dims(pe, 0)

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, length, dmodel = shape_list(inputs)
        center_pos = tf.shape(self.pe)[1] // 2
        start_pos = center_pos - length + 1
        end_pos = center_pos + length
        pos_emb = self.pe[:, start_pos:end_pos]
        return tf.cast(pos_emb, dtype=inputs.dtype)

    def get_config(self):
        conf = super().get_config()
        return conf.update({"max_len": self.max_len})



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        head_size,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self._droput_rate = dropout


    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )
        input_max = (self.num_heads * self.head_size) ** -0.5
        self.query = tf.keras.layers.Dense(
            self.num_heads * self.head_size, activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
            name='query_proj'
        )
        self.key = tf.keras.layers.Dense(
            self.num_heads * self.head_size, activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
            name='key_proj'
        )
        self.value = tf.keras.layers.Dense(
            self.num_heads * self.head_size, activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
            name = 'value_proj'
        )
        self.projection_kernel = self.add_weight(
            name="out_proj_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
        )
        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="out_proj_bias",
                shape=[output_size],
                initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
            )
        else:
            self.projection_bias = None

    def call_qkv(self, query, key, value, training=False):
        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to "
                "the same as the number of elements in 'value'"
            )
        # Linear transformations
        query = self.query(query)
        B, T, E = shape_list(query)
        query = tf.reshape(query, [B, T, self.num_heads, self.head_size])

        key = self.key(key)
        B, T, E = shape_list(key)
        key = tf.reshape(key, [B, T, self.num_heads, self.head_size])

        value = self.value(value)
        B, T, E = shape_list(value)
        value = tf.reshape(value, [B, T, self.num_heads, self.head_size])

        return query, key, value

    def call_attention(self, query, key, value, logits, training=False, mask=None):
        # mask = attention mask with shape [B, Tquery, Tkey] with 1 is for positions we want to attend, 0 for masked
        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have at least 2 dimensions")
            if query.shape[-3] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to "
                    "the number of elements in 'query'"
                )
            if key.shape[-3] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )
        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -65504 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum("...NHI,HIO->...NO", multihead_output, self.projection_kernel)

        if self.projection_bias is not None:
            output += self.projection_bias

        return output, attn_coef

    def call(self, inputs, training=False, mask=None, **kwargs):
        query, key, value = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        output, attn_coef = self.call_attention(query, key, value, logits,
                                                training=training, mask=mask)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._droput_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
        )

        return config


class RelPositionMultiHeadAttentionTF(MultiHeadAttention):
    def __init__(self, kernel_sizes=None, strides=None, **kwargs):
        super(RelPositionMultiHeadAttentionTF, self).__init__(**kwargs)
        
    def build(self, input_shape):
        num_pos_features = input_shape[-1][-1]
        input_max = (self.num_heads * self.head_size) ** -0.5
        self.pos_kernel = self.add_weight(
            name="pos_proj",
            shape=[self.num_heads, num_pos_features, self.head_size],
            initializer=tf.keras.initializers.RandomUniform(minval=-input_max, maxval=input_max),
        )
        self.pos_bias_u = self.add_weight(
            name="u_bias",
            shape=[self.num_heads, self.head_size],
            initializer=tf.keras.initializers.Zeros(),
        )
        self.pos_bias_v = self.add_weight(
            name="v_bias",
            shape=[self.num_heads, self.head_size],
            initializer=tf.keras.initializers.Zeros(),
        )
        super(RelPositionMultiHeadAttentionTF, self).build(input_shape[:-1])

    @staticmethod
    def relative_shift(x):
        x_shape = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
        x = tf.reshape(x[:, :, 1:, :], x_shape)
        return x

    def call(self, inputs, training=False, mask=None, **kwargs):
        query, key, value, pos = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        pos = tf.einsum("...MI,HIO->...MHO", pos, self.pos_kernel)

        query_with_u = query + self.pos_bias_u
        query_with_v = query + self.pos_bias_v

        logits_with_u = tf.einsum("...NHO,...MHO->...HNM", query_with_u, key)
        logits_with_v = tf.einsum("...NHO,...MHO->...HNM", query_with_v, pos)
        logits_with_v = self.relative_shift(logits_with_v)

        logits = logits_with_u + logits_with_v[:, :, :, :tf.shape(logits_with_u)[3]]

        depth = tf.constant(self.head_size, dtype=tf.float32)
        logits /= tf.sqrt(depth)

        output, attn_coef = self.call_attention(query, key, value, logits,
                                                training=training, mask=mask)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output




class MultiHeadedSelfAttentionModuleTF(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout_p=0.1,
        name="mhsa_module",
        **kwargs,
    ):
        super(MultiHeadedSelfAttentionModuleTF, self).__init__(name=name, **kwargs)

        
        self.positional_encoding = RelPositionalEncodingTF(d_model)
        self.attention = RelPositionMultiHeadAttentionTF(name=f"attention",head_size=d_model//num_heads, num_heads=num_heads)
        self.dropout = tf.keras.layers.Dropout(dropout_p, name=f"{name}_dropout")

    def call(self, inputs, training=False, mask=None, **kwargs):
        
        pos = self.positional_encoding(inputs)
        outputs = self.attention([inputs, inputs, inputs, pos], training=training, mask=mask)
        outputs = self.dropout(outputs, training=training)
        return outputs


class GLUTF(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 name="glu_activation",
                 **kwargs):
        super(GLUTF, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)

    def get_config(self):
        conf = super(GLU, self).get_config()
        conf.update({"axis": self.axis})
        return conf

class ConvModuleTF(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        kernel_size=31,
        dropout_p=0.1,
        expansion_factor=2,
        name="conv_module",
        **kwargs,
    ):
        super(ConvModuleTF, self).__init__(name=name, **kwargs)

        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=expansion_factor * in_channels, kernel_size=1, strides=1,
            padding="valid", name=f"{name}_pw_conv_1",
        )
        self.act1 = GLUTF(name=f"{name}_act_1")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1), strides=1,
            padding="same", name=f"{name}_dw_conv",
            depth_multiplier=1,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn",
            momentum=0.985, epsilon=1e-5
        )
        self.act2 = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_act_2")
        self.pw_conv_2 = tf.keras.layers.Conv2D(
            filters=in_channels, kernel_size=1, strides=1,
            padding="valid", name=f"{name}_pw_conv_2",
        )
        self.do = tf.keras.layers.Dropout(dropout_p, name=f"{name}_dropout")

    def call(self, outputs, training=False, **kwargs):

        B, T, E = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, 1, E])
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.act1(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.act2(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.do(outputs, training=training)
        return outputs

def make_scale_tf(encoder_dim, name = 'mhsa'):
    scale = tf.Variable(initial_value=tf.ones([1, 1, encoder_dim]), trainable=False, name = f'scale_{name}')
    bias = tf.Variable(initial_value=tf.zeros([1, 1, encoder_dim]), trainable=False, name = f'bias_{name}')
    return scale, bias


class SqueezeformerBlockTF(tf.keras.layers.Layer):
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        name="squeezeformer_block",
        **kwargs,
    ):
        super(SqueezeformerBlockTF, self).__init__(name=name, **kwargs)
        
        
        self.scale_mhsa = tf.Variable(initial_value=tf.ones([1, 1, encoder_dim]), trainable=False, name = f'{name}/scale_mhsa')
        self.bias_mhsa = tf.Variable(initial_value=tf.zeros([1, 1, encoder_dim]), trainable=False, name = f'{name}/bias_mhsa')

        self.scale_ff_mhsa = tf.Variable(initial_value=tf.ones([1, 1, encoder_dim]), trainable=False, name = f'{name}/scale_ff_mhsa')
        self.bias_ff_mhsa = tf.Variable(initial_value=tf.zeros([1, 1, encoder_dim]), trainable=False, name = f'{name}/bias_ff_mhsa')

        self.scale_conv = tf.Variable(initial_value=tf.ones([1, 1, encoder_dim]), trainable=False, name = f'{name}/scale_conv')
        self.bias_conv = tf.Variable(initial_value=tf.zeros([1, 1, encoder_dim]), trainable=False, name = f'{name}/bias_conv')

        self.scale_ff_conv = tf.Variable(initial_value=tf.ones([1, 1, encoder_dim]), trainable=False, name = f'{name}/scale_ff_conv')
        self.bias_ff_conv = tf.Variable(initial_value=tf.zeros([1, 1, encoder_dim]), trainable=False, name = f'{name}/bias_ff_conv')

        # self.mhsa = MultiHeadedSelfAttentionModuleTF(d_model=encoder_dim,num_heads=num_attention_heads, name='mhsa')
        # encoder_dim = 144; num_attention_heads = 4
        self.mhsa_llama = TFLlamaAttention(LlamaConfig(hidden_size = encoder_dim, 
                                       num_attention_heads = num_attention_heads, 
                                       max_position_embeddings = 384))
        
        self.ln_mhsa = tf.keras.layers.LayerNormalization(name=f"ln_mhsa",epsilon=1e-05)
        self.ff_mhsa = FeedForwardModuleTF(encoder_dim=encoder_dim,
                                           expansion_factor=feed_forward_expansion_factor,
                                           dropout_p=feed_forward_dropout_p,name='ff_mhsa')
        self.ln_ff_mhsa = tf.keras.layers.LayerNormalization(name=f"ln_ff_mhsa",epsilon=1e-05)
        self.conv = ConvModuleTF(in_channels=encoder_dim,
                                 kernel_size=conv_kernel_size,
                                 expansion_factor=conv_expansion_factor,
                                 dropout_p=conv_dropout_p,name='conv')
        self.ln_conv = tf.keras.layers.LayerNormalization(name=f"ln_conv",epsilon=1e-05)
        self.ff_conv = FeedForwardModuleTF(encoder_dim=encoder_dim,
                                           expansion_factor=feed_forward_expansion_factor,
                                           dropout_p=feed_forward_dropout_p,name='ff_conv')
        self.ln_ff_conv = tf.keras.layers.LayerNormalization(name=f"ln_ff_conv",epsilon=1e-05)
        
        
    def call(self, x, cos, sin, training=False, mask=None, **kwargs):
        
        
        residual = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = residual + self.mhsa_llama(x, cos, sin, training=training)
        x = self.ln_mhsa(x, training=training)
        
        residual = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = residual + self.ff_mhsa(x, training=training)
        x = self.ln_ff_mhsa(x, training=training)        
        
        residual = x
        x = x * self.scale_conv + self.bias_conv
        x = residual + self.conv(x, training=training)
        x = self.ln_conv(x, training=training)     
        
        residual = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = residual + self.ff_conv(x, training=training)
        x = self.ln_ff_conv(x, training=training)
        
        return x



class SqueezeformerEncoderTF(tf.keras.Model):
    def __init__(
        self,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        name='sequeeze_former_encoder',
        **kwargs
    ):
        super(SqueezeformerEncoderTF, self).__init__(name=name, **kwargs)
        rotary_emb = LlamaRotaryEmbedding(encoder_dim//num_attention_heads, max_position_embeddings=384)
        self.cos = tf.convert_to_tensor(rotary_emb.cos_cached.numpy()) #[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        self.sin = tf.convert_to_tensor(rotary_emb.sin_cached.numpy()) #[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        
        self.blocks = []
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlockTF(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    name=f'squeezeformer_block_{idx}'
                )
            )
            
    def call(self, inputs, training=False, mask=None, **kwargs):   
        
        for idx, block in enumerate(self.blocks):
            inputs = block(inputs, self.cos, self.sin)
        return inputs



class tf_FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, out_dim, in_channels):
        super().__init__()
        
        self.in_channels = in_channels
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
        xc = tf.reshape(xc, [1, -1, self.in_channels])
    
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





# https://github.com/huggingface/transformers/blob/tf_llama_port/src/transformers/models/llama/modeling_tf_llama.py
def tf_rotate_half(x):
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat((-x2, x1), axis=-1)


def tf_apply_rotary_pos_emb(q, k, cos, sin):

    q = tf.expand_dims(q, 1)  
    k = tf.expand_dims(k, 1)  
    q_embed = (q * cos) + (tf_rotate_half(q) * sin)
    k_embed = (k * cos) + (tf_rotate_half(k) * sin)
    q_embed = tf.squeeze(q_embed, (1))  
    k_embed = tf.squeeze(k_embed, (1))  
    return q_embed, k_embed


class TFLlamaAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="v_proj")
        self.o_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name="o_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        cos: tf.Tensor,
        sin: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        bsz, q_len, _ = shape_list(hidden_states)

        query_states = self._shape(self.q_proj(hidden_states), q_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), q_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), q_len, bsz)

        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            seq_offset = shape_list(past_key_value[0])[-2]
            kv_seq_len += seq_offset
        else:
            seq_offset = 0

        query_states, key_states = tf_apply_rotary_pos_emb(query_states, key_states, cos[:, :, :kv_seq_len], sin[:, :, :kv_seq_len])
        if past_key_value is not None:
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = tf.matmul(query_states, tf.transpose(key_states, perm=[0, 1, 3, 2])) / tf.math.sqrt(
            float(self.head_dim)
        )

        tf.debugging.assert_equal(
            shape_list(attn_weights),
            [bsz, self.num_heads, q_len, kv_seq_len],
            message=f"Attention weights should be of size {[bsz, self.num_heads, q_len, kv_seq_len]}, but is {shape_list(attn_weights)}",
        )


        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_output = tf.matmul(attn_weights, value_states)

        tf.debugging.assert_equal(
            shape_list(attn_output),
            [bsz, self.num_heads, q_len, self.head_dim],
            message=f"`attn_output` should be of size {[bsz, self.num_heads, q_len, self.head_dim]}, but is {shape_list(attn_output)}",
        )

        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output 


