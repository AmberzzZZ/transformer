from keras.layers import Input, Conv2D, Reshape, add, Dropout, Dense, Lambda, concatenate
from WMSA import WindowMultiHeadAttention, FeedForwardNetwork, gelu
from  swin import Pad_HW, basicStage, PatchMerging, SwinTransformerBlock
from LayerNormalization import LayerNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf
import math


def SwinUnet(input_shape=(224,224,3), patch_size=4, emb_dim=96, ape=False, n_classes=1000,    # in/out hypers
             num_layers=[2,2,2,2], num_heads=[3,6,12,24],                            # structual hypers
             window_size=7, qkv_bias=True, qk_scale=None, mlp_ratio=4,               # swin-block hypers
             attn_drop=0., ffn_drop=0., residual_drop=0.2):

    inpt = Input(input_shape)

    # patch embedding
    x = Conv2D(emb_dim, patch_size, strides=patch_size, padding='same')(inpt)    # (b,h/4,w/4,C)
    H, W = (math.ceil(input_shape[0]/patch_size), math.ceil(input_shape[1]/patch_size))
    x = LayerNormalization()(x)      # [b,T,D]

    # absolute positional embeddings
    if ape:
        pe = Lambda(lambda x: tf.truncated_normal((H,W,emb_dim), mean=0.0, stddev=.02))(x)  # (1,H,W,D)
        x = add([x, pe])   # [b,N+1,D]

    x = Dropout(ffn_drop)(x)

    n_stages = len(num_layers)
    dbr = np.linspace(0, residual_drop, num=sum(num_layers))    # drop block rate

    # -------- encoder --------
    block_idx = 0
    skips = []
    skip_shapes = []
    for i in range(n_stages):
        merging = True if i<n_stages-1 else False
        # pad on the top
        pad_l = pad_t = 0
        pad_b, pad_r = (window_size - H % window_size) % window_size, (window_size - W % window_size) % window_size
        # x = Lambda(lambda x: tf.pad(x, [[0,0],[pad_t, pad_b], [pad_l, pad_r], [0,0]]))(x)
        x = Pad_HW(pad_t, pad_b, pad_l, pad_r)(x)
        H += pad_b
        W += pad_r
        # WSA+SWSA: 2 blocks
        x, skip = basicStage(x, emb_dim, (H,W), num_layers[i]//2, num_heads[i], window_size, mlp_ratio, qkv_bias,
                             attn_drop, ffn_drop, residual_drop=dbr[sum(num_layers[:i]):sum(num_layers[:i+1])],
                             patch_merging=merging, idx=block_idx, stage=i)
        skips.append(skip)
        skip_shapes.append([H,W])
        emb_dim *= 2
        block_idx += num_layers[i]//2
        if merging:
            H = (H+1) // 2
            W = (W+1) // 2

    x = LayerNormalization()(x)     # [H/32*W/32,8C]

    # -------- decoder --------
    for i in range(n_stages):
        emb_dim = emb_dim//2
        if i==0:
            x = Lambda(PatchExpand, arguments={'feature_shape': (H,W), 'emb_dim': emb_dim}, name='PatchExpand%d' % (n_stages-1-i))(x)
        else:
            expand = True if i<n_stages-1 else False
            skip_H, skip_W = skip_shapes[n_stages-1-i]
            pad_b, pad_r = H-skip_H, W-skip_W
            skip = Pad_HW(0,pad_b,0,pad_r)(skips[n_stages-1-i])
            x = basicUpStage(x, skip, emb_dim, (H,W),
                             num_layers[n_stages-1-i]//2, num_heads[n_stages-1-i],
                             window_size, mlp_ratio, qkv_bias, attn_drop, ffn_drop,
                             residual_drop=dbr[sum(num_layers[:n_stages-1-i]):sum(num_layers[:n_stages-i])],
                             patch_expand=expand, idx=block_idx, stage=n_stages-1-i)
            block_idx += num_layers[n_stages-1-i]//2
        if i<n_stages-1:
            H, W = H*2, W*2

    # head
    x = Lambda(up_x4, arguments={'feature_shape': (H,W), 'emb_dim': emb_dim})(x)    # (b,h,w,emb_dim)
    x = Conv2D(n_classes, 1, strides=1, padding='same', activation='sigmoid', use_bias=False)(x)   # official false bias,  (b,h,w,cls_dim)

    model = Model(inpt, x)

    return model


# alternative [swin_block + patch_merging] for each stage
def basicStage(x, emb_dim, feature_shape, depth, n_heads, window_size, mlp_ratio=4, qkv_bias=True,
               attn_drop=0., ffn_drop=0., residual_drop=[], patch_merging=False, idx=None, stage=None):
    # assert depth==len(residual_drop)
    # swin blocks
    for i in range(depth):
        x = SwinTransformerBlock(emb_dim, feature_shape, n_heads, window_size, mlp_ratio, qkv_bias,
                                 attn_drop, ffn_drop, residual_drop[i], idx=idx+i)(x)
    skip = x
    # downsampling
    if patch_merging:
        x = Lambda(PatchMerging, arguments={'feature_shape': feature_shape, 'emb_dim': emb_dim}, name='PatchMerging%d' % stage)(x)
    return x, skip


# alternative [swin_block + patch_expand] for each stage
def basicUpStage(x, skip, emb_dim, feature_shape, depth, n_heads, window_size, mlp_ratio=4, qkv_bias=True,
                 attn_drop=0., ffn_drop=0., residual_drop=[], patch_expand=False, idx=None, stage=None):
    # assert depth==len(residual_drop)
    # concat
    x = concatenate([x,skip])
    # linear
    x = Dense(emb_dim, use_bias=False)(x)

    # swin blocks
    for i in range(depth):
        x = SwinTransformerBlock(emb_dim, feature_shape, n_heads, window_size, mlp_ratio, qkv_bias,
                                 attn_drop, ffn_drop, residual_drop[i], idx=idx+i)(x)
    # upsampling
    if patch_expand:
        x = Lambda(PatchExpand, arguments={'feature_shape': feature_shape, 'emb_dim': emb_dim}, name='PatchExpand%d' % stage)(x)
    return x


def PatchExpand(x, feature_shape, emb_dim):
    # input: (b,H,W,D)
    # output: (b,2H,2W,D//2)

    h, w = feature_shape

    x = Dense(emb_dim*2, use_bias=False)(x)
    x = Reshape((h,w,2,2,emb_dim//2))(x)    # (b,h,w,2,2,c//2)
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = Reshape((h*2,w*2,emb_dim//2))(x)   # (b,2h,2w,c//2)

    return x


def up_x4(x, feature_shape, emb_dim):
    # input: (b,H,W,D)
    # output: (b,4H,4W,D)

    h, w = feature_shape

    x = Dense(emb_dim*16, use_bias=False)(x)
    x = Reshape((h,w,4,4,emb_dim))(x)    # (b,h,w,4,4,c)
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = Reshape((h*4,w*4,emb_dim))(x)   # (b,4h,4w,c)

    return x


if __name__ == '__main__':

    model = SwinUnet((256,412,3), patch_size=4, window_size=7)
    model.summary()






