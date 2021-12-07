from keras.layers import Input, Conv2D, Reshape, add, Dropout, Dense, Lambda, concatenate
from WMSA import WindowMultiHeadAttention, FeedForwardNetwork, gelu
from LayerNormalization import LayerNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf


def SwinUnet(input_size=224, patch_size=4, emb_dim=96, ape=False, n_classes=1000,    # in/out hypers
             num_layers=[2,2,2,2], num_heads=[3,6,12,24],                            # structual hypers
             window_size=7, qkv_bias=True, qk_scale=None, mlp_ratio=4,               # swin-block hypers
             attn_drop=0., ffn_drop=0., residual_drop=0.2):

    inpt = Input((input_size, input_size, 3))
    assert input_size%7==0 and input_size%16==0, 'input_size can not be divided clean'

    # patch embedding
    x = Conv2D(emb_dim, patch_size, strides=patch_size, padding='same')(inpt)    # (b,h/4,w/4,C)
    N = input_size//patch_size        # grid_size
    x = Reshape((N*N, emb_dim))(x)
    x = LayerNormalization()(x)      # [b,T,D]

    # absolute positional embeddings
    if ape:
        pe = Lambda(lambda x: tf.truncated_normal((1,tf.shape(x)[1],tf.shape(x)[2]), mean=0.0, stddev=.02))(x)  # [1,T,D]
        x = add([x, pe])   # [b,N+1,D]

    x = Dropout(ffn_drop)(x)

    n_stages = len(num_layers)
    dbr = np.linspace(0, residual_drop, num=sum(num_layers))    # drop block rate

    # -------- encoder --------
    feature_size = N      # start from s4
    block_idx = 0
    skips = []
    for i in range(n_stages):
        merging = True if i<n_stages-1 else False
        x, skip = basicStage(x, emb_dim, feature_size, num_layers[i], num_heads[i], window_size, mlp_ratio, qkv_bias,
                             attn_drop, ffn_drop, residual_drop=dbr[sum(num_layers[:i]):sum(num_layers[:i+1])],
                             patch_merging=merging, idx=block_idx, stage=i)
        skips.append(skip)
        emb_dim *= 2
        block_idx += num_layers[i]
        if merging:
            feature_size //= 2
    x = LayerNormalization()(x)     # [H/32*W/32,8C]

    # -------- decoder --------
    for i in range(n_stages):
        emb_dim = emb_dim//2
        if i==0:
            x = Lambda(PatchExpand, arguments={'feature_size': feature_size}, name='PatchExpand%d' % (n_stages-1-i))(x)
        else:
            expand = True if i<n_stages-1 else False
            x = basicUpStage(x, skips[n_stages-1-i], emb_dim, feature_size,
                             num_layers[n_stages-1-i], num_heads[n_stages-1-i],
                             window_size, mlp_ratio, qkv_bias, attn_drop, ffn_drop,
                             residual_drop=dbr[sum(num_layers[:n_stages-1-i]):sum(num_layers[:n_stages-i])],
                             patch_expand=expand, idx=block_idx, stage=n_stages-1-i)
            block_idx += num_layers[n_stages-1-i]
        if i<n_stages-1:
            feature_size *= 2

    # head
    x = Lambda(up_x4, arguments={'feature_size': feature_size})(x)    # (b,h,w,emb_dim)
    x = Conv2D(n_classes, 1, strides=1, padding='same', activation='sigmoid', use_bias=False)(x)   # official false bias,  (b,h,w,cls_dim)

    model = Model(inpt, x)

    return model


# alternative [swin_block + patch_merging] for each stage
def basicStage(x, emb_dim, feature_size, depth, n_heads, window_size, mlp_ratio=4, qkv_bias=True,
               attn_drop=0., ffn_drop=0., residual_drop=[], patch_merging=False, idx=None, stage=None):
    assert depth==len(residual_drop)
    # swin blocks
    for i in range(depth):
        x = SwinTransformerBlock(emb_dim, feature_size, n_heads, window_size, mlp_ratio, qkv_bias,
                                 attn_drop, ffn_drop, residual_drop[i], idx=idx+i)(x)
    skip = x
    # downsampling
    if patch_merging:
        x = Lambda(PatchMerging, arguments={'feature_size': feature_size}, name='PatchMerging%d' % stage)(x)
    return x, skip


# alternative [swin_block + patch_expand] for each stage
def basicUpStage(x, skip, emb_dim, feature_size, depth, n_heads, window_size, mlp_ratio=4, qkv_bias=True,
                 attn_drop=0., ffn_drop=0., residual_drop=[], patch_expand=False, idx=None, stage=None):
    assert depth==len(residual_drop)
    # concat
    x = concatenate([x,skip])
    # linear
    x = Dense(emb_dim, use_bias=False)(x)
    # swin blocks
    for i in range(depth):
        x = SwinTransformerBlock(emb_dim, feature_size, n_heads, window_size, mlp_ratio, qkv_bias,
                                 attn_drop, ffn_drop, residual_drop[i], idx=idx+i)(x)
    # upsampling
    if patch_expand:
        x = Lambda(PatchExpand, arguments={'feature_size': feature_size}, name='PatchExpand%d' % stage)(x)
    return x


class SwinTransformerBlock(Model):
    def __init__(self, emb_dim, feature_size, n_heads, window_size, mlp_ratio=4, qkv_bias=True,
                 attn_drop=0., ffn_drop=0., residual_drop=0., idx=None, **kwargs):
        super(SwinTransformerBlock, self).__init__(name='STB_%d' % idx, **kwargs)
        self.feature_size = feature_size
        self.window_size = window_size
        self.shift_size = window_size//2

        # W-MSA
        self.ln1 = LayerNormalization()
        self.wmsa = WindowMultiHeadAttention(emb_dim, n_heads, window_size, qkv_bias, attn_drop, ffn_drop)
        self.res_drop1 = Dropout(residual_drop, noise_shape=(None, 1, 1))

        self.ln2 = LayerNormalization()
        self.ffn = FeedForwardNetwork(emb_dim*mlp_ratio, emb_dim, activation=gelu, drop_rate=ffn_drop)
        self.res_drop2 = Dropout(residual_drop, noise_shape=(None, 1, 1))

        # SW-MSA
        self.ln3 = LayerNormalization()
        self.wmsa_s = WindowMultiHeadAttention(emb_dim, n_heads, window_size, qkv_bias, attn_drop, ffn_drop)
        self.res_drop3 = Dropout(residual_drop, noise_shape=(None, 1, 1))

        self.ln4 = LayerNormalization()
        self.ffn_s = FeedForwardNetwork(emb_dim*mlp_ratio, emb_dim, activation=gelu, drop_rate=ffn_drop)
        self.res_drop4 = Dropout(residual_drop, noise_shape=(None, 1, 1))

    def call(self, x):
        # input_shape: [b,T,D]

        # WMSA block
        inpt = x
        x = self.ln1(x)
        x = WindowPartition(x, self.feature_size, self.window_size)   # [b,nW,local_L,D]
        x = self.wmsa(x)
        x = WindowReverse(x, self.feature_size, self.window_size)      # [b,T,D]
        x = self.res_drop1(x)
        x = x + inpt

        inpt = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.res_drop2(x)
        x = x + inpt

        # SWMSA block
        inpt = x
        x = self.ln3(x)
        x = CyclicShift(x, self.feature_size, self.shift_size)
        x = WindowPartition(x, self.feature_size, self.window_size)   # [b,nW,local_L,D]
        x = self.wmsa_s(x)
        x = WindowReverse(x, self.feature_size, self.window_size)      # [b,T,D]
        x = CyclicShift(x, self.feature_size, -self.shift_size)
        x = self.res_drop3(x)
        x = x + inpt

        inpt = x
        x = self.ln4(x)
        x = self.ffn_s(x)
        x = self.res_drop4(x)
        x = x + inpt

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def WindowPartition(x, feature_size, window_size):
    # input: [b,L,D]
    # output: [b,nW,local_L,D]
    b = tf.shape(x)[0]
    c = K.int_shape(x)[-1]
    n_windows = feature_size//window_size
    x = tf.reshape(x, (b, n_windows, window_size, n_windows, window_size, c))
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = tf.reshape(x, (b, n_windows*n_windows, window_size*window_size, c))   # [b,nW,Wh*Ww,D]
    return x


def WindowReverse(x, feature_size, window_size):
    # input: [b,nW,local_L,D]
    # output: [b,L,D]
    b = tf.shape(x)[0]
    c = K.int_shape(x)[-1]
    nW = feature_size//window_size
    x = tf.reshape(x, (b, nW, nW, window_size, window_size, c))
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = tf.reshape(x, (b, nW*window_size*nW*window_size, c))
    return x


def CyclicShift(x, feature_size, shift_size):
    # input: [b, L, D]
    b = tf.shape(x)[0]
    c = K.int_shape(x)[-1]
    x = tf.reshape(x, (b, feature_size, feature_size, c))
    x = tf.manip.roll(x, shift=[shift_size, shift_size], axis=[1,2])
    x = tf.reshape(x, (b, feature_size*feature_size, c))
    return x


def PatchMerging(x, feature_size):
    # input: (b,hw,c), output: (b,h//2*w//2,2c)

    b, hw, c = K.int_shape(x)
    assert feature_size*feature_size==hw, "dim not match"
    h = w = feature_size
    assert h%2==0 and w%2==0, "size not even"

    # downsample
    x = Reshape((h,w,c))(x)    # [b,h,w,c]
    x0 = x[:,0::2,0::2,:]
    x1 = x[:,1::2,0::2,:]
    x2 = x[:,0::2,1::2,:]
    x3 = x[:,1::2,1::2,:]
    x = K.concatenate([x0,x1,x2,x3], axis=-1)  # [b,h/2,w/2,4c]
    x = Reshape((hw//4,4*c))(x)
    x = Dense(2*c, use_bias=False)(x)

    return x


def PatchExpand(x, feature_size):
    # input: (b,h*w,c), output: (b,2h*2w,c//2)

    b, hw, c = K.int_shape(x)
    assert feature_size*feature_size==hw, "dim not match"
    h = w = feature_size

    x = Dense(c*2, use_bias=False)(x)
    x = Reshape((h,w,2,2,c//2))(x)    # (b,h,w,2,2,c//2)
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = Reshape((h*2*w*2,c//2))(x)   # (b,2h*2w,c//2)

    return x


def up_x4(x, feature_size):
    # input: (b,h*w,c), output: (b,4h,4w,c)

    b, hw, c = K.int_shape(x)
    assert feature_size*feature_size==hw, "dim not match"
    h = w = feature_size

    x = Dense(c*16, use_bias=False)(x)
    x = Reshape((h,w,4,4,c))(x)    # (b,h,w,4,4,c)
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = Reshape((h*4,w*4,c))(x)   # (b,4h*4w,c)

    return x


if __name__ == '__main__':

    model = SwinUnet(input_size=224, patch_size=4, window_size=7)
    model.summary()






