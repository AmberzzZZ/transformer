# efficient multi-head self-attention & efficient MLP
from keras.layers import Dense, Activation, DepthwiseConv2D, Dropout, Reshape, Conv2D
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import math
from LayerNormalization import LayerNormalization


def gelu(x, approx=False):
    if approx:
        return 0.5 * x * (1 + K.tanh(K.sqrt(K.constant(2./math.pi)) * (x + 0.044715 * K.pow(x, 3))))
    else:
        return 0.5 * x * (1. + tf.math.erf(x / K.sqrt(K.constant(2.))))


class EfficientMSA(Model):

    def __init__(self, feature_shape, emb_dim, num_heads, sr_ratio, attn_drop=0., proj_drop=0.,
                 qkv_bias=True, qkv_scale=None, **kwargs):
        super(EfficientMSA, self).__init__(**kwargs, )

        self.feature_shape = feature_shape
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self.dense_q = Dense(emb_dim, use_bias=qkv_bias)
        self.dense_kv = Dense(emb_dim*2, use_bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(emb_dim)
        self.proj_drop = Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2D(emb_dim, kernel_size=sr_ratio, strides=sr_ratio)   # [b,hw,c]->[b,hw/rr,c]
            self.norm = LayerNormalization(1e-6)

    def call(self, x, mask=None):
        # input: [b,hw,c]
        inpt_h, inpt_w = self.feature_shape
        # split heads
        q = tf.transpose(Reshape((inpt_h*inpt_w, self.num_heads,self.emb_dim//self.num_heads))(self.dense_q(x)), (0,2,1,3))   # [b,nH,hw,c/nH]
        if self.sr_ratio>1:
            # narrow the hw-dim
            inpt_dim = int(x.shape[-1])
            x_ = Reshape((inpt_h,inpt_w,inpt_dim))(x)
            x_ = self.sr(x_)    # [b,hw/rr,c]
            x_ = self.norm(x_)
            narrow_h, narrow_w = int(x_.shape[1]), int(x_.shape[2])
            # -1: inpt_h
            kv = tf.transpose(Reshape((narrow_h*narrow_w, 2, self.num_heads,self.emb_dim//self.num_heads))(self.dense_kv(x_)), (2,0,3,1,4))   # [2,b,nH,hw,c/nH]
        else:
            kv = tf.transpose(Reshape((inpt_h*inpt_w, 2, self.num_heads,self.emb_dim//self.num_heads))(self.dense_kv(x)), (2,0,3,1,4))   # [2,b,nH,hw,c/nH]
        k, v = kv[0], kv[1]

        # qk similarity, q [b,nH,Lq,C], k [b,nH,Lk,C]
        matmul_qk = tf.matmul(q, k, transpose_b=True)   # [b,nH,Lq,Lk]
        score = matmul_qk / tf.math.sqrt(tf.cast(int(q.shape[-1]), tf.float32))   # or qkv_scale
        score = tf.nn.softmax(score)
        attn = self.attn_drop(score)

        # weighted sum: v [b,nH,Lk,C]
        attn = tf.matmul(attn, v)    # [b,nH,Lq,C]
        attn = Reshape((inpt_h*inpt_w,self.emb_dim))(tf.transpose(attn, (0,2,1,3)))    # [b,Lq,nH*C]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class EfficientMLP(Model):

    def __init__(self, feature_shape, emb_dim, mlp_ratio=4, drop_rate=0., **kwargs):
        super(EfficientMLP, self).__init__(**kwargs)

        self.hid_dim = emb_dim*mlp_ratio
        self.feature_shape = feature_shape
        self.emb_dim = emb_dim

        self.fc1 = Dense(self.hid_dim)
        self.fc2 = Dense(emb_dim)
        self.dwconv = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=True)
        self.drop1 = Dropout(drop_rate)
        self.drop2 = Dropout(drop_rate)
        self.act = Activation(gelu)

    def call(self, x, mask=None):
        x = self.fc1(x)   # [b,hw,hid_dim]
        # reshape-conv-reshape
        x = Reshape(self.feature_shape+(self.hid_dim,))(x)
        x = self.dwconv(x)
        x = Reshape((self.feature_shape[0]*self.feature_shape[1], self.hid_dim))(x)
        # end of dwconv: [b,hw,c]
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.emb_dim,)


if __name__ == '__main__':

    from keras.layers import Input

    x = Input((196,256))
    y = EfficientMSA((14,14), emb_dim=256, num_heads=8, sr_ratio=2, attn_drop=0., proj_drop=0., qkv_bias=True, qkv_scale=None)(x)
    print(y)
    y = EfficientMLP((14,14), emb_dim=256, mlp_ratio=4, drop_rate=0.)(x)
    print(y)


