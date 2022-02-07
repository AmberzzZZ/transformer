# masked autoencoder with VisionTransformer backbone
from keras.layers import Input, Conv2D, Reshape, Lambda, Concatenate, Layer, Dropout, Dense, add
from keras.models import Model
from keras.initializers import RandomNormal
from LayerNormalization import LayerNormalization
from MSA import MultiHeadAttention, FeedForwardNetwork
import tensorflow as tf
import numpy as np


def VIT(input_shape=(224,224,3), n_classes=1000, patch_size=16, use_cls_token=True,
        emb_dim=768, depth=12, n_heads=12, mlp_ratio=4, qkv_bias=False,
        att_drop_rate=0.1, drop_rate=0.1, drop_block=0., mlp_hidden_layers=0):

    inpt = Input(input_shape)  # [b,h,w,c]

    # into patch embeddings
    h, w = input_shape[0]//patch_size, input_shape[1]//patch_size
    N = h*w
    x = Conv2D(emb_dim, patch_size, strides=patch_size, padding='valid', name='proj')(inpt)
    x = Reshape((N, emb_dim))(x)     # [b,N,emb_dim]

    # cat trainable cls_token
    if use_cls_token:
        cls_token = AddToken(shape=(1,1,emb_dim), init_std=.02, name='cls_token')(x)   # [b,1,emb_dim]
        x = Concatenate(axis=1)([x,cls_token])   # [b,N+1,emb_dim]

    # encoder pe: trainable, std init
    encoder_pe = AddToken(shape=(1,N+int(use_cls_token), emb_dim), init_std=.02,
                          name='encoder_pe')(x)    # [b,N+1,emb_dim]
    x = add([x, encoder_pe])
    x = Dropout(drop_rate)(x)

    # encoder
    dbr = np.linspace(0, drop_block, depth)
    for i in range(depth):
        x = ViTEncoderBlock(emb_dim, n_heads, mlp_ratio, attn_drop=att_drop_rate, ffn_drop=drop_rate,
                            dbr=dbr[i], qkv_bias=qkv_bias)(x)
    x = LayerNormalization()(x)      # [b,N+1,emb_dim]

    # head: MLP on cls_token
    x = Lambda(lambda x: x[:,0,:], name='out_token')(x)   # [b,D] cls_token/mean_on_seq
    for i in range(mlp_hidden_layers):
        x = Dense(emb_dim, activation='tanh', name='pre_logits')(x)
    x = Dense(n_classes, activation='softmax', name='pred')(x)

    model = Model(inpt, x)

    return model


class ViTEncoderBlock(Model):
    def __init__(self, emb_dim=1024, n_heads=16, mlp_ratio=4, attn_drop=0., ffn_drop=0., dbr=0., qkv_bias=False):
        super(ViTEncoderBlock, self).__init__()
        self.ln1 = LayerNormalization()
        self.msa = MultiHeadAttention(emb_dim, num_heads=n_heads, attn_drop=attn_drop, ffn_drop=ffn_drop, qkv_bias=qkv_bias)

        self.ln2 = LayerNormalization()
        self.ffn = FeedForwardNetwork(emb_dim, mlp_ratio, drop_rate=ffn_drop)

        if dbr:
            self.res_drop1 = Dropout(dbr, noise_shape=(None,1,1))     # drop path
            self.res_drop2 = Dropout(dbr, noise_shape=(None,1,1))
        self.dbr = dbr

    def call(self, x):
        # MSA
        inpt = x
        x = self.ln1(x)
        x = self.msa([x,x,x])   # self-attention
        if self.dbr:
            x = self.res_drop1(x)
        x = inpt + x

        # FFN
        inpt = x
        x = self.ln2(x)
        x = self.ffn(x)
        if self.dbr:
            x = self.res_drop2(x)
        x = inpt + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class AddToken(Layer):
    def __init__(self, shape, tile_N=0, init_std=0.02, trainable=True, **kargs):
        super(AddToken, self).__init__(**kargs)
        self.tile_N = tile_N
        self.a = self.add_weight(shape, initializer=RandomNormal(stddev=init_std),
                                 trainable=trainable, name='add_variable')

    def call(self, x):
        # tile by batch-dim
        if self.tile_N:
            tile_N = self.tile_N - tf.shape(x)[1]
            self.a = tf.tile(self.a, [tf.shape(x)[0],tile_N,1])    # [b,N_masked,emb_dim]
        else:
            self.a = tf.tile(self.a, [tf.shape(x)[0],1,1])    # [b,1,emb_dim]
        return self.a

    def compute_output_shape(self, input_shape):
        b, N, D = input_shape
        if self.tile_N:
            return (b,self.tile_N-N,D)
        else:
            return (b,1,D)


def vit_small_patch16_224(n_classes):
    model = VIT((224,224,3), n_classes, patch_size=16, emb_dim=768, depth=8,
                n_heads=8, mlp_ratio=3, qkv_bias=False)
    return model


def vit_base_patch16_224(n_classes):   # vit_base_patch16_384, vit_base_patch32_384
    model = VIT((224,224,3), n_classes, patch_size=16, emb_dim=768, depth=12,
                n_heads=12, mlp_ratio=4, qkv_bias=True)
    return model


def vit_large_patch16_224(n_classes):   # vit_large_patch16_384, vit_large_patch32_384
    model = VIT((224,224,3), n_classes, patch_size=16, emb_dim=1024, depth=24,
                n_heads=16, mlp_ratio=4, qkv_bias=True)
    return model


def vit_huge_patch16_224(n_classes):   # vit_huge_patch32_384
    model = VIT((224,224,3), n_classes, patch_size=16, emb_dim=1280, depth=32,
                n_heads=16, mlp_ratio=4, qkv_bias=False)
    return model


if __name__ == '__main__':

    # vit_base = VIT(emb_dim=768, depth=12, n_heads=12, qkv_bias=True)
    vit_large = VIT(emb_dim=1024, depth=24, n_heads=16, qkv_bias=True)
    # vit_huge = VIT(emb_dim=1280, depth=32, n_heads=16)
    vit_large.summary()


