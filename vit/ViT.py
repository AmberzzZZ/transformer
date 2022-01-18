# masked autoencoder with VisionTransformer backbone
from keras.layers import Input, Conv2D, Reshape, Lambda, Concatenate, Layer, Dropout, Dense
from keras.models import Model
from keras.initializers import RandomNormal
from LayerNormalization import LayerNormalization
from MSA import MultiHeadAttention, FeedForwardNetwork
import tensorflow as tf
import keras.backend as K
import numpy as np
import math


def VIT(input_shape=(224,224,3), n_classes=1000, patch_size=16, use_cls_token=True,
        emb_dim=768, depth=12, n_heads=16, mlp_ratio=4, att_drop_rate=0., drop_rate=0.1):

    inpt = Input(input_shape)  # [b,h,w,c]

    # into patch embeddings
    h, w = input_shape[0]//patch_size, input_shape[1]//patch_size
    N = h*w
    x = Conv2D(emb_dim, patch_size, strides=patch_size, padding='same', name='proj')(inpt)
    x = Reshape((N, emb_dim))(x)     # [b,N,emb_dim]

    # encoder pe: constant, sincos embedding, with cls token
    encoder_pe = Lambda(lambda x: PositionalEmbeddingSine(emb_dim, (h,w), cls_token=False),
                        name='EncoderPESine')(x)       # [1,N,emb_dim]
    x = Lambda(lambda x: x[0]+x[1])([x,encoder_pe])    # [b,N,emb_dim]

    # cat trainable cls_token
    if use_cls_token:
        cls_token = AddToken(shape=(1,1,emb_dim), init_std=.02)(x)
        x = Concatenate(axis=1)([x,cls_token])   # [b,N+1,emb_dim]

    # encoder
    for i in range(depth):
        x = ViTAttentionBlock(emb_dim, n_heads=n_heads)(x)
    x = LayerNormalization()(x)      # [b,N+1,emb_dim]

    # head: MLP on cls_token
    x = Lambda(lambda x: x[:,0,:], name='cls_token')(x)   # [b,D]
    x = Dense(n_classes, activation='tanh')(x)

    model = Model(inpt, x)

    return model


class ViTEncoderBlock(Model):
    def __init__(self,emb_dim=1024, n_heads=16, mlp_ratio=4, dbr=0., attn_drop=0., ffn_drop=0.):
        super(ViTEncoderBlock, self).__init__()
        self.ln1 = LayerNormalization()
        self.msa = MultiHeadAttention(emb_dim, num_heads=n_heads, attn_drop=attn_drop, ffn_drop=ffn_drop)
        self.res_drop1 = Dropout(dbr, noise_shape=(None,1,1))     # drop block

        self.ln2 = LayerNormalization()
        self.ffn = FeedForwardNetwork(emb_dim, mlp_ratio, drop_rate=ffn_drop)
        self.res_drop2 = Dropout(dbr, noise_shape=(None,1,1))     # drop block

    def call(self, x):
        # MSA
        inpt = x
        x = self.ln1(x)
        x = self.msa([x,x,x])   # self-attention
        x = inpt + self.res_drop1(x)

        # FFN
        inpt = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = inpt + self.res_drop2(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class ViTAttentionBlock(Model):
    def __init__(self,emb_dim=1024, n_heads=16, mlp_ratio=4, dbr=0., attn_drop=0., ffn_drop=0.):
        super(ViTAttentionBlock, self).__init__()
        self.ln1 = LayerNormalization()
        self.msa = MultiHeadAttention(emb_dim, num_heads=n_heads, attn_drop=attn_drop, ffn_drop=ffn_drop)
        self.res_drop1 = Dropout(dbr, noise_shape=(None,1,1))     # drop block

        self.ln2 = LayerNormalization()
        self.ffn = FeedForwardNetwork(emb_dim, mlp_ratio, drop_rate=ffn_drop)
        self.res_drop2 = Dropout(dbr, noise_shape=(None,1,1))     # drop block

    def call(self, x):
        # MSA
        inpt = x
        x = self.ln1(x)
        x = self.msa([x,x,x])   # self-attention
        x = inpt + self.res_drop1(x)

        # FFN
        inpt = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = inpt + self.res_drop2(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def PositionalEmbeddingSine(emb_dim, feature_shape, temp=1000, normalize=True, eps=1e-6, cls_token=False):
    # feature_shape: (h,w)
    # returns: [1,h,w,emd_dim] constant embedding, without weights, not trainable
    assert emb_dim%2==0, 'illegal embedding dim'
    h, w = feature_shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))   # [h,w]
    if normalize:
        grid_x = grid_x / (w+eps) * 2 * math.pi
        grid_y = grid_y / (h+eps) * 2 * math.pi
    single_dim = np.arange(emb_dim//2)            # [half_dim,]
    single_dim = temp ** (2*single_dim/emb_dim)   # enlarge the unlinear range [1,1000]

    pe_x = np.tile(np.expand_dims(grid_x, axis=2), [1,1,emb_dim//2]) / single_dim   # [h,w,half_dim]
    pe_y = np.tile(np.expand_dims(grid_y, axis=2), [1,1,emb_dim//2]) / single_dim

    pe_x = np.concatenate([np.sin(pe_x[:,:,::2]), np.cos(pe_x[:,:,1::2])], axis=2)   # [h,w,half_dim]
    pe_y = np.concatenate([np.sin(pe_y[:,:,::2]), np.cos(pe_y[:,:,1::2])], axis=2)

    PE = np.concatenate([pe_x,pe_y], axis=2)    # [h,w,emb_dim]
    PE = K.constant(np.reshape(PE, (1,h*w,emb_dim)))   # [1,hw,emb_dim]

    if cls_token:
        PE = tf.concat([tf.zeros((1,1,emb_dim)),PE],axis=1)   # [1,hw+1,emb_dim]

    return PE


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


if __name__ == '__main__':

    model = VIT()
    model.summary()


