from keras.layers import Input, Conv2D, Reshape, Concatenate, add, Dropout, Dense, Lambda, \
                         BatchNormalization, ReLU, multiply
from MSA import MultiHeadAttention, FeedForwardNetwork, gelu
from LayerNormalization import LayerNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf


def LV_ViT(input_size=224, patch_size=16, emb_dim=384, mlp_dim=1152, out_dim=9,
           num_layers=16, num_heads=6, attn_drop=0., ffn_drop=0., residual_drop=0.1,
           residual_scale=2., mix_token=False, aux_loss=False):

    inpt = Input((input_size, input_size, 3))

    # patch embedding
    x = ConvBN(inpt, 64, kernel_size=7, strides=2)
    x = ConvBN(x, 64, kernel_size=3, strides=1)
    x = ConvBN(x, 64, kernel_size=3, strides=1)
    x = Conv2D(emb_dim, 8, strides=8, padding='same')(x)
    N = (input_size//patch_size)

    # mixtoken
    if mix_token:
        mix_mask = Input((N,N,1))   # random crop based on beta distribution
        x = Lambda(mix_tokens)([x, mix_mask])

    # cls token
    x = Reshape((N*N, emb_dim))(x)
    tmp = Lambda(lambda x: x[:,0:1,:])(x)   # [b,1,D]
    x0 = Lambda(lambda x: K.zeros_like(x))(tmp)
    x = Concatenate(axis=1)([x0,x])   # [b,N+1,D]

    # positional embeddings
    pe = Lambda(lambda x: tf.tile(positional_embedding(N*N+1, emb_dim), [tf.shape(x)[0], 1,1]))(x)
    x = add([x, pe])   # [b,N+1,D]
    x = Dropout(ffn_drop)(x)

    # transformer blocks
    for i in range(num_layers):
        x = encoder_block(x, emb_dim, num_heads, mlp_dim, attn_drop, ffn_drop,
                          residual_scale, residual_drop)
    x = LayerNormalization()(x)

    # take cls token
    x_cls = Lambda(lambda x: x[:,0,:])(x)   # [b,D]
    if out_dim:
        x_cls = Dense(out_dim, activation='softmax')(x_cls)

    # take aux tokens
    if aux_loss:
        x_aux = Lambda(lambda x: x[:,1:,:])(x)   # [b,N,D]
        if mix_token:
            x_aux = Reshape((N,N,emb_dim))(x_aux)
            x_aux = Lambda(mix_tokens)([x_aux, mix_mask])
            x_aux = Reshape((N*N,emb_dim))(x_aux)
        x_aux = Dense(out_dim, activation='softmax')(x_aux)

    model_inputs = [inpt, mix_mask] if mix_token else inpt
    model_outputs = [x_cls, x_aux] if aux_loss else x_cls
    model = Model(model_inputs, model_outputs)

    return model


def encoder_block(x, att_dim=512, num_heads=12, mlp_dim=2048, attn_drop=0., ffn_drop=0.1,
                  residual_scale=2., residual_drop=0.):
    # MSA
    inpt = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(att_dim, num_heads, attn_drop, ffn_drop)([x,x,x])    # self-attention
    x = Dropout(residual_drop, noise_shape=(None, 1, 1))(x)    # stochastic-depth by sample
    x = Lambda(lambda x: x[0]+x[1]/residual_scale)([inpt, x])

    # FFN
    inpt = x
    x = LayerNormalization()(x)
    x = FeedForwardNetwork(mlp_dim, att_dim, activation=gelu, drop_rate=ffn_drop)(x)
    x = Dropout(residual_drop, noise_shape=(None, 1, 1))(x)
    x = Lambda(lambda x: x[0]+x[1]/residual_scale)([inpt, x])

    return x


def positional_embedding(seq_len, model_dim):
    PE = np.zeros((seq_len, model_dim))
    for i in range(seq_len):
        for j in range(model_dim):
            if j % 2 == 0:
                PE[i, j] = np.sin(i / 10000 ** (j / model_dim))
            else:
                PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_dim))
    PE = K.constant(np.expand_dims(PE, axis=0))
    return PE


def mix_tokens(args):
    x, mix_mask = args
    return x * mix_mask + K.reverse(x, axes=0) * (1-mix_mask)


def ConvBN(x, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = LV_ViT(input_size=224, patch_size=16, emb_dim=384, mlp_dim=1152, out_dim=8,
                   num_layers=16, num_heads=6, attn_drop=0., ffn_drop=0., residual_drop=0.1,
                   residual_scale=2., mix_token=True, aux_loss=True)
    model.summary()
    print(model.inputs)
    print(model.outputs)




