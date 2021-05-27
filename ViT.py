from keras.layers import Input, Conv2D, Reshape, Concatenate, add, Dropout, Dense, Lambda
from MSA import MultiHeadAttention, FeedForwardNetwork, gelu
from transformer import positional_embedding
from LayerNormalization import LayerNormalization
from keras.models import Model
import keras.backend as K
import tensorflow as tf


def visionTransformer(input_size=224, patch_size=16, drop_rate=0.1, num_layers=12,
                      hidden_dim=768, att_drop_rate=0., num_heads=12, mlp_dim=3072,
                      out_dim=None):

    inpt = Input((input_size, input_size, 3))

    # linear project patches
    x = Conv2D(hidden_dim, patch_size, strides=patch_size, padding='valid')(inpt)   # [b,Np,Np,D]

    # reshape
    b, h, w, c = K.int_shape(x)
    x = Reshape((h*w, c))(x)   # [b,N,D]

    # prepend class token
    x0 = Lambda(lambda x: K.placeholder((None, 1, hidden_dim)))(x)
    x = Concatenate(axis=1)([x0,x])   # [b,N+1,D]

    b, sq_len, input_dim = K.int_shape(x)
    # add fixed/learnable positional embeddings
    pe = Lambda(lambda x: positional_embedding(sq_len, input_dim))(x)
    x = add([x, pe])   # [b,N+1,D]
    x = Dropout(drop_rate)(x)

    # transformer encoder
    for i in range(num_layers):
        x = encoder_block(x, hidden_dim, att_drop_rate, num_heads, mlp_dim, drop_rate)
    x = LayerNormalization()(x)

    # take cls token
    x = Lambda(lambda x: x[:,0,:])(x)   # [b,D]
    if out_dim:
        x = Dense(out_dim, activation='tanh')(x)

    model = Model(inpt, x)

    return model


def encoder_block(x, hidden_dim=768, att_drop_rate=0., num_heads=12, mlp_dim=3072, drop_rate=0.1):
    # MSA
    inpt = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(hidden_dim, num_heads)([x,x,x])   # self-attention
    x = Dropout(drop_rate)(x)
    x = add([inpt, x])
    # layer norm

    # FFN
    inpt = x
    out_dim = K.int_shape(x)[-1]
    x = LayerNormalization()(x)
    x = FeedForwardNetwork(mlp_dim, out_dim, activation=gelu, drop_rate=drop_rate)(x)
    x = add([inpt, x])

    return x


if __name__ == '__main__':

    model = visionTransformer(patch_size=16, drop_rate=0.1, num_layers=12, hidden_dim=768, att_drop_rate=0., num_heads=12, mlp_dim=3072, out_dim=None)
    model.summary()














