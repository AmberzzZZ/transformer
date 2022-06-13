# simmim: swin-back + 1-layer-linear
from swin import SwinTransformerBlock, PatchMerging
from LayerNormalization import LayerNormalization
from keras.layers import Input, Conv2D, Reshape, Layer, Lambda, Dropout
from keras.initializers import TruncatedNormal
from keras.models import Model
import tensorflow as tf
import numpy as np


def SimMIM(input_shape=(192,192,3), patch_size=4, emb_dim=128, encoder_stride=32,
           num_layers=[2,2,18,2], num_heads=[4,8,16,32], window_size=6,
           drop_rate=0., att_drop_rate=0., drop_path_rate=0.):

    # inputs: img [b,h,w,3] & mask [b,h/s,w/s]
    inpt = Input(input_shape)
    mask = Input((input_shape[0]//patch_size, input_shape[1]//patch_size))

    # into embeddings
    x = Conv2D(emb_dim, patch_size, strides=patch_size, padding='same')(inpt)  # [b,h/s,w/s,128]
    feat_h, feat_w = int(x.shape[1]), int(x.shape[2])
    x = Reshape((feat_h*feat_w, emb_dim))(x)
    flatten_mask = Reshape((feat_h*feat_w, 1))(mask)
    mask_tokens = MaskToken(emb_dim)(x)    # [b,hw/ss,128]

    # masking: x = (1-mask) * x + mask * mask_tokens
    x = Lambda(lambda x: (1-x[2])*x[0]+x[2]*x[1])([x,mask_tokens,flatten_mask])

    # pos drop
    x = Dropout(drop_rate)(x)

    # swin blocks
    n_stages = len(num_layers)
    dbr = np.linspace(0, drop_path_rate, num=sum(num_layers))    # drop block rate
    for i in range(n_stages):
        # alternative swin blocks: WMSA-SWMSA-WMSA...
        if i==0:
            x = Reshape((feat_h,feat_w,emb_dim))(x)   # [bhwc] for window split
        depth = num_layers[i]
        n_heads = num_heads[i]
        for d in range(depth//2):
            x = SwinTransformerBlock(emb_dim, (feat_h,feat_w), n_heads, window_size, attn_drop=att_drop_rate,
                                     ffn_drop=drop_rate, residual_drop=dbr[sum(num_layers[:i]):sum(num_layers[:i+1])][d:d+2],
                                     idx=d+sum(num_layers[:i])//2)(x)
        x = Reshape((feat_h,feat_w,emb_dim))(x)
        if i!=n_stages-1:
            # downsamp: PatchMerging
            x = PatchMerging((feat_h, feat_w), emb_dim, stage_idx=i)(x)
            feat_h, feat_w = feat_h//2, feat_w//2
            emb_dim *= 2

    # final norm
    x = LayerNormalization()(x)   # [b,h/32,w/32, 128*8]

    # linear head
    x = Conv2D(encoder_stride*encoder_stride*3, kernel_size=1, strides=1)(x)   # [b,h/32,w/32,32*32*3]
    x = Lambda(PixelShuffle, arguments={'encoder_stride': encoder_stride})(x)   # [b,h,w,3]

    model = Model([inpt,mask], x)

    return model


def PixelShuffle(x, encoder_stride=32):
    b = tf.shape(x)[0]
    feat_h, feat_w, c = map(int, x.shape[1:])
    x = tf.reshape(x, (b,feat_h, feat_w,encoder_stride,encoder_stride,3))
    x = tf.transpose(x, (0,1,3,2,4,5))
    x = tf.reshape(x, (b,feat_h*encoder_stride,feat_w*encoder_stride,3))
    return x


class MaskToken(Layer):

    # generate mask tokens

    def __init__(self, emb_dim=128, **kwargs):
        super(MaskToken, self).__init__(**kwargs)
        self.emb_dim = emb_dim

    def build(self, input_shape):
        # trainable varaible, normal init(0.,.02)
        self.mask_token = self.add_weight(shape=(1,1,self.emb_dim),
                                          initializer=TruncatedNormal(mean=0., stddev=0.02),
                                          name='mask_token')
        super(MaskToken, self).build(input_shape)

    def call(self, x):
        # broadcast
        b = tf.shape(x)[0]
        length = int(x.shape[1])
        mask_tokens = tf.tile(self.mask_token, [b,length,1])
        return mask_tokens

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':

    model = SimMIM()
    model.summary()




