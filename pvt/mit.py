from LayerNormalization import LayerNormalization
from EMSA import EfficientMSA, EfficientMLP
from keras.layers import Input, Conv2D, Reshape, Dropout, GlobalAveragePooling1D, Dense
from keras.models import Model
import numpy as np


def MixViT(input_shape=224, n_classes=1000, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
           embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
           qkv_bias=True, drop_attn=0., drop_mlp=0., drop_path_rate=0.1):

    if isinstance(input_shape, tuple):
        inpt = Input(input_shape)
        feature_shape = input_shape[:2]
    else:
        inpt = Input((input_shape,input_shape,3))
        feature_shape = (input_shape, input_shape)

    # transformer encoder
    x = inpt
    n_stages = len(depths)
    dbr = np.linspace(0, drop_path_rate, num=sum(depths))
    for i in range(n_stages):
        # overlap patch embedding: Conv, kernel=4, stride=2, same padding
        if i==0:
            x, feature_shape = OverlapPatchEmbedding(x, embed_dims[i], patch_size=7, stride=4, name_idx=i+1)   # downsmap x4
        else:
            # reshape to [b,h,w,c]
            x = Reshape(feature_shape+(embed_dims[i-1],))(x)
            x, feature_shape = OverlapPatchEmbedding(x, embed_dims[i], patch_size=3, stride=2, name_idx=i+1)   # downsmap x2
        # print('---feature shape', feature_shape)
        # efficient self-attention blocks
        for b in range(depths[i]):
            x = EfficientAttentionBlock(feature_shape, num_heads[i], embed_dims[i], mlp_ratios[i], sr_ratios[i],
                                        attn_drop=drop_attn, mlp_drop=drop_mlp, dbr=dbr[sum(depths[:i]):sum(depths[:i+1])][b],
                                        stage_idx=i+1, block_idx=b)(x)      # [b,hw,c]
        # norm
        x = LayerNormalization(1e-6, name='norm%d' % (i+1))(x)

    # cls head:
    x = GlobalAveragePooling1D()(x)   # [b,c]
    x = Dense(n_classes, activation='softmax')(x)   # [b,cls]

    return Model(inpt, x)


def OverlapPatchEmbedding(x, embed_dim, patch_size=7, stride=4, name_idx=None):
    # conv: weight: fan_out_normal init, bias: zeros init
    x = Conv2D(embed_dim, kernel_size=patch_size, strides=stride, padding='same', use_bias=True,
               name='patch_embed%d.conv' % name_idx)(x)
    feature_h, feature_w = int(x.shape[1]), int(x.shape[2])
    # reshape
    x = Reshape((feature_h*feature_w, embed_dim))(x)   # [b,hw,c]
    # layer norm: weight: ones init, bias: zeros init
    x = LayerNormalization(epsilon=1e-6, name='patch_embed%d.norm' % name_idx)(x)
    return x, (feature_h, feature_w)


class EfficientAttentionBlock(Model):

    def __init__(self, feature_shape, n_heads, emb_dim, mlp_ratio, reduction_ratio=1.,
                 attn_drop=0., mlp_drop=0., dbr=0., stage_idx=None, block_idx=None, **kwargs):
        super(EfficientAttentionBlock, self).__init__(name='EAttB%d.%d' % (stage_idx,block_idx), **kwargs)

        self.ln1 = LayerNormalization()
        self.emsa = EfficientMSA(feature_shape, emb_dim, n_heads, reduction_ratio, attn_drop, mlp_drop,
                                 qkv_bias=True, qkv_scale=None)
        self.emsa_drop = Dropout(dbr, noise_shape=(None, 1, 1))
        self.ln2 = LayerNormalization()
        self.emlp = EfficientMLP(feature_shape, emb_dim, mlp_ratio, mlp_drop)
        self.emlp_drop = Dropout(dbr, noise_shape=(None, 1, 1))

    def call(self, x, mask=None):
        # inpt: [b,hw,c]
        inpt = x

        # EMSA with residual
        x = self.ln1(x)
        x = self.emsa(x)
        x = inpt + self.emlp_drop(x)

        # EMLP with residual
        inpt = x
        x = self.ln2(x)
        x = self.emlp(x)
        x = inpt + self.emlp_drop(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def mit_b0(input_shape, n_classes=1000):
    return MixViT(input_shape, n_classes, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                  embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True, drop_attn=0., drop_mlp=0., drop_path_rate=0.1)


def mit_b1(input_shape, n_classes=1000):
    return MixViT(input_shape, n_classes, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                  embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True, drop_attn=0., drop_mlp=0., drop_path_rate=0.1)


def mit_b2(input_shape, n_classes=1000):
    return MixViT(input_shape, n_classes, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                  embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True, drop_attn=0., drop_mlp=0., drop_path_rate=0.1)


def mit_b3(input_shape, n_classes=1000):
    return MixViT(input_shape, n_classes, depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
                  embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True, drop_attn=0., drop_mlp=0., drop_path_rate=0.1)


def mit_b4(input_shape, n_classes=1000):
    return MixViT(input_shape, n_classes, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
                  embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True, drop_attn=0., drop_mlp=0., drop_path_rate=0.1)


def mit_b5(input_shape, n_classes=1000):
    return MixViT(input_shape, n_classes, depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
                  embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True, drop_attn=0., drop_mlp=0., drop_path_rate=0.1)


if __name__ == '__main__':

    # model = MixViT()
    # model.summary()

    model = mit_b1(input_shape=(512,768,3), n_classes=1000)
    model.summary()



