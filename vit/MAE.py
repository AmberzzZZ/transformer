# masked autoencoder with VisionTransformer backbone
from keras.layers import Input, Conv2D, Reshape, Lambda, Concatenate, Layer, add, Dropout, Dense
from keras.models import Model
from keras.initializers import RandomNormal
from LayerNormalization import LayerNormalization
from MSA import MultiHeadAttention, FeedForwardNetwork
import tensorflow as tf
import keras.backend as K
import numpy as np
import math


def MAE_VIT(input_shape=(224,224,3), patch_size=16, mask_ratio=.75, use_cls_token=True,
            emb_dim=1024, depth=24, n_heads=16, mlp_ratio=4,
            decoder_emb_dim=512, decoder_depth=8, decoder_n_heads=16,
            norm_pix_loss=False,):

    inpt = Input(input_shape)    # full image, [b,h,w,c]

    # into patch embeddings
    h, w = input_shape[0]//patch_size, input_shape[1]//patch_size
    N = h*w
    x = Conv2D(emb_dim, patch_size, strides=patch_size, padding='same')(inpt)
    x = Reshape((N, emb_dim))(x)     # [b,N,emb_dim]

    # encoder pe: constant, sincos embedding, with cls token
    encoder_pe = Lambda(lambda x: PositionalEmbeddingSine(emb_dim, (h,w), cls_token=False),
                        name='EncoderPESine')(x)       # [1,N/N+1,emb_dim]
    x = Lambda(lambda x: x[0]+x[1])([x,encoder_pe])    # [b,N,emb_dim]

    # random masking
    x, mask, pos_idx, sort_idx = RandomMask(mask_ratio=mask_ratio)(x)   # [b,N1,emb_dim]

    # cat trainable cls_token
    if use_cls_token:
        cls_token = AddToken(shape=(1,1,emb_dim), init_std=.02)(x)
        x = Concatenate(axis=1)([x,cls_token])   # [b,N1+1,emb_dim]

    # encoder
    for i in range(depth):
        x = ViTAttentionBlock(emb_dim, n_heads=n_heads)(x)
    x = LayerNormalization()(x)

    # proj: narrow the dim
    x = Dense(decoder_emb_dim)(x)

    # mask tokens
    mask_token = AddToken(shape=(1,1,decoder_emb_dim), init_std=0.02, tile_N=N+int(use_cls_token))(x)  # [b,N2,emb_dim]
    # merged_token = MergeTokens(x, mask_token, pos_idx, use_cls_token)    # [b,N/N+1,emb_dim]
    x = Lambda(MergeTokens, arguments={'use_cls_token':use_cls_token}, name='MergeTokens'
               )([x,mask_token,pos_idx])   # [b,N/N+1,emb_dim]

    # decoder pe: constant, sincos embedding, with cls token
    decoder_pe = Lambda(lambda x: PositionalEmbeddingSine(decoder_emb_dim, (h,w),
                        cls_token=use_cls_token), name='DecoderPESine')(x)  # [1,N/N+1,emb_dim]
    x = Lambda(lambda x: x[0]+x[1])([x,decoder_pe])    # [b,N,emb_dim]

    # decoder
    for i in range(decoder_depth):
        x = ViTAttentionBlock(decoder_emb_dim, n_heads=decoder_n_heads)(x)   # [b,N/N+1,emb_dom]
    x = LayerNormalization()(x)

    # head
    in_channel = input_shape[-1]
    x = Dense((patch_size*patch_size*in_channel))(x)
    if use_cls_token:
        x = Lambda(lambda x: x[:,1:,:], name='RmClsToken')(x)   # [b,N,p*p*3]

    # loss
    loss = Lambda(ReconLoss, arguments={'patch_size': patch_size, 'norm_pix_loss': norm_pix_loss},
                  name='ReconLoss')([inpt, x, mask])

    model = Model(inpt, loss)

    return model


def ReconLoss(args, patch_size=16, norm_pix_loss=False):
    target, pred, mask = args
    # target: [b,h,w,c]
    # pred: [b,N,p*p*c], N=hw//pp
    # mask: [b,N]
    b, h, w, c = tf.shape(target)[0], int(target.shape[1]), int(target.shape[2]), int(target.shape[3])
    # patchify
    target = tf.reshape(target, (b, h//patch_size, patch_size, w//patch_size, patch_size, c))
    target = tf.transpose(target, (0,1,3,2,4,5))
    target = tf.reshape(target, (b, h//patch_size*w//patch_size, patch_size*patch_size*c))
    if norm_pix_loss:
        # norm per patch
        mean = K.mean(target, axis=-1, keep_dims=True)
        var = K.var(target, axis=-1, keep_dims=True)
        target = (target - mean) / var
    # per-pixel(patch) MSE
    loss = K.mean((target-pred)**2, axis=-1)   # [b,N]
    # valid mask
    return K.sum(loss*mask, axis=[1]) / K.sum(mask, axis=[1])   # [b,]


def MergeTokens(args, use_cls_token=True):
    x, mask_token, pos_idx = args
    if use_cls_token:
        cls_token = x[:,:1,:]
        patch_token = x[:,1:,:]
    else:
        patch_token = x
    x = tf.concat([patch_token,mask_token], axis=1)   # [b,N,emb_dim]
    # unshuffle
    x = tf.batch_gather(x, indices=K.cast(pos_idx, tf.int32))
    if use_cls_token:
        x = tf.concat([cls_token,x],axis=1)
    return x


class RandomMask(Model):
    # uniform distribution patch sampling
    def __init__(self, mask_ratio=.75):
        super(RandomMask, self).__init__(name='RandomMask')
        self.mask_ratio = mask_ratio

    def call(self, x):

        b, N = tf.shape(x)[0], int(x.shape[1])   # tensor, int, int
        self.N_keep = int(N*(1-self.mask_ratio))

        noise = tf.random.uniform(shape=(b, N))
        sorted_idx = tf.argsort(noise, axis=1)    # sorted patch indices, [b,N]
        pos_idx = tf.argsort(sorted_idx, axis=1)     # 0-N patch position in the sorted sequence

        keep_idx = sorted_idx[:,:self.N_keep]   # [b,N1]
        x_keep = tf.batch_gather(x, indices=keep_idx)  # [b,N1,D]

        mask = tf.where(tf.range(N)<self.N_keep, tf.zeros((N,)), tf.ones((N,)))  # [0,0,1,1,1,...]
        mask = tf.tile(tf.expand_dims(mask, axis=0), [b,1])   # [b,N]
        mask = tf.batch_gather(mask, indices=pos_idx)    # activated patches

        return [x_keep, mask, pos_idx, sorted_idx]

    def compute_output_shape(self, input_shape):
        b,N,D = input_shape
        return [(b,self.N_keep,D), (b,N), (b,N), (b,N)]


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

    model = MAE_VIT()
    model.summary()

    import numpy as np
    x = np.random.uniform(0,1,(16,224,224,3))
    y = model.predict(x)
    print(y.shape)





