from swin import SwinTransformer
from AdamW import AdamW
from lr_scheduler import CosineLRScheduler
from keras.metrics import top_k_categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np


def top3_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top5_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


if __name__ == '__main__':

    input_shape = (224,244)
    n_classes = 6
    batch_size = 48    # for swinT-224-window7-patch4

    # data
    X = np.zeros((48,224,224,3))
    Y = np.zeros((48,6))

    # model
    model = SwinTransformer(input_shape=(224,224,3), n_classes=n_classes, patch_size=4, emb_dim=96,
                            ape=False, num_layers=[2,2,6,2], num_heads=[3,6,12,24], window_size=7,
                            qkv_bias=True, qk_scale=None, mlp_ratio=4, attn_drop=0., ffn_drop=0.,
                            residual_drop=0.2)
    model.load_weights('weights/swin_tiny_patch4_window7_224.h5', by_name=True, skip_mismatch=True)
    opt = AdamW(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                weight_decay=5e-2, ema_momentum=0, clip_norm=1)
    model.compile(opt, loss=categorical_crossentropy, metrics=['acc', top3_acc, top5_acc])

    # lr schedule
    cos_lr = CosineLRScheduler(warmups=5, warmup_init=4e-5, lr_base=1e-3, lr_min=4e-5, lr_decay=0.9, 
                               init_cycle=30, cycle_exp=1, verbose=1)

    # checkpoint
    filepath = 'weights/swinT_input%d_cls%d_epoch_{epoch:02d}_loss_{loss:.3f}.h5' % (input_shape[0], n_classes)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_weights_only=True, period=1)

    # train
    model.fit(X, Y, batch_size=16, callbacks=[cos_lr, checkpoint])









