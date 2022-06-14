import os
import cv2
import numpy as np
import pandas as pd
from simmim import SimMIM
from keras.models import Model


if __name__ == '__main__':

    input_shape = (192,192)
    encoder_stride = 32
    mask_ratio = .6

    model = SimMIM(input_shape=(192,192,3), window_size=6, drop_path_rate=0.)
    model.load_weights("weights/mim_ep60_loss0.028.h5")
    model = Model(model.inputs, model.get_layer(index=-2).output)

    mask_shape = (input_shape[0]//encoder_stride, input_shape[1]//encoder_stride)
    token_cnt = int(mask_shape[0] * mask_shape[1])
    mask_cnt = int(token_cnt * mask_ratio)

    test_lst = ["/Users/amber/Downloads/cat.jpeg"]
    for file in test_lst:
        img = cv2.imread(file, 1)
        img = cv2.resize(img, input_shape)
        if np.max(img)>1:
            img = img / 255.
        inpt = np.expand_dims(img, axis=0)
        mask_idx = np.random.permutation(token_cnt)[:mask_cnt]
        mask = np.zeros(token_cnt)
        mask[mask_idx] = 1
        mask = np.reshape(mask, (1, mask_shape[0], mask_shape[1], 1))   # x32

        pred = model.predict([inpt, mask])[0]

        # clip
        pred[pred<0] = 0.
        pred[pred>1.] = 1.

        # vis
        cv2.imshow('pred', pred)
        cv2.imshow('inpt', img)
        mask = mask[0,:,:,0].repeat(encoder_stride,axis=0).repeat(encoder_stride,axis=1)   # [h,w]
        img[mask>0] = [1,1,1]
        cv2.imshow('masked_inpt', img)
        cv2.waitKey(0)









