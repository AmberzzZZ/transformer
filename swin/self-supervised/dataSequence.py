from keras.utils import Sequence
import os
import math
import cv2
import random
import numpy as np
import pandas as pd


class dataSequence(Sequence):

    def __init__(self, data_dir, input_shape, batch_size, mask_ratio=0.6, mask_patch_size=32,
                 rand_layers=2, rand_magnitude=7, pkl_dir=None):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.batch_size = batch_size

        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

        self.rand_layers = rand_layers
        self.rand_magnitude = rand_magnitude

        self.train = pd.read_pickle(pkl_dir)
        print('train', len(self.train))
        self.indices = np.arange(len(self.train))

    def __len__(self):
        return math.floor(len(self.train)/float(self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_lst = [self.train[i] for i in batch_indices]
        x_batch = self.data_generator(batch_lst)
        mask_batch = self.mask_generator()

        return [x_batch,mask_batch], np.zeros(self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def mask_generator(self):
        mask_shape = (self.input_shape[0]//self.mask_patch_size, self.input_shape[1]//self.mask_patch_size)
        token_cnt = mask_shape[0] * mask_shape[1]
        mask_cnt = int(token_cnt * self.mask_ratio)
        mask_idx = np.random.permutation(token_cnt)[:mask_cnt]
        mask = np.zeros(token_cnt)
        mask[mask_idx] = 1

        # repeat
        mask = np.repeat(np.expand_dims(mask,axis=0), self.batch_size, axis=0)  # [b,hs,ws,1]
        # factor = self.mask_patch_size // 4
        # mask = mask.repeat(factor, axis=1).repeat(factor, axis=2)   # [b,hp,wp,1]
        mask = np.reshape(mask, (self.batch_size,mask_shape[0],mask_shape[1],1))

        return mask

    def data_generator(self, batch_lst):
        x_batch = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3))

        for file_name in self.train:
            img = cv2.imread(os.path.join(self.data_dir, file_name), 1)   # [b,h,w,c]
            # a light dataaug:
            # RandomResizedCrop, scale=(0.67, 1.), ratio=(3./4., 4./3.)
            img = RandomResizedCrop(img, self.input_shape, scale=(0.67, 1.), ratio=(3./4., 4./3.))
            # RandomHorizontalFlip
            img = RandomHorizontalFlip(img, p=0.5)
            # Normalize by ImageNet mean&std
            if np.max(img)>1:
                img = img / 255.

        return x_batch


def RandomResizedCrop(img, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=cv2.INTER_LINEAR):
    # Crop a random portion of image and resize it to a given size
    h, w = img.shape[:2]
    area = h * w
    # generate a [top,left,height,width]
    target_area = area * random.uniform(scale[0], scale[1])
    log_ratio = (math.log(ratio[0],10), math.log(ratio[1],10))
    aspect_ratio = math.pow(10, random.uniform(log_ratio[0], log_ratio[1]))
    new_w = int(round(math.sqrt(target_area * aspect_ratio)))
    new_h = int(round(math.sqrt(target_area / aspect_ratio)))
    if 0 < new_w <= w and 0 < new_h <= h:
        # random crop
        i = random.randint(0, h-new_h+1)
        j = random.randint(0, w-new_w+1)
    else:
        # center crop
        in_ratio = float(w) / float(h)
        if in_ratio<min(ratio):
            new_w = w
            new_h = int(round(new_w / min(ratio)))
        else:
            new_h = h
            new_w = int(round(new_h * max(ratio)))
            i = (h - new_h) // 2
            j = (w - new_w) // 2
    # crop & resize
    img = img[i:i+new_h,j:j+new_w]
    img = cv2.resize(img, size, interpolation=interpolation)
    return img


def RandomHorizontalFlip(img, p=0.5):
    if random.uniform(0, 1)>p:
        img = np.flip(img, axis=1)
    return img


if __name__ == '__main__':

    img = cv2.imread("/Users/amber/Downloads/cat.jpeg", 1)
    for i in range(10):
        new_img = RandomResizedCrop(img, (224,224))
        cv2.imshow('tmp', new_img)
        cv2.waitKey(0)








