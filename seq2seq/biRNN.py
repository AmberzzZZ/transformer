from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np


def biRNN(input_shape, n_classes):
    inpt = Input(input_shape)
    # biLSTM
    x = Bidirectional(LSTM(units=128))(inpt)
    # dense prediction
    x = Dense(n_classes, activation='softmax')(x)
    # model
    model = Model(inpt, x)
    return model


def load_mnist(data_path, n_samples, n_classes, n_step, n_input):
    import os, glob, cv2
    X = np.zeros((n_samples, n_step, n_input))
    Y = np.zeros((n_samples, n_classes))
    n_samples_percls = n_samples // n_classes
    idx = 0
    for folder in os.listdir(data_path):
        if len(folder)!=2:
            continue
        digit = int(folder[-1])
        cls_cnt = 0
        for file in glob.glob(os.path.join(data_path, folder) + '/*png'):
            img = cv2.imread(file, 0)
            X[idx] = img/255. if np.max(img)>1 else img
            Y[idx, digit] = 1
            cls_cnt += 1
            idx += 1
            if cls_cnt >= n_samples_percls:
                break
    # shuffle
    indices = np.arange(n_samples)
    X = X[indices]
    Y = Y[indices]
    return X, Y


if __name__ == '__main__':

    # data process
    data_path = '/Users/amber/dataset/mnist'
    n_step = 28
    n_input = 28
    n_classes = 10
    batch_size = 256
    n_samples = 6000

    X, Y = load_mnist(data_path, n_samples, n_classes, n_step, n_input)

    model = biRNN((n_step, n_input), n_classes)

    # # train
    # model.compile(Adam(0.01), loss='categorical_crossentropy')
    # ckpt = ModelCheckpoint('biLSTM_ep{epoch:02d}.h5', monitor='loss')
    # model.fit(X, Y,
    #           batch_size=batch_size,
    #           epochs=100,
    #           callbacks=[ckpt],
    #           validation_split=0.2)

    # inference
    model.load_weights("biLSTM_ep15.h5")
    test_x = X[-1:]
    pred = model.predict(test_x)[0]
    print(np.argmax(pred))
    import cv2
    cv2.imshow("tmp", np.uint8(test_x[0]*255))
    cv2.waitKey(0)





