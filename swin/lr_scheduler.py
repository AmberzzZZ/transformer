from keras.callbacks import Callback
import keras.backend as K
import math


class CosineLRScheduler(Callback):

    # reference: https://github.com/rwightman/pytorch-image-models/issues/353

    def __init__(self, warmups=0, warmup_init=1e-8, lr_base=1e-3, lr_min=0., lr_decay=1.,
                 init_cycle=30, cycle_exp=1, verbose=1):
        super(CosineLRScheduler, self).__init__()

        self.warmups = warmups             # warmup steps
        self.warmup_init = warmup_init     # warmup_init_rate
        self.lr_base = lr_base             # lr_base_rate
        self.lr_min = lr_min               # lr_min_rate
        self.lr_decay = lr_decay           # lr_decay_rate
        self.init_cycle = init_cycle       # first cycle length
        self.cycle_exp = cycle_exp         # cycle expand ratio
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch<self.warmups:
            # linearly increase
            lr = self.warmup_init + (self.lr_base - self.warmup_init) / self.warmups * epoch
        else:
            current_step = epoch-self.warmups
            print('cur', current_step)

            # cycle of cosine decrease
            if self.cycle_exp==1:
                # constant epochs in a cycle
                cycle_idx = current_step // self.init_cycle
                i = current_step % self.init_cycle     # index inside a cycle
                ti = self.init_cycle                    # current cycle

            else:
                # increasing epochs in a cycle
                cycle_idx = math.floor(math.log(1-current_step/self.init_cycle*(1-self.cycle_exp), self.cycle_exp))
                i = current_step - (1-self.cycle_exp**cycle_idx) / (1-self.cycle_exp) * self.init_cycle
                ti = self.cycle_exp ** cycle_idx * self.init_cycle

            lr = self.lr_min + self.lr_decay ** cycle_idx * (self.lr_base - self.lr_min) * 0.5 * (1 + math.cos(math.pi * i/ti))
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: CosineLRScheduler setting learning rate to %s.' % (epoch+1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class LinearLRScheduler(Callback):

    def __init__(self, epochs, warmups=0, lr_base=1e-3, lr_end=0., warmup_init=1e-8, verbose=1):
        super(LinearLRScheduler, self).__init__()

        self.epochs = epochs        # total epochs
        self.warmups = warmups      # warmup steps
        self.lr_base = lr_base      # lr_base_rate
        self.lr_end = lr_end        # lr_end_rate
        self.warmup_init = warmup_init      # lr_init_rate
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        epoch += 1
        if epoch<self.warmups:
            # linearly increase
            lr = self.warmup_init + (self.lr_base - self.warmup_init) / self.warmups * epoch
        else:
            # linearly decrease
            current_step = epoch-self.warmups
            steps = float(self.epochs - self.warmups - 1)
            lr = self.lr_base - (self.lr_base - self.lr_end)*(current_step/steps)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: LinearLRScheduler setting learning rate to %s.' % (epoch, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


if __name__ == '__main__':

    linear_lr = LinearLRScheduler(epochs=10, warmups=5)
    cosine_lr = CosineLRScheduler(warmups=5, lr_base=1e-3, lr_min=0., lr_decay=0.5, init_cycle=30, cycle_exp=1.5)


    # import matplotlib.pyplot as plt
    # lrs = []
    # x = []
    # for i in range(300):
    #     lrs.append(cosine_lr.on_epoch_begin(i))
    #     x.append(i)

    # plt.plot(x,lrs)
    # plt.show()

