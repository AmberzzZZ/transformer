from keras.optimizers import Optimizer
import keras.backend as K
import tensorflow as tf


class AdamW(Optimizer):
    # Adam optimizer with weight decay & ema

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,          # lr decay, 0.01 if try to use
                 amsgrad=False,
                 weight_decay=4e-5,
                 clip_norm=5.,
                 ema_momentum=0,   # 0.999 if try to use
                 **kwargs):
        super(AdamW, self).__init__(**kwargs)
        # variables
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1)
            self.beta_2 = K.variable(beta_2)
            self.decay = K.variable(decay)
        # costants
        # self.beta_1 = beta_1
        # self.beta_2 = beta_2
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.ema_momentum = ema_momentum

    def _create_all_weights(self, params):
        ms = [K.zeros(K.int_shape(p), name=p.name.strip(':0')+'/ms') for p in params]
        vs = [K.zeros(K.int_shape(p), name=p.name.strip(':0')+'/vs') for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), name=p.name.strip(':0')+'/vhats') for p in params]
        else:
            vhats = [K.zeros(1, name=p.name.strip(':0')+'/vhats') for p in params]
        ema_weights = [tf.Variable(p, name=p.name.strip(':0')) for p in params]
        self.weights = [self.iterations] + ms + vs + vhats + ema_weights
        return ms, vs, vhats, ema_weights

    def get_updates(self, loss, params):
        # gradient clip
        grads = self.get_gradients(loss, params)
        grads = [tf.clip_by_norm(g, self.clip_norm) for g in grads]

        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if K.get_value(self.decay) > 0:
            lr = lr * (1. / (1. + self.decay * tf.cast(self.iterations, K.dtype(self.decay))))

        # EMA bias correction
        t = tf.cast(self.iterations+1, tf.float32)
        lr_t = lr * (K.sqrt(1. - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t)))
        # self.updates.append(K.update(self.lr, lr_t))

        ms, vs, vhats, ema_weights = self._create_all_weights(params)
        for p, g, m, v, vhat, e in zip(params, grads, ms, vs, vhats, ema_weights):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)
            # gradient descent
            if self.amsgrad:
                vhat_t = tf.maximum(vhat, v_t)
                new_p = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                new_p = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            # weight decay excludes normalization
            if self.weight_decay and 'normalization' not in p.name:
                new_p = new_p - p*self.weight_decay*lr_t

            # weight update
            self.updates.append(K.update(p, new_p))

            # EMA
            if self.ema_momentum:
                ema = self.ema_momentum * e - (1.-self.ema_momentum)*new_p
                # bias correction
                ema = ema / (1-K.pow(self.ema_momentum, t))
                self.updates.append(K.update(e, ema))

        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': self.weight_decay,
            'clip_norm': self.clip_norm,
            'ema_momentum': self.ema_momentum,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':

    from swin import SwinTransformer
    import numpy as np

    model = SwinTransformer(input_shape=(224,224,3), n_classes=6, patch_size=4, emb_dim=96,
                            ape=False, num_layers=[2,2,6,2], num_heads=[3,6,12,24], window_size=7,
                            qkv_bias=True, qk_scale=None, mlp_ratio=4, attn_drop=0., ffn_drop=0.,
                            residual_drop=0.2)
    model.compile(AdamW(0.001), loss='categorical_crossentropy', metrics=['acc'])

    X = np.zeros((32,224,224,3))
    Y = np.zeros((32,6))
    model.fit(X,Y)


