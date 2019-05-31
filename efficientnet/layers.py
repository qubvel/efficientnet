import tensorflow as tf
import keras.backend as K
import keras.layers as KL
from keras.utils.generic_utils import get_custom_objects


class Swish(KL.Layer):

    def call(self, inputs):
        return tf.nn.swish(inputs)


class DropConnect(KL.Layer):

    def __init__(self, drop_connect_rate=0.):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.div(inputs, keep_prob) * binary_tensor
            return output


        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config['drop_connect_rate'] = self.drop_connect_rate
        return config


get_custom_objects().update({
    'DropConnect': DropConnect,
    'Swish': Swish,
})
