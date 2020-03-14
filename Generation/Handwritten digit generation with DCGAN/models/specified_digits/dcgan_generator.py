import tensorflow as tf
from tensorflow.keras import layers, Model
from models.random_noise.dcgan_generator import DCGANGenerator as DCGANGeneratorPreDefine


class DCGANGenerator(DCGANGeneratorPreDefine):

    def build(self):
        input_specified_number = layers.Input(shape=[1, ], dtype=tf.float32)
        input_noise = layers.Input(shape=self.config.NOISE_DIM,
                                   dtype=tf.float32)

        inputs = tf.concat(
            values=[input_specified_number, input_noise],
            axis=1
        )
        assert tuple(inputs.shape) == (None, 1+self.config.NOISE_DIM)

        hidden_layer = layers.Dense(7 * 7 * 256)(inputs)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Reshape(target_shape=(7, 7, 256))(hidden_layer)

        hidden_layer = layers.Conv2DTranspose(filters=64,
                                              kernel_size=(2, 2),
                                              padding="SAME")(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 7, 7, 64)

        hidden_layer = layers.Conv2DTranspose(filters=16,
                                              kernel_size=(2, 2),
                                              padding="SAME",
                                              strides=(2, 2))(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 14, 14, 16)

        hidden_layer = layers.Conv2DTranspose(filters=1,
                                              kernel_size=(5, 5),
                                              padding="SAME",
                                              activation="tanh",  # use tanh as activate function
                                              strides=(2, 2))(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 28, 28, 1)

        self.model = Model(inputs=[input_specified_number, input_noise],
                           outputs=hidden_layer,
                           name="generator")
