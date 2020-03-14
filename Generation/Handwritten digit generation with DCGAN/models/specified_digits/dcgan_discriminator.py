import tensorflow as tf
from tensorflow.keras import layers, Model
from models.random_noise.dcgan_discriminator import DCGANDiscriminator as DCGANDiscriminatorPreDefine


class DCGANDiscriminator(DCGANDiscriminatorPreDefine):

    def build(self):
        input_digits = layers.Input(shape=(10,), dtype=tf.float32)  # one_hot
        input_image = layers.Input(shape=(28, 28, 1), dtype=tf.float32)

        input_image_flat = layers.Flatten()(input_image)
        inputs = layers.concatenate([input_digits, input_image_flat])
        assert tuple(inputs.shape) == (None, 28 * 28 + 10)

        hidden_layer = layers.Dense(28 * 28)(inputs)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Reshape((28, 28, 1))(hidden_layer)

        hidden_layer = layers.Conv2D(filters=16,
                                     kernel_size=(5, 5),
                                     padding="SAME",
                                     strides=(2, 2))(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 14, 14, 16)

        hidden_layer = layers.Conv2D(filters=64,
                                     kernel_size=(5, 5),
                                     padding="SAME",
                                     strides=(2, 2))(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 7, 7, 64)

        hidden_layer = layers.Conv2D(filters=256,
                                     kernel_size=(5, 5),
                                     padding="SAME",
                                     strides=(2, 2))(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 4, 4, 256)

        hidden_layer = layers.GlobalAveragePooling2D()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 256)

        hidden_layer = layers.Dense(1)(hidden_layer)

        self.model = Model(inputs=[input_digits, input_image], outputs=hidden_layer, name="discriminator")
