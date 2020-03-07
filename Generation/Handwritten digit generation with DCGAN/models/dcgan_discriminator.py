import tensorflow as tf
from tensorflow.keras import layers, losses, Model

from templates.model_templet import ModelTemplate


class DCGANDiscriminator(ModelTemplate):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__(None)
        self.build()

    def build(self):
        inputs = layers.Input(shape=(28, 28, 1))

        hidden_layer = layers.Conv2D(filters=16,
                                     kernel_size=(5, 5),
                                     padding="SAME",
                                     strides=(2, 2))(inputs)
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

        self.model = Model(inputs=inputs, outputs=hidden_layer, name="discriminator")


if __name__ == "__main__":
    discriminator_model = DCGANDiscriminator().model
    print("input:   ", discriminator_model.input_shape)
    print("output:  ", discriminator_model.output_shape)
