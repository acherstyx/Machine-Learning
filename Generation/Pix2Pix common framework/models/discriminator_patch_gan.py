import tensorflow as tf

from tensorflow.keras import layers, Model, losses
from templates.model_template import ModelTemplate


class PatchGAN(ModelTemplate):
    def __init__(self, config):
        super(PatchGAN, self).__init__(config)

    def build(self):
        # same to the inputs of generator
        init_inputs = layers.Input(shape=(256, 256, 3),
                                   dtype=tf.float32)
        # the label image or generated image, should classify which is fake
        target_inputs = layers.Input(shape=(256, 256, 3),
                                     dtype=tf.float32)

        hidden_layer = layers.Concatenate()([init_inputs, target_inputs])

        hidden_layer = layers.Conv2D(filters=64,
                                     kernel_size=4,
                                     strides=(2, 2),
                                     padding="SAME",
                                     activation=None,
                                     )(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 128, 128, 64)

        hidden_layer = layers.Conv2D(filters=128,
                                     kernel_size=4,
                                     strides=(2, 2),
                                     padding="SAME",
                                     activation=None)(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 64, 64, 128)

        hidden_layer = layers.Conv2D(filters=256,
                                     kernel_size=4,
                                     strides=(2, 2),
                                     padding="SAME",
                                     activation=None)(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 32, 32, 256)

        # add zero to the edge of 2d hidden layer
        hidden_layer = layers.ZeroPadding2D(padding=(1, 1))(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 34, 34, 256)

        initializer = tf.random_normal_initializer(0., 0.02)

        hidden_layer = layers.Conv2D(filters=512,
                                     kernel_size=4,
                                     activation=None,
                                     kernel_initializer=initializer)(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.LeakyReLU()(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 31, 31, 512)

        hidden_layer = layers.ZeroPadding2D(padding=(1, 1))(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 33, 33, 512)

        outputs = layers.Conv2D(filters=1,
                                kernel_size=4,
                                activation='tanh',
                                kernel_initializer=initializer)(hidden_layer)
        assert tuple(outputs.shape) == (None, 30, 30, 1)

        self.model = Model(inputs=[init_inputs, target_inputs],
                           outputs=outputs)
        return self


if __name__ == "__main__":
    model = PatchGAN(None)
    model.build()
    model.show_summary(with_plot=True)

    from data_loaders.load_cmp_facade import *

    config = DataLoaderConfig(resize_up_size=300,
                              output_size=256,
                              buffer_size=20,
                              batch_size=10,
                              test_batch_size=5)
    data_loader = DataLoader(config)
    data_loader.load()
    disc = model.model
    for test_input, target_input in data_loader.get_dataset()["val"]:
        output = disc([test_input*177.5+177.5, target_input], training=False)
        quick_view(output[0, ..., -1])
