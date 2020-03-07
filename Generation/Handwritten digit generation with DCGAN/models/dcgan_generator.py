import tensorflow as tf
from tensorflow.keras import layers, Model
from templates.model_templet import ModelTemplate


class DCGANGenerator(ModelTemplate):
    def __init__(self, config):
        super(DCGANGenerator, self).__init__(config=config)
        self.build()

    def build(self):
        inputs = layers.Input(shape=self.config.NOISE_DIM,
                              dtype=tf.float32)

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
                                              activation="tanh",    # use tanh as activate function
                                              strides=(2, 2))(hidden_layer)
        assert tuple(hidden_layer.shape) == (None, 28, 28, 1)

        self.model = Model(inputs=inputs, outputs=hidden_layer, name="generator")


if __name__ == "__main__":
    class GeneratorConfig:
        NOISE_DIM = 100

    model_generator = DCGANGenerator(GeneratorConfig).model
    print("input    ", model_generator.input_shape)
    print("output:  ", model_generator.output_shape)

    noise = tf.random.normal(shape=(1, 100))
    output = model_generator(noise)

    import matplotlib.pyplot as plt

    plt.imshow(output[0, :, :, 0], cmap="gray")
    plt.show()
