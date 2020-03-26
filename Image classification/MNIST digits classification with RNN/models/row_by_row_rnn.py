import tensorflow as tf
from tensorflow.keras import layers, Model
from templates.model_template import ModelTemplate


class RowByRowRNNConfig:
    def __init__(self,
                 image_size,
                 unit_stack):
        self.IMAGE_SIZE = image_size
        self.UNIT_STACK = unit_stack # for example, [64, 128, 512, 1024]


class RowByRowRNN(ModelTemplate):
    def __init__(self, config):
        super(RowByRowRNN, self).__init__(config)

    def build(self, *args):
        input_images = layers.Input(shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                                    name="inputs",
                                    dtype=tf.float32)

        hidden_layer = input_images
        for rnn_unit in self.config.UNIT_STACK[:-1]:
            hidden_layer = layers.GRU(units=rnn_unit,
                                      return_sequences=True)(hidden_layer)
            hidden_layer = layers.BatchNormalization()(hidden_layer)

        hidden_layer = layers.GRU(units=self.config.UNIT_STACK[-1],
                                  return_sequences=False)(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)

        hidden_layer = layers.Dense(units=1024,
                                    activation="relu")(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Dense(units=1024,
                                    activation="relu")(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Dense(units=10,
                                    activation="relu")(hidden_layer)
        # outputs = layers.Softmax()(hidden_layer)
        outputs = hidden_layer

        self.model = Model(inputs=input_images,
                           outputs=outputs)
        return self


if __name__ == "__main__":
    test_config = RowByRowRNNConfig(image_size=28,
                                    unit_stack=[64, 128, 512, 1024])
    model = RowByRowRNN(test_config)
    model.show_summary(True)
