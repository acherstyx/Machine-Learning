from models.rnn import *


class AllRNNConfig(RowByRowRNNConfig):
    pass


class AllRNN(RowByRowRNN):

    def build(self, *args):
        input_images = layers.Input(shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                                    name="inputs",
                                    dtype=tf.float32)

        hidden_layer = input_images
        hidden_layer = layers.Dropout(self.config.DROPOUT_RATE)(hidden_layer)
        for rnn_unit in self.config.UNIT_STACK[:-1]:
            hidden_layer = layers.GRU(units=rnn_unit,
                                      return_sequences=True)(hidden_layer)
            hidden_layer = layers.BatchNormalization()(hidden_layer)

        hidden_layer = layers.GRU(units=self.config.UNIT_STACK[-1],
                                  return_sequences=False)(hidden_layer)

        outputs = hidden_layer

        self.model = Model(inputs=input_images,
                           outputs=outputs)
        return self


if __name__ == "__main__":
    config = AllRNNConfig(image_size=28,
                          unit_stack=[64, 128, 256, 512, 128, 10],
                          dropout_rate=0.2)
    model = AllRNN(config)

    model.show_summary(with_plot=True)
