from templates.trainer_template import TrainerTemplate
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics


class MNISTTrainerConfig:
    def __init__(self,
                 epochs,
                 learning_rate):
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate


class MNISTTrainer(TrainerTemplate):
    def __init__(self, model, data, config):
        super(MNISTTrainer, self).__init__(model, data, config)

    def train(self, *args):
        self.model: tf.keras.Model

        self.model.compile(
            optimizer=optimizers.Adam(self.config.LEARNING_RATE),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        self.model.fit_generator(self.data,
                                 epochs=self.config.EPOCHS)
