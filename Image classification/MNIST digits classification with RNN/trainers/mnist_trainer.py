from templates.trainer_template import TrainerTemplate
import tensorflow as tf
import os
from tensorflow.keras import optimizers, losses, metrics, callbacks


class MNISTTrainerConfig:
    def __init__(self,
                 epochs,
                 log_root,
                 experiment_name,
                 scheduler,
                 clip_norm):
        self.EXPERIMENT_NAME = experiment_name
        self.EPOCHS = epochs

        self.LOG_ROOT = log_root
        self.SCHEDULER = scheduler
        self.CLIP_NORM = clip_norm


class MNISTTrainer(TrainerTemplate):
    def __init__(self, model, data, config):
        super(MNISTTrainer, self).__init__(model, data, config)

        self.CHECKPOINT_PATH = os.path.join(".",
                                            self.config.LOG_ROOT,
                                            self.config.EXPERIMENT_NAME,
                                            self.timestamp,
                                            "checkpoint",
                                            "save.ckpt")
        self.TENSORBOARD_PATH = os.path.join(".",
                                             self.config.LOG_ROOT,
                                             self.config.EXPERIMENT_NAME,
                                             self.timestamp,
                                             "tensorboard")

        self.__add_callbacks()

    def train(self, *args):
        self.model: tf.keras.Model

        self.model.compile(
            optimizer=optimizers.Adam(clipnorm=self.config.CLIP_NORM),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        self.model.fit(self.data["train"],
                       validation_data=self.data["valid"],
                       epochs=self.config.EPOCHS,
                       callbacks=self.callbacks)

    def __add_callbacks(self):
        checkpoint = callbacks.ModelCheckpoint(self.CHECKPOINT_PATH,
                                               save_best_only=False,
                                               save_weights_only=True)
        tensorboard = callbacks.TensorBoard(log_dir=self.TENSORBOARD_PATH,
                                            update_freq="batch",
                                            histogram_freq=1)
        scheduler = callbacks.LearningRateScheduler(schedule=self.config.SCHEDULER,
                                                    verbose=1)
        self.callbacks = [checkpoint, tensorboard, scheduler]
