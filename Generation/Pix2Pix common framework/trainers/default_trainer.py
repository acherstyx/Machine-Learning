import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses, optimizers, Model
from tqdm import tqdm

from templates.trainer_template import TrainerTemplate
from utils.image_io import *


class DefaultTrainerConfig:
    def __init__(self,
                 experiment_name,
                 _lambda,
                 generator_learning_rate,
                 discriminator_learning_rate,
                 epoch,
                 log_root,
                 save_freq,
                 predict_dpi):
        # the weight of l1 loss in total loss
        self.LAMBDA = _lambda
        self.GENERATOR_LEARNING_RATE = generator_learning_rate
        self.DISCRIMINATOR_LEARNING_RATE = discriminator_learning_rate
        self.EPOCH = epoch
        self.LOG_ROOT = log_root
        self.EXPERIMENT_NAME = experiment_name
        self.SAVE_FREQ = save_freq
        self.PREDICT_DPI = predict_dpi


class DefaultTrainer(TrainerTemplate):
    def __init__(self, generator: Model, discriminator: Model, data_loader: object, config: DefaultTrainerConfig):
        super(DefaultTrainer, self).__init__(model=None,
                                             data=data_loader,
                                             config=config)
        self.generator = generator
        self.discriminator = discriminator

        self.generator_optimizer = optimizers.Adam(self.config.GENERATOR_LEARNING_RATE, beta_1=0.5)
        self.discriminator_optimizer = optimizers.Adam(self.config.DISCRIMINATOR_LEARNING_RATE, beta_1=0.5)

        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(".",
                         self.config.LOG_ROOT,
                         self.config.EXPERIMENT_NAME,
                         self.timestamp,
                         "summary"
                         )
        )

        self.image_save_path = os.path.join(".",
                                            self.config.LOG_ROOT,
                                            self.config.EXPERIMENT_NAME,
                                            self.timestamp,
                                            "image")

        self.step = 1

    def __generator_loss(self, generate_image, target_image, disc_result):
        """
        loss for generator, include the l1 loss and discriminator loss
        :param generate_image: the image generated
        :param target_image: the target result of generator (label)
        :param disc_result: the output of discriminator
        :return: total loss
        """
        disc_loss = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_result),
                                                                disc_result)
        l1_loss = tf.reduce_mean(tf.abs(generate_image - target_image))

        total_generate_loss = disc_loss + l1_loss * self.config.LAMBDA

        return total_generate_loss

    @staticmethod
    def __discriminator_loss(disc_real_output, disc_fake_output):
        """
        loss function for discriminator
        :param disc_real_output: the output of discriminator when the input is real label
        :param disc_fake_output: the output of discriminator when the input is fake label
        :return: total loss
        """
        real_loss = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output),
                                                                disc_real_output)
        fake_loss = losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_fake_output),
                                                                disc_fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def __gan_train_step(self, input_image, target_image):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_fake_output = self.discriminator([input_image, gen_output], training=True)
            disc_real_output = self.discriminator([input_image, target_image], training=True)

            generate_loss = self.__generator_loss(gen_output, target_image, disc_fake_output)
            discriminator_loss = self.__discriminator_loss(disc_real_output, disc_fake_output)

        generate_gradient = gen_tape.gradient(generate_loss,
                                              self.generator.trainable_variables)
        discriminator_gradient = disc_tape.gradient(discriminator_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generate_gradient,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient,
                                                         self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            self.step += 1
            tf.summary.scalar('gen_total_loss', generate_loss, step=self.step)
            tf.summary.scalar('disc_total_loss', generate_loss, step=self.step)

    def save(self):
        checkpoint = tf.train.Checkpoint(generate_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        checkpoint.save(
            os.path.join(".",
                         self.config.LOG_ROOT,
                         self.config.EXPERIMENT_NAME,
                         self.timestamp,
                         "checkpoint",
                         "save.ckpt")
        )

    def train(self):
        for epoch in range(self.config.EPOCH):
            for n, (input_image, target_image) in enumerate(tqdm(self.data["train"],
                                                                 desc="Epoch {:02d}".format(epoch + 1))):
                self.__gan_train_step(input_image, target_image)

            for test_input, target_input in self.data["val"]:
                self.predict(test_input, target_input, epoch)
                break

            if epoch % self.config.SAVE_FREQ == 0:
                self.save()

        self.save()
        self.generate_gif()

    def predict(self, test_input, target_image, epoch):
        predict_image = self.generator(test_input, training=False)
        batch_size = tuple(test_input.shape)[0]
        plt.figure(figsize=(3, batch_size),dpi=self.config.PREDICT_DPI)

        for img_index in range(batch_size):
            for index, image in enumerate([test_input[img_index],
                                           predict_image[img_index],
                                           target_image[img_index]]):
                plt.subplot(batch_size, 3, img_index * 3 + index + 1)
                plt.imshow((image + 1) / 2)
                plt.axis('off')

        os.makedirs(self.image_save_path, exist_ok=True)
        plt.savefig(self.image_save_path + '/image_at_epoch_{:02d}.png'.format(epoch))
        plt.show()

    def generate_gif(self):
        generate_gif(self.image_save_path, self.image_save_path + "generate_result.gif")


if __name__ == "__main__":
    test_config = DefaultTrainerConfig(_lambda=100,
                                       generator_learning_rate=1e-3,
                                       discriminator_learning_rate=1e-3)
