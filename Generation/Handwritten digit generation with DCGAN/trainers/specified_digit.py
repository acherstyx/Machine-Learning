import tensorflow as tf
# reuse the trainer
from trainers.random_noise import Trainer as TrainerPreDefine
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class Trainer(TrainerPreDefine):

    @tf.function
    def __dcgan_special_train_step(self, real_image, label):
        noise = tf.random.normal([self.config.BATCH_SIZE, self.config.NOISE_DIM])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            fake_image = self.generator([label, noise], training=True)

            real_output = self.discriminator([label, real_image], training=True)
            fake_output = self.discriminator([label, fake_image], training=True)

            generator_loss = self.__generator_loss(fake_output)
            discriminator_loss = self.__discriminator_loss(real_output, fake_output)

        grad_generator = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
        grad_discriminator = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(grad_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(grad_discriminator, self.discriminator.trainable_variables))

    def train(self):
        checkpoint = self.get_checkpoint()
        checkpoint_path = os.path.join(".",
                                       self.config.LOG_ROOT,
                                       self.config.EXPERIMENT_NAME,
                                       self.config.CHECKPOINT_PATH,
                                       "ckpt")

        for epoch in range(self.config.EPOCH):
            for batch_image, batch_label_int in tqdm(self.data,
                                                     total=self.config.TOTAL_IMAGES / self.config.BATCH_SIZE,
                                                     desc="Epoch {:02d}".format(epoch + 1),
                                                     ncols=65):
                batch_label = tf.one_hot(batch_label_int, 10)
                assert tuple(batch_label.shape) == (64, 10)
                self.__dcgan_special_train_step(batch_image, batch_label)

            self.generate_preview(epoch=epoch)
            if (epoch + 1) % self.config.CHECKPOINT_SAVE_FREQUENCY == 0:
                checkpoint.save(file_prefix=checkpoint_path)

        self.generate_gif()

    def generate_preview(self, epoch):
        fig = plt.figure(figsize=(10, 1))
        tf.random.set_seed(self.config.RANDOM_SEED)

        specified_number = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                       dtype=tf.int32)
        specified_number = tf.one_hot(specified_number, 10)

        assert tuple(specified_number.shape) == (10, 10)
        default_noise = tf.random.normal([10, self.config.NOISE_DIM])

        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(self.generator([specified_number, default_noise],
                                      training=False)[i, :, :, 0] * 127.5 + 127.5,
                       cmap="gray")
            plt.axis('off')

        save_path = os.path.join(".",
                                 self.config.LOG_ROOT,
                                 self.config.EXPERIMENT_NAME,
                                 self.config.GENERATE_PATH)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + '/image_at_epoch_{:02d}.png'.format(epoch))

        plt.show()
        return

    # @staticmethod
    # def __discriminator_loss(real_output, fake_output, label):
    #     real_loss = tf.keras.losses.sparse_categorical_crossentropy(label, real_output)
    #     fake_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.constant(10, shape=(64, 1)), fake_output)
    #     return real_loss + fake_loss
    #
    # @staticmethod
    # def __generator_loss(fake_output, label):
    #     return tf.keras.losses.sparse_categorical_crossentropy(label, fake_output)
