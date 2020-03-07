import os

# close warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from templates.trainer_template import TrainerTemplate
from configs.default import *
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import imageio


class Trainer(TrainerTemplate):

    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, data, config):
        super(Trainer, self).__init__(model=None,
                                      data=data,
                                      config=config)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(self.config.GENERATOR_LEARNING_RATE)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.config.GENERATOR_LEARNING_RATE)

    @staticmethod
    def __generator_loss(fake_output):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def __discriminator_loss(real_output, fake_output):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)

        return real_loss + fake_loss

    @tf.function
    def __dcgan_special_train_step(self, real_image):
        noise = tf.random.normal([self.config.BATCH_SIZE, self.config.NOISE_DIM])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            fake_image = self.generator(noise, training=True)

            real_output = self.discriminator(real_image, training=True)
            fake_output = self.discriminator(fake_image, training=True)

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
            for batch_image, _ in tqdm(self.data, total=self.config.TOTAL_IMAGES / self.config.BATCH_SIZE,
                                       ncols=65,
                                       desc="Epoch {:02d}".format(epoch + 1)):
                self.__dcgan_special_train_step(batch_image)

            if (epoch+1) % self.config.CHECKPOINT_SAVE_FREQUENCY == 0:
                self.generate_preview(epoch=epoch)
                checkpoint.save(file_prefix=checkpoint_path)

        self.generate_gif()

    def generate_preview(self, epoch):
        fig = plt.figure(figsize=(4, 4))
        tf.random.set_seed(self.config.RANDOM_SEED)
        default_noise = tf.random.normal([4 * 4, self.config.NOISE_DIM])

        for i in range(4 * 4):
            plt.subplot(4, 4, i + 1)
            plt.imshow(self.generator(default_noise, training=False)[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis('off')

        save_path = os.path.join(".",
                                 self.config.LOG_ROOT,
                                 self.config.EXPERIMENT_NAME,
                                 self.config.GENERATE_PATH)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + '/image_at_epoch_{:02d}.png'.format(epoch))

        plt.show()
        return

    def generate_gif(self):
        png_image_path = os.path.join(".",
                                      self.config.LOG_ROOT,
                                      self.config.EXPERIMENT_NAME,
                                      self.config.GENERATE_PATH,
                                      "image_at_epoch_*.png")
        save_path = os.path.join(".",
                                 self.config.LOG_ROOT,
                                 self.config.EXPERIMENT_NAME,
                                 self.config.GIF_NAME)

        file_list = sorted(glob.glob(png_image_path))
        frames = []
        for file in file_list:
            frames.append(imageio.imread(file))

        imageio.mimsave(save_path, frames, 'GIF', duration=0.1)

    def get_checkpoint(self):
        checkpoint = tf.train.Checkpoint(generator=self.generator,
                                         generator_optimizer=self.generator_optimizer,
                                         discriminator=self.discriminator,
                                         discriminator_optimizer=self.discriminator_optimizer)
        return checkpoint
