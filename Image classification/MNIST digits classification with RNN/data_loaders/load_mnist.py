import tensorflow as tf
from templates.data_loader_template import DataLoaderTemplate


class LoadMNISTConfig:
    def __init__(self,
                 batch_size,
                 drop_reminder):
        self.BATCH_SIZE = batch_size
        self.DROP_REMINDER = drop_reminder


class LoadMNIST(DataLoaderTemplate):
    def __init__(self, config):
        super(LoadMNIST, self).__init__(config)

    @staticmethod
    def __load_mnist() -> tf.data.Dataset:
        """
        dataset will be save to '~/.keras/datasets' automatically
        :return: a init dataset
        """
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        # reshape to (-1, 28, 28, 1) and change data type to tf.float32
        all_images = tf.cast(
            tf.reshape(
                tf.concat((train_images, test_images), 0),
                (-1, 28, 28)
            ),
            tf.float32
        )
        # normalize
        all_images = (all_images - 127.5) / 127.5
        all_labels = tf.concat((train_labels, test_labels), 0)

        return tf.data.Dataset.from_tensor_slices((all_images, all_labels))

    def load(self):
        self.dataset = self.__load_mnist().batch(batch_size=self.config.BATCH_SIZE,
                                                 drop_remainder=self.config.DROP_REMINDER)


if __name__ == "__main__":
    class SampleConfig:
        BATCH_SIZE = 1
        DROP_REMINDER = True

    dataset = LoadMNIST(SampleConfig).get_dataset()

    for sample in dataset:
        print("image:", sample[0].shape)
        print("label:", sample[1].shape)
        import matplotlib.pyplot as plt

        plt.imshow(sample[0][0, :, :, 0], cmap="gray")
        print("image sample: \n", sample[0][0, :, :, 0])
        plt.show()
        break
