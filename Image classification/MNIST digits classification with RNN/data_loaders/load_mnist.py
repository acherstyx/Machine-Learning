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

    def __load_mnist(self):
        """
        dataset will be save to '~/.keras/datasets' automatically
        :return: a init dataset
        """
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        # reshape to (-1, 28, 28) and change data type to tf.float32
        train_images = tf.cast(
            tf.reshape(train_images,
                       (-1, 28, 28)),
            tf.float32
        )
        test_images = tf.cast(
            tf.reshape(test_images,
                       (-1, 28, 28)),
            tf.float32
        )
        # normalize
        train_images = (train_images - 127.5) / 127.5
        test_images = (test_images - 127.5) / 127.5

        return tf.data.Dataset.from_tensor_slices((train_images, train_labels)) \
                   .batch(batch_size=self.config.BATCH_SIZE,
                          drop_remainder=self.config.DROP_REMINDER), \
               tf.data.Dataset.from_tensor_slices((test_images, test_labels)) \
                   .batch(batch_size=self.config.BATCH_SIZE,
                          drop_remainder=self.config.DROP_REMINDER)

    def load(self):
        train, val = self.__load_mnist()
        self.dataset = {
            "train": train,
            "valid": val
        }


if __name__ == "__main__":
    class SampleConfig:
        BATCH_SIZE = 1
        DROP_REMINDER = True


    dataset = LoadMNIST(SampleConfig).get_dataset()

    for sample in dataset["train"]:
        print("image:", sample[0].shape)
        print("label:", sample[1].shape)
        import matplotlib.pyplot as plt

        plt.imshow(sample[0][0, :, :], cmap="gray")
        print("image sample: \n", sample[0][0, :, :])
        plt.show()
        break
