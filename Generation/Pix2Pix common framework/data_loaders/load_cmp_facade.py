import os
import random

from templates.data_loader_template import DataLoaderTemplate
from utils.data_augmentation import *
from utils.image_io import *


class DataLoaderConfig:
    def __init__(self,
                 resize_up_size,
                 output_size,
                 buffer_size,
                 batch_size,
                 test_batch_size):
        # used in __random_jitter
        self.RESIZE_UP_SIZE = resize_up_size  # the resolution will go up to RESIZE_UP_SIZE first
        self.OUTPUT_SIZE = output_size  # randomly cropped to OUTPUT_SIZE

        # dataset config
        self.BUFFER_SIZE = buffer_size  # random buffer
        self.BATCH_SIZE = batch_size
        self.TEST_BATCH_SIZE = test_batch_size


class DataLoader(DataLoaderTemplate):

    def __init__(self, config):
        super(DataLoader, self).__init__(config=config)

    @staticmethod
    def __read_image(image_file):
        """
        load image from file,
        special: split image in cmp facade dataset.
        :param image_file: image path
        :return: image
        """
        image_merged = normalize_uint8(read_image(image_file))  # [h,w,3] uint8

        width = tf.shape(image_merged)[1] // 2

        real_image = image_merged[:, :width, :]
        input_image = image_merged[:, width:, :]

        return input_image, real_image

    def __random_jitter(self, image_a, image_b):
        """
        1. resize to a bigger size
        2. randomly crop image to the target size
        3. randomly flip the image
        process 2 image at the sab=me time, with the same pattern for crop and flip
        :param image_a:
        :param image_b:
        :return: processed image
        """
        image_a = resize(image_a, [self.config.RESIZE_UP_SIZE, self.config.RESIZE_UP_SIZE])
        image_b = resize(image_b, [self.config.RESIZE_UP_SIZE, self.config.RESIZE_UP_SIZE])

        seed = random.randint(0, 99999)

        image_a = random_crop(image_a, [self.config.OUTPUT_SIZE, self.config.OUTPUT_SIZE, 3], seed)
        image_b = random_crop(image_b, [self.config.OUTPUT_SIZE, self.config.OUTPUT_SIZE, 3], seed)

        if random.uniform(0, 1) > 0.5:
            a_out = tf.image.flip_left_right(image_a)
            b_out = tf.image.flip_left_right(image_b)
            return a_out, b_out
        else:
            return image_a, image_b

    def __resize(self, image_a, image_b):
        """
        resize image for test and validation data,
        they do not need flip and crop in __random_jitter()
        :param image_a:
        :param image_b:
        :return:
        """
        return resize(image_a, [self.config.OUTPUT_SIZE, self.config.OUTPUT_SIZE]), \
               resize(image_b, [self.config.OUTPUT_SIZE, self.config.OUTPUT_SIZE])

    def load(self):
        _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

        path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                              origin=_URL,
                                              extract=True)

        data_path = os.path.join(os.path.dirname(path_to_zip), 'facades/')

        # create dataset
        self.dataset = {
            "train": tf.data.Dataset.list_files(data_path + 'train/*.jpg')
                .map(self.__read_image)
                .map(self.__random_jitter)
                .shuffle(self.config.BUFFER_SIZE)
                .batch(self.config.BATCH_SIZE, drop_remainder=True)
            ,
            "test": tf.data.Dataset.list_files(data_path + 'test/*.jpg')
                .map(self.__read_image)
                .map(self.__resize)
                .batch(self.config.TEST_BATCH_SIZE, drop_remainder=True)
            ,
            "val": tf.data.Dataset.list_files(data_path + 'val/*.jpg')
                .map(self.__read_image)
                .map(self.__resize)
                .batch(self.config.TEST_BATCH_SIZE, drop_remainder=True)
            ,
        }


if __name__ == "__main__":
    test_config = DataLoaderConfig(
        resize_up_size=290,
        output_size=256,
        buffer_size=200,
        batch_size=10,
        test_batch_size=10
    )

    data = DataLoader(test_config)
    data.load()
    dataset = data.get_dataset()

    for x, y in dataset["train"]:
        quick_view_image_sequence([x[0], y[0]], n=2)
        print(x[0], y[0])
        break
    for x, y in dataset["test"]:
        quick_view(x[2])
        break
    for x, y in dataset["val"]:
        quick_view(x[2])
        break
