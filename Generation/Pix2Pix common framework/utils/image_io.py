import matplotlib.pyplot as plt
import tensorflow as tf


def read_image(image_file):
    """
    read image in file
    :param image_file: file path
    :return:
    """
    return tf.image.decode_image(
        tf.io.read_file(image_file),
        channels=3,
        expand_animations=False
    )


def quick_view(image, n=1):
    """
    show image with matplotlib
    :param image:
    :param n: the number of image in parameter 'image'
    """
    if n == 1:
        plt.imshow(image)
        plt.show()
    else:
        plt.figure(figsize=(n, 1), dpi=200)
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(image[i])
            plt.axis('off')
        plt.show()
