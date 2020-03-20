import tensorflow as tf


def random_crop(input_image, new_size, seed=None):
    """
    crop image to specified size
    :param input_image:
    :param new_size: format [w, h, channel]
    :param seed:
    :return:
    """
    return tf.image.random_crop(value=input_image,
                                size=new_size,
                                seed=seed)


def resize(input_image, new_size):
    """
    resize image
    :param input_image:
    :param new_size: image shape in [w, h]
    :return:
    """
    return tf.image.resize(images=input_image,
                           size=new_size,
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def __normalize_common(image, high, low=0):
    image = tf.cast(image, tf.float32)
    return (image / (high - low)) * 2 - 1


def normalize_uint8(image):
    return __normalize_common(image, 255)
