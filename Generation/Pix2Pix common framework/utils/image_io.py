import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import imageio


def read_image(image_file):
    """
    read image in file
    :param image_file: file path
    :return:
    """
    return tf.image.decode_jpeg(
        tf.io.read_file(image_file),
        channels=3,
        # expand_animations=False
    )


def quick_view_image_sequence(image, n=1):
    """
    show image with matplotlib
    :param image: a python list contain `n` iamges
    :param n: the number of image in parameter 'image'
    """
    plt.figure(figsize=(n, 1), dpi=300)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(image[i])
        plt.axis('off')
    plt.show()


def quick_view(image):
    plt.imshow(image)
    plt.show()


def generate_gif(png_image_path, gif_save_path):
    """
    turn all the images in the folder into a gif
    :param png_image_path: the 
    :param gif_save_path: output gif
    """
    file_list = sorted(glob.glob(png_image_path))
    frames = []
    for file in file_list:
        frames.append(imageio.imread(file))

    imageio.mimsave(gif_save_path, frames, 'GIF', duration=0.1)
