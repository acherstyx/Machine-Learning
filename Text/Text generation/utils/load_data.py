import tensorflow as tf
import numpy as np


def load_text_single_ver(file_path: str) -> (tf.data.Dataset, dict):
    """
    load text from a single file and transform it into a dataset
    :param file_path: the path to a text file
    """

    row_text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # transform all the character into a set
    all_char = sorted(set(row_text))

    word2vec_dict = {a_char: index for index, a_char in enumerate(all_char)}
    vec2word_dict = {index: a_char for index, a_char in enumerate(all_char)}

    text_vector = tf.data.Dataset.from_tensor_slices([word2vec_dict[a_char] for a_char in row_text])

    return text_vector, vec2word_dict


def make_windows_dataset(text_material: tf.data.Dataset, windows_size=10, shift=1, stride=1) -> tf.data.Dataset:
    windows = text_material.window(windows_size,
                                   shift=shift,
                                   stride=stride,
                                   drop_remainder=True)

    def map_func(sub: tf.data.Dataset):
        return sub.batch(windows_size,
                         drop_remainder=True)

    return windows.flat_map(map_func)


def split_train_sample(batched_data: tf.data.Dataset, shift=1) -> tf.data.Dataset:
    def split(single_batch: tf.Tensor):
        return single_batch[:-shift], single_batch[shift:]

    return batched_data.map(split)


sample_dataset, sample_dict = load_text_single_ver(
    tf.keras.utils.get_file('shakespeare.txt',
                            'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
)
sample_dataset = split_train_sample(
    make_windows_dataset(sample_dataset)
)