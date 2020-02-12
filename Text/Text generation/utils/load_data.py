import tensorflow as tf
import numpy as np


def load_text_single_ver(file_path: str) -> (tf.data.Dataset, dict):
    """
    load text from a single file and transform it into a dataset
    :param file_path: the path to a text file
    """

    raw_text = open(file_path, 'rb').read().decode(encoding='utf-8')

    # transform all the character into a set
    all_char = sorted(set(raw_text))

    word2vec_dict = {a_char: index for index, a_char in enumerate(all_char)}
    vec2word_dict = {index: a_char for index, a_char in enumerate(all_char)}

    text_vector = tf.data.Dataset.from_tensor_slices(np.array([word2vec_dict[a_char] for a_char in raw_text]))

    return text_vector, vec2word_dict


def make_windows_dataset(text_material: tf.data.Dataset, windows_size, shift=1, stride=1):
    windows = text_material.window(windows_size,
                                   shift=shift,
                                   stride=stride,
                                   drop_remainder=True)

    def map_func(sub: tf.data.Dataset):
        return sub.batch(windows_size)

    return windows.flat_map(map_func)


def make_batch_dataset(text_material: tf.data.Dataset, batch_size, shift):
    return text_material.batch(batch_size=batch_size + shift,
                               drop_remainder=True)


def split_train_sample(batched_data: tf.data.Dataset, shift=1) -> tf.data.Dataset:
    def split(single_batch: tf.Tensor):
        return single_batch[:-shift], single_batch[shift:]

    return batched_data.map(split)


WINDOWS_SIZE = 100
SHIFT = 1

sample_dataset, sample_dict = load_text_single_ver(
    tf.keras.utils.get_file('shakespeare.txt',
                            'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
)
sample_dataset = split_train_sample(
    make_batch_dataset(sample_dataset, WINDOWS_SIZE, SHIFT),
    shift=SHIFT,
)

# ii = 0
# for i in sample_dataset.batch(64,drop_remainder=True):
#     ii+=1
#     print(ii)