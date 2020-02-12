import os

# close warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from config import *  # load CONST
from utils import load_data
from model import simple_gru as rnn_model
import tensorflow as tf

dataset, vec2word = load_data.load_text_single_ver(
    tf.keras.utils.get_file('shakespeare.txt',
                            'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
)
dataset = load_data.split_train_sample(
    load_data.make_batch_dataset(dataset, WINDOWS_SIZE, SHIFT),
    shift=SHIFT,
)

dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
model = rnn_model.build_model(len(vec2word), EMBEDDING_DIM, RNN_UNITES, BATCH_SIZE)

callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                                         save_weights_only=True)

if __name__ == "__main__":
    history = model.fit(dataset,
                        epochs=EPOCHS,
                        callbacks=[callback_checkpoint])
