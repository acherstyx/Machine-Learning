import os
# close warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from config import *  # load CONST
from model import simple_gru as rnn_model
import tensorflow as tf

dataset = load_data.sample_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vec2word = load_data.sample_dict

model = rnn_model.build_model(len(vec2word), 256, 1024, 64)

# for input, label in dataset.take(1):
#     output = model(input)
#     print(output)

callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                                         save_weights_only=True)

history = model.fit(dataset,
                    epochs=EPOCHS,
                    callbacks=[callback_checkpoint])
