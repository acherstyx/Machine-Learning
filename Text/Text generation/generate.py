import os

# close warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from config import *  # load CONST
from utils import load_data
from model import simple_gru as rnn_model
import tensorflow as tf

from train import vec2word

model = rnn_model.build_model(len(vec2word), EMBEDDING_DIM, RNN_UNITES, 1)
model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_ROOT))
model.build(tf.TensorShape([1, None]))
model.summary()


def generate_text(trained_model, start_string, vec2word_dict, num_generate):
    trained_model.reset_states()

    word2vec_dict = {v: k for k, v in vec2word_dict.items()}

    start_string_vec = [word2vec_dict[s] for s in start_string]
    start_string_vec = tf.expand_dims(start_string_vec, 0)

    text_generated = []

    for _ in range(num_generate):
        prediction = trained_model(start_string_vec)
        prediction = tf.squeeze(prediction, 0)

        predicted_id = tf.random.categorical(prediction, num_samples=1)[-1, 0].numpy()

        start_string_vec = tf.expand_dims([predicted_id], 0)

        text_generated.append(vec2word_dict[predicted_id])

    return start_string + ''.join(text_generated)


print(generate_text(model, start_string=u"ROMEO: ", vec2word_dict=vec2word, num_generate=1000))
