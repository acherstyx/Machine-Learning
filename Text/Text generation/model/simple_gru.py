import tensorflow as tf
import tensorflow.keras.layers as layers


def loss(label, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(label,
                                                           logits,
                                                           from_logits=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    text_vec = layers.Input(shape=(None,), dtype=tf.int32, batch_size=batch_size)
    embedding_vector = layers.Embedding(input_dim=vocab_size,
                                        output_dim=embedding_dim)(text_vec)
    rnn_final_state = layers.GRU(units=rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer="glorot_uniform")(embedding_vector)
    output = layers.Dense(vocab_size)(rnn_final_state)

    model = tf.keras.Model(inputs=text_vec,
                           outputs=output,
                           name="Simple_GRU_model")

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss)

    model.summary()

    return model
