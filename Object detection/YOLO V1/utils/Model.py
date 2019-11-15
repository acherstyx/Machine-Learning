import os
import tensorflow as tf
import tensorflow_hub as hub
import utils.Config as Config
from utils.YoloLoss import yolo_loss


os.environ["PATH"] += os.pathsep + "D:/graphviz/bin"
os.environ['TFHUB_CACHE_DIR'] = './.data/'


def yolo_model(show_summary=False):
    input_layer = tf.keras.Input(shape=(Config.ImageSize, Config.ImageSize, 3), name="input")

    # Inception model from tfhub
    inception_feature_extractor = hub.keras_layer.KerasLayer(
        "https://storage.googleapis.com/tfhub-modules/google/tf2-preview/inception_v3/feature_vector/4.tar.gz",
        output_shape=[2048],
        trainable=False)
    # Follow layers
    hidden_layer = inception_feature_extractor(input_layer)
    hidden_layer = tf.keras.layers.Dense(Config.CellSize * Config.CellSize * 30,
                                         activation="sigmoid")(hidden_layer)
    output = tf.keras.layers.Reshape((Config.CellSize,
                                      Config.CellSize,
                                      5 * Config.BoxPerCell + Config.ClassesNum),
                                     name="output")(hidden_layer)
    # Get model
    net_model = tf.keras.Model(inputs=input_layer,
                               outputs=output,
                               name="Yolo_Model")
    net_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                      loss=yolo_loss)
    if show_summary:
        # Layer summary
        net_model.summary()
        tf.keras.utils.plot_model(model=net_model,
                                  to_file='Net.png',
                                  show_shapes=True,
                                  dpi=300)
    return net_model


if __name__ == "__main__":
    print("Is there a GPU available: ", tf.test.is_gpu_available())
    model = yolo_model(show_summary=True)
