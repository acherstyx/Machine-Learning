import os
import tensorflow as tf
import tensorflow_hub as hub
import utils.Config as Config
from utils.YoloLoss import yolo_loss

os.environ["PATH"] += os.pathsep + "D:/graphviz/bin"
os.environ['TFHUB_CACHE_DIR'] = './.data/'


def yolo_model(model_type="TRANSFER", show_summary=False):
    """
    build neural network model for yolo
    @param model_type: "TRANSFER" or "ORIGINAL"
    @param show_summary: Show summary of the model
    @return: keras model
    """
    input_layer = tf.keras.Input(shape=(Config.ImageSize, Config.ImageSize, 3), name="input")

    hidden_layer = tf.keras.layers.Dropout(Config.Dropout_Image)(input_layer)

    # TODO: Model switch for different model
    if model_type == "TRANSFER":
        # Inception model from tfhub
        inception_feature_extractor = hub.keras_layer.KerasLayer(
            "https://storage.googleapis.com/tfhub-modules/google/tf2-preview/inception_v3/feature_vector/4.tar.gz",
            output_shape=[2048],
            trainable=True)
        # Follow layers
        hidden_layer = inception_feature_extractor(hidden_layer)
        hidden_layer = tf.keras.layers.Dropout(Config.Dropout_Output)(hidden_layer)
    elif model_type == "ORIGINAL":
        # period 1 - reduce image size
        hidden_layer = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=(7, 7),
                                              strides=(2, 2),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                 strides=(2, 2),
                                                 padding="SAME",
                                                 )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=192,
                                              kernel_size=(3, 3),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                 strides=(2, 2),
                                                 padding="SAME",
                                                 )(hidden_layer)

        # period 2 - increase deep
        hidden_layer = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=(1, 1),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=(3, 3),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=(1, 1),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(3, 3),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                 strides=(2, 2),
                                                 padding="SAME",
                                                 )(hidden_layer)

        # period 3 - inception_1
        #   group 1
        inception_1_group_1 = tf.keras.layers.Conv2D(filters=256,
                                                     kernel_size=(1, 1),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(hidden_layer)
        inception_1_group_1 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_1)
        inception_1_group_1 = tf.keras.layers.Conv2D(filters=512,
                                                     kernel_size=(3, 3),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(inception_1_group_1)
        inception_1_group_1 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_1)
        #   group 2
        inception_1_group_2 = tf.keras.layers.Conv2D(filters=256,
                                                     kernel_size=(1, 1),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(hidden_layer)
        inception_1_group_2 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_2)
        inception_1_group_2 = tf.keras.layers.Conv2D(filters=512,
                                                     kernel_size=(3, 3),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(inception_1_group_2)
        inception_1_group_2 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_2)
        #   group 3
        inception_1_group_3 = tf.keras.layers.Conv2D(filters=256,
                                                     kernel_size=(1, 1),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(hidden_layer)
        inception_1_group_3 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_3)
        inception_1_group_3 = tf.keras.layers.Conv2D(filters=512,
                                                     kernel_size=(3, 3),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(inception_1_group_3)
        inception_1_group_3 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_3)
        #    group 4
        inception_1_group_4 = tf.keras.layers.Conv2D(filters=256,
                                                     kernel_size=(1, 1),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(hidden_layer)
        inception_1_group_4 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_4)
        inception_1_group_4 = tf.keras.layers.Conv2D(filters=512,
                                                     kernel_size=(3, 3),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(inception_1_group_4)
        inception_1_group_4 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_1_group_4)
        #   merge all layers
        hidden_layer = tf.keras.layers.Concatenate(axis=-1)([inception_1_group_1, inception_1_group_2,
                                                             inception_1_group_3, inception_1_group_4])
        hidden_layer = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(1, 1),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=1024,
                                              kernel_size=(3, 3),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                 strides=(2, 2),
                                                 padding="SAME",
                                                 )(hidden_layer)

        # period 4 - inception 2
        #   group 1
        inception_2_group_1 = tf.keras.layers.Conv2D(filters=512,
                                                     kernel_size=(1, 1),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(hidden_layer)
        inception_2_group_1 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_2_group_1)
        inception_2_group_1 = tf.keras.layers.Conv2D(filters=1024,
                                                     kernel_size=(3, 3),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(inception_2_group_1)
        inception_2_group_1 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_2_group_1)
        #   group 2
        inception_2_group_2 = tf.keras.layers.Conv2D(filters=512,
                                                     kernel_size=(1, 1),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(hidden_layer)
        inception_2_group_2 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_2_group_2)
        inception_2_group_2 = tf.keras.layers.Conv2D(filters=1024,
                                                     kernel_size=(3, 3),
                                                     padding="SAME",
                                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                                     use_bias=False,
                                                     )(inception_2_group_2)
        inception_2_group_2 = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(inception_2_group_2)
        #   merge all layers
        hidden_layer = tf.keras.layers.Concatenate(axis=-1)([inception_2_group_1, inception_2_group_2])
        hidden_layer = tf.keras.layers.Conv2D(filters=1024,
                                              kernel_size=(3, 3),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=1024,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False,
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)

        # period 5 last conventional layers
        hidden_layer = tf.keras.layers.Conv2D(filters=1024,
                                              kernel_size=(3, 3),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=1024,
                                              kernel_size=(3, 3),
                                              padding="SAME",
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                              use_bias=False
                                              )(hidden_layer)
        hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPool2D(pool_size=(4, 4),
                                                 strides=(2, 2),
                                                 padding="valid")(hidden_layer)
        hidden_layer = tf.keras.layers.Dropout(Config.Dropout_Output)(hidden_layer)
    else:
        raise TypeError

    # FC layers
    hidden_layer = tf.keras.layers.Flatten()(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(4096,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                         )(hidden_layer)
    hidden_layer = tf.keras.layers.ReLU(negative_slope=Config.ReLU_Slope)(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(Config.CellSize * Config.CellSize * 30,
                                         activation="sigmoid",
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal())(hidden_layer)
    output = tf.keras.layers.Reshape((Config.CellSize,
                                      Config.CellSize,
                                      5 * Config.BoxPerCell + Config.ClassesNum),
                                     name="output")(hidden_layer)
    # Get model
    net_model = tf.keras.Model(inputs=input_layer,
                               outputs=output,
                               name="Yolo_Model")
    net_model.compile(optimizer=tf.keras.optimizers.Adam(Config.LearningRate),
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
