import tensorflow as tf
import tensorflow.contrib.slim as slim
import Poker_Dataloader as tool
import os
import warnings
import numpy as np
import ImageIO as image_io
import cv2.cv2 as cv2
import os
import Data_Augmentation as tool

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.001  # 程序最多只能占用指定gpu50%的显存

STYLES = ["Clubs", "Diamonds", "Hearts", "Spades"]

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 加载模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 保存训练的地址
TRAIN_FILE_SUIT = './.save/suit recognition/save_model'
TRAIN_FILE_NUMBER = './.save/number recognition/save_model'

# 从获取参数，确认那些参数需要加载
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    # 枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中移除。
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []

    # 枚举所有需要训练的参数前缀，并通过这些前缀找到所有需要训练的参数。
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main():
    G1 = tf.Graph()
    G2 = tf.Graph()

    with G1.as_default():
        NUM_CLASS = 4
        # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签。
        images1 = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits1, _ = inception_v3.inception_v3(images1, num_classes=NUM_CLASS, is_training=True)

        softmaxed1 = tf.nn.softmax(logits1)

    with G2.as_default():
        NUM_CLASS = 13
        # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签。
        images2 = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits2, _ = inception_v3.inception_v3(images2, num_classes=NUM_CLASS, is_training=True)

        softmaxed2 = tf.nn.softmax(logits2)

    sess1 = tf.Session(graph=G1)
    sess2 = tf.Session(graph=G2)

    with sess1.as_default():
        with G1.as_default():
            saver = tf.train.Saver()
            saver.restore(sess1, TRAIN_FILE_SUIT)
    with sess2.as_default():
        with G2.as_default():
            saver = tf.train.Saver()
            saver.restore(sess2, TRAIN_FILE_NUMBER)
    print("Model restored.")

    print("Start recognize")
    image_reader = image_io.ImgReader()

    # predict
    counter1 = 0
    counter2 = 0
    record1 = 0
    record2 = 0
    txt = ""
    while True:
        _, image = image_reader.read(False)
        image_print = cv2.putText(image,txt,(0,20),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),thickness=2)
        cv2.imshow("Camera",image_print)

        image_resized = np.reshape(cv2.resize(image, (299, 299)), (1, 299, 299, 3))
        image_resized = [tool.__contrast_img(image_resized[0], 1.0, 80)]
        cv2.imshow("layer input", image_resized[0])
        cv2.waitKey(1)
        predict_logits_1 = sess1.run(softmaxed1, feed_dict={images1: image_resized})
        predict_logits_2 = sess2.run(softmaxed2, feed_dict={images2: image_resized})

        # suit
        # print(predict_logits_1,predict_logits_2)
        if max(predict_logits_1[0] > 0.9):
            counter1 += 1
            predict_num1 = np.argmax(predict_logits_1)
            predict_num2 = np.argmax(predict_logits_2)
            if counter1 > 3 and record1 == predict_num1:
                # number
                if max(predict_logits_2[0] > 0.3):
                    counter2 += 1
                    if counter2 > 3 and record2 == predict_num2:
                        txt = str(STYLES[int(predict_num1)]+str(predict_num2.item() + 1))
                        print(txt)
                    else:
                        txt = "N/A"
                        print(txt)
                    record2 = predict_num2
                else:
                    counter2 = 0
                    txt = "N/A"
                    print(txt)
            else:
                txt = "N/A"
                print(txt)
            record1 = predict_num1
        else:
            counter1 = 0
            txt = "N/A"
            print(txt)
        pass


if __name__ == '__main__':
    main()
