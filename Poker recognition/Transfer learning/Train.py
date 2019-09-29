import tensorflow as tf
import tensorflow.contrib.slim as slim
import Poker_Dataloader as tool
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 加载模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 保存训练的地址
TRAIN_FILE = './.save/save_model'
# 已训练好的模型参数
CKPT_FILE = './.ckpt/inception_v3.ckpt'

# 不从模型中加载的参数
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数名称
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

LEARNING_RATE = 0.00005
TRAIN_EPOCH = 1
BATCH_SIZE = 4

NUM_CLASS = 4

reply_load = input('>>> Load model?(y/n): ')


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
    # 加载预处理好的数据。
    train_dataloader, test_dataloader = tool.Create_dataloader_color("./.dataset", BATCH_SIZE, 10, True, True)

    # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签。
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # 定义inception-v3模型。因为谷歌给出的只有模型参数取值，所以这里
    # 需要在这个代码中定义inception-v3的模型结构。虽然理论上需要区分训练和
    # 测试中使用到的模型，也就是说在测试时应该使用is_training=False，但是
    # 因为预先训练好的inception-v3模型中使用的batch normalization参数与
    # 新的数据会有出入，所以这里直接使用同一个模型来做测试。
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=NUM_CLASS, is_training=True)

    trainable_variables = get_trainable_variables()
    # 定义损失函数和训练过程。
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, NUM_CLASS), logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义加载Google训练好的Inception-v3模型的Saver。
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tuned_variables(),
        ignore_missing_vars=True)

    # 定义保存新模型的Saver。
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化没有加载进来的变量。
        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载谷歌已经训练好的模型。
        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)

        if reply_load == 'y':
            saver.restore(sess, TRAIN_FILE)
            print("Model restored.")

        counter = 0
        for i in range(TRAIN_EPOCH):
            for training_images, training_labels in train_dataloader:
                counter += 1
                if counter % 100 == 0:
                    test_counter = 0
                    validation_accuracy = []
                    for validation_images, validation_labels in test_dataloader:
                        test_counter += 1
                        validation_accuracy.append(sess.run(evaluation_step,
                                                            feed_dict={images: validation_images.asnumpy(),
                                                                       labels: validation_labels.asnumpy()}))
                        if test_counter > 20:
                            break
                    print('Step %d: Training loss is %.1f Validation accuracy = %.1f%%' % (
                        counter, loss, np.mean(validation_accuracy) * 100.0))
                    saver.save(sess, TRAIN_FILE)

                # train
                _, loss = sess.run([train_step, total_loss], feed_dict={
                    images: training_images.asnumpy(),
                    labels: training_labels.asnumpy()})
        # 在最后的测试数据上测试正确率。
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: validation_images, labels: validation_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()
