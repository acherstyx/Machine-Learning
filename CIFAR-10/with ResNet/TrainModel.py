import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import mxnet.ndarray as nd

import Tools as tool

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

TRAIN_EPOCH = 20
EPOCH_OFFSET = 20
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY_STEP = 2500
LEARNING_RATE_DECAY_RATE = 0.99
REGULARIZATION_RATE = 1e-3
GRADE_LIMIT = 20
DROPOUT_RATE = 0.3
CHECK_FREQUENCY = 2500

DATASET_DIR_PATH = "./.dataset"
IMAGE_SHAPE = [32, 32, 3]


def loss_on_test(data_loader):
    test_loss = []
    for images, labels in data_loader:
        test_feed_dict = {image: images.asnumpy(), label_: labels.asnumpy()}
        test_loss.append(sess.run(loss, feed_dict=test_feed_dict))
        break
    return np.mean(test_loss)


with tf.variable_scope("Data_in"):
    image = tf.placeholder(tf.float32, shape=[None] + IMAGE_SHAPE, name="image")
    label_ = tf.placeholder(tf.int32, shape=[None, ], name="label_")
    rate = tf.placeholder_with_default(0.0, shape=())

with tf.variable_scope("ResNet"):
    hidden_layer = []
    # Layer 1 size:3(in)->32
    layer1_filter = tf.get_variable(name="layer1_filter",
                                    shape=[3, 3, 3, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer1_init = tf.nn.conv2d(input=image,
                               filter=layer1_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer1_active = tf.nn.relu(layer1_init)
    hidden_layer.append(layer1_active)

    # Layer 2 size:same
    layer2_filter = tf.get_variable(name="layer2_filter",
                                    shape=[3, 3, 32, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer2_init = tf.nn.conv2d(input=layer1_active,
                               filter=layer2_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer2_active = tf.nn.relu(layer2_init)
    hidden_layer.append(layer2_active)

    # layer 3 size:same
    layer3_filter = tf.get_variable(name="layer3_filter",
                                    shape=[3, 3, 32, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer3_init = tf.nn.conv2d(input=layer2_active,
                               filter=layer3_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer3_active = tf.nn.relu(layer3_init)
    hidden_layer.append(layer3_active)

    # layer 4 size:32->48
    layer4_filter = tf.get_variable(name="layer4_filter",
                                    shape=[3, 3, 32, 48],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer4_init = tf.nn.conv2d(input=layer3_active,
                               filter=layer4_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer4_active = tf.nn.relu(layer4_init)
    hidden_layer.append(layer4_active)

    # layer 5 size:48->48
    layer5_filter = tf.get_variable(name="layer5_filter",
                                    shape=[3, 3, 48, 48],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer5_init = tf.nn.conv2d(input=layer4_active,
                               filter=layer5_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer5_active = tf.nn.relu(layer5_init)
    hidden_layer.append(layer5_active)

    # max pool 4 2
    pool1 = tf.nn.max_pool(layer5_active,
                           ksize=[1, 4, 4, 1],
                           strides=[1, 2, 2, 1],
                           padding="SAME")

    # dropout rate=0.4
    dropout1 = tf.nn.dropout(pool1, rate=rate)

    # layer 6 size:48->80
    layer6_filter = tf.get_variable(name="layer6_filter",
                                    shape=[3, 3, 48, 80],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer6_init = tf.nn.conv2d(input=dropout1,
                               filter=layer6_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer6_active = tf.nn.relu(layer6_init)
    hidden_layer.append(layer6_active)

    # layer 7 size:80->80
    layer7_filter = tf.get_variable(name="layer7_filter",
                                    shape=[3, 3, 80, 80],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer7_init = tf.nn.conv2d(input=layer6_active,
                               filter=layer7_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer7_active = tf.nn.relu(layer7_init)
    hidden_layer.append(layer7_active)

    # layer 8 size:80->80
    layer8_filter = tf.get_variable(name="layer8_filter",
                                    shape=[3, 3, 80, 80],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer8_init = tf.nn.conv2d(input=layer7_active,
                               filter=layer8_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer8_active = tf.nn.relu(layer8_init)
    hidden_layer.append(layer8_active)

    # layer 9 size:80->80
    layer9_filter = tf.get_variable(name="layer9_filter",
                                    shape=[3, 3, 80, 80],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer9_init = tf.nn.conv2d(input=layer8_active,
                               filter=layer9_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer9_active = tf.nn.relu(layer9_init)
    hidden_layer.append(layer9_active)

    # layer 10 size:80->80
    layer10_filter = tf.get_variable(name="layer10_filter",
                                     shape=[3, 3, 80, 80],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer10_init = tf.nn.conv2d(input=layer9_active,
                                filter=layer10_filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
    layer10_active = tf.nn.relu(layer10_init)
    hidden_layer.append(layer10_active)

    # avg pool 4 2
    pool2 = tf.nn.avg_pool(layer10_active,
                           ksize=[1, 4, 4, 1],
                           strides=[1, 2, 2, 1],
                           padding="SAME")

    # dropout rate=0.4
    dropout2 = tf.nn.dropout(pool2, rate=rate)

    # layer 11 size:80->128
    layer11_filter = tf.get_variable(name="layer11_filter",
                                     shape=[3, 3, 80, 128],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer11_init = tf.nn.conv2d(input=pool2,
                                filter=layer11_filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
    layer11_active = tf.nn.relu(layer11_init)
    hidden_layer.append(layer11_active)

    # layer 12 size:128->128
    layer12_filter = tf.get_variable(name="layer12_filter",
                                     shape=[3, 3, 128, 128],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer12_init = tf.nn.conv2d(input=layer11_active,
                                filter=layer12_filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
    layer12_active = tf.nn.relu(layer12_init)
    hidden_layer.append(layer12_active)

    # layer 13 size:128->128
    layer13_filter = tf.get_variable(name="layer13_filter",
                                     shape=[3, 3, 128, 128],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer13_init = tf.nn.conv2d(input=layer12_active,
                                filter=layer13_filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
    layer13_active = tf.nn.relu(layer13_init)
    hidden_layer.append(layer13_active)

    # layer 14 size:128->128
    layer14_filter = tf.get_variable(name="layer14_filter",
                                     shape=[3, 3, 128, 128],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer14_init = tf.nn.conv2d(input=layer13_active,
                                filter=layer14_filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
    layer14_active = tf.nn.relu(layer14_init)
    hidden_layer.append(layer14_active)

    # layer 15 size:128->128
    layer15_filter = tf.get_variable(name="layer15_filter",
                                     shape=[3, 3, 128, 128],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer15_init = tf.nn.conv2d(input=layer14_active,
                                filter=layer15_filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
    layer15_active = tf.nn.relu(layer15_init)
    hidden_layer.append(layer15_active)

    # avg pool 8 4
    pool3 = tf.nn.avg_pool(layer15_active,
                           ksize=[1, 8, 8, 1],
                           strides=[1, 4, 4, 1],
                           padding="SAME")

    # dropout rate=0.4
    dropout3 = tf.nn.dropout(pool3, rate=rate)

    line_nodes = dropout3.shape[1] * dropout3.shape[2] * dropout3.shape[3]
    line_reshaped = tf.reshape(tensor=dropout3,
                               shape=[-1, line_nodes],
                               name="line_reshaped")

with tf.variable_scope("DNN"):
    # layer 1 size:line_nodes->1024
    fc_layer1_weight = tf.get_variable(name="fc_layer1_weight",
                                       shape=[line_nodes, 512],
                                       initializer=tf.random_normal_initializer(0.1))
    fc_layer1_init = tf.matmul(line_reshaped, fc_layer1_weight)
    fc_layer1_active = tf.nn.relu(fc_layer1_init)
    # layer 2 size:1024->10(out)
    fc_layer2_weight = tf.get_variable(name="fc_layer2_weight",
                                       shape=[512, 10],
                                       initializer=tf.random_normal_initializer(0.1))
    fc_layer2_init = tf.matmul(fc_layer1_active, fc_layer2_weight)
    fc_layer2_active = tf.nn.relu(fc_layer2_init)

with tf.variable_scope("Train_model"):
    global_step = tf.Variable(initial_value=0,
                              trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,
                                               global_step=global_step,
                                               decay_steps=LEARNING_RATE_DECAY_STEP,
                                               decay_rate=LEARNING_RATE_DECAY_RATE,
                                               name="Learning_rate")
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_,
                                                                                  logits=fc_layer2_active))
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(fc_layer1_weight) + regularizer(fc_layer2_weight)
    loss = cross_entropy + regularization
    # optimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads, global_norm = tf.clip_by_global_norm(grads, GRADE_LIMIT)
    train_step = optimizer.apply_gradients(zip(grads, variables),global_step)

with tf.variable_scope("Analyze_and_save"):
    saver = tf.train.Saver()
    # tensorboard image
    tf.summary.image("Image", image[:, :, :, :3], max_outputs=1)
    # histogram filter and layer output
    histogram_filter = [layer1_filter, layer2_filter, layer3_filter, layer4_filter, layer5_filter,
                        layer6_filter, layer7_filter, layer8_filter, layer9_filter, layer10_filter,
                        layer11_filter, layer12_filter, layer13_filter, layer14_filter, layer15_filter]
    hidden_layer = [layer1_active, layer2_active, layer3_active, layer4_active, layer5_active,
                    layer6_active, layer7_active, layer8_active, layer9_active, layer10_active,
                    layer11_active, layer12_active, layer13_active, layer14_active, layer15_active]
    for i in range(15):
        tf.add_to_collection("summary_image",
                             tf.summary.image("Layer image/Layer{ord}".format(ord=i + 1),
                                              hidden_layer[i][:, :, :, :3],
                                              max_outputs=1))
        tf.add_to_collection("summary_histogram",
                             tf.summary.histogram("filter/conv{layer_order}_filter".format(layer_order=i),
                                                  histogram_filter[i]))

with tf.variable_scope("Accuracy"):
    prediction = tf.cast(tf.argmax(fc_layer2_active, axis=1), tf.int32, name="Prediction")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, label_), tf.float32), name="Accuracy")
    summary_accuracy_test = tf.summary.scalar("Accuracy_test", accuracy)
    summary_accuracy_train = tf.summary.scalar("Accuracy_train", accuracy)

print(">>> Nodes of cnn output layer: ", line_nodes)

# timer init
timer = tool.TrainTimer()

with tf.Session() as sess:
    # initialize
    sess.run(tf.global_variables_initializer())
    reply_load = input('>>> Load model?(y/n): ')
    if reply_load == 'y':
        saver.restore(sess, './.save/model.ckpt')
        print("Model restored.")
    # load data set
    print(">>> Loading data set ... ", end='')
    timer.reset()
    train_data_loader, test_data_loader = tool.Create_dataloader(path=DATASET_DIR_PATH,
                                                                 train_batch_size=4,
                                                                 test_batch_size=100,
                                                                 dataAug=True)
    # get train sample
    train_data_loader2, test_data_loader2 = tool.Create_dataloader(path=DATASET_DIR_PATH,
                                                                   train_batch_size=2000,
                                                                   test_batch_size=2000,
                                                                   shuffle=True,
                                                                   dataAug=False)
    for train_data_image, train_data_label in train_data_loader2:
        train_feed_dict_sample = {image: train_data_image.asnumpy(), label_: train_data_label.asnumpy()}
        break
    for test_data_image, test_data_label in test_data_loader2:
        test_feed_dict_sample = {image: test_data_image.asnumpy(), label_: test_data_label.asnumpy()}
        break
    print("Finished ", timer.read())
    # summary
    merge_summary_info = tf.summary.merge([tf.get_collection("summary_image"),
                                           tf.get_collection("summary_histogram")])
    merge_summary_accuracy_test = tf.summary.merge([summary_accuracy_test])
    merge_summary_accuracy_train = tf.summary.merge([summary_accuracy_train])
    # write graph
    writer = tf.summary.FileWriter("./.log/with ResNet", tf.get_default_graph())
    writer_train = tf.summary.FileWriter("./.log/with ResNet/train")
    writer_test = tf.summary.FileWriter("./.log/with ResNet/test")

    counter = 0
    timer.reset()
    for i in range(TRAIN_EPOCH):
        # train model
        i += EPOCH_OFFSET
        for train_data_image, train_data_label in train_data_loader:
            if counter % CHECK_FREQUENCY == 0:
                print(">>> After {batch} batch: ".format(batch=counter))
                print("Learning rate: {lr}".format(lr=sess.run(learning_rate)))
                print("Loss on train: {train_loss:.5f}, Loss on test: {test_loss:.5f}".format(
                    train_loss=sess.run(loss, feed_dict=train_feed_dict_sample),
                    test_loss=sess.run(loss, feed_dict=test_feed_dict_sample)))
                print("Cross entropy: {ce:.4f} Regularization: {reg:.4f}".format(
                    ce=sess.run(cross_entropy, feed_dict=train_feed_dict_sample),
                    reg=sess.run(regularization, feed_dict=train_feed_dict_sample)))
                print("Accuracy on train: {acc:.2f}% Accuracy on test: {acc_test:.2f}%".format(
                    acc=sess.run(accuracy, feed_dict=train_feed_dict_sample) * 100,
                    acc_test=sess.run(accuracy, feed_dict=test_feed_dict_sample) * 100))
            counter += 1

            # generate feed dict
            train_feed_dict = {image: train_data_image.asnumpy(), label_: train_data_label.asnumpy(), rate: DROPOUT_RATE}
            # run
            sess.run(train_step, feed_dict=train_feed_dict)
        # write summary
        writer.add_summary(sess.run(merge_summary_info, feed_dict=test_feed_dict_sample), global_step=i)
        writer_test.add_summary(sess.run(merge_summary_accuracy_test, feed_dict=test_feed_dict_sample), global_step=i)
        writer_train.add_summary(sess.run(merge_summary_accuracy_train, feed_dict=train_feed_dict_sample), global_step=i)
    # save model
    saver.save(sess, './.save/model.ckpt')
