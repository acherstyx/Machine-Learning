import tensorflow as tf
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from CIFAR10_Dataloader import CIFAR10

DATASET_DIR_PATH = "./.dataset"
IMAGE_SHAPE = [32, 32, 3]

with tf.variable_scope("Data_in"):
    image = tf.placeholder(tf.float32, shape=[None] + IMAGE_SHAPE, name="image")
    label_ = tf.placeholder(tf.int32, shape=[None, ], name="label_")

with tf.variable_scope("ResNet"):
    # Layer 1 size:3(in)->32
    layer1_filter = tf.get_variable(name="layer1_filter",
                                    shape=[3, 3, 3, 32],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
    layer1_bias = tf.get_variable(name="Layer1_bias",
                                  shape=[32, ],
                                  initializer=tf.constant_initializer(0.0))
    layer1_init = tf.nn.conv2d(input=image,
                               filter=layer1_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer1_with_bias = tf.nn.bias_add(layer1_init, layer1_bias)
    layer1_active = tf.nn.relu(layer1_with_bias)

    # Layer 2 size:same
    layer2_filter = tf.get_variable(name="layer2_filter",
                                    shape=[3, 3, 32, 32],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
    layer2_bias = tf.get_variable(name="Layer2_bias",
                                  shape=[32, ],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
    layer2_init = tf.nn.conv2d(input=layer1_active,
                               filter=layer2_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer2_with_bias = tf.nn.bias_add(layer2_init, layer2_bias)
    layer2_active = tf.nn.relu(layer2_with_bias)

    # layer 3 size:same
    layer3_filter = tf.get_variable(name="layer3_filter",
                                    shape=[3, 3, 32, 32],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
    layer3_bias = tf.get_variable(name="Layer3_bias",
                                  shape=[32, ],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
    layer3_init = tf.nn.conv2d(input=layer2_active,
                               filter=layer3_filter,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
    layer3_with_bias = tf.nn.bias_add(layer3_init, layer3_bias)
    layer3_active = tf.nn.relu(layer3_with_bias)

    line_nodes = layer3_active.shape[1] * layer3_active.shape[2] * layer3_active.shape[3]
    print("Nodes of CNN output: ", line_nodes)
    line_reshaped = tf.reshape(tensor=layer3_active,
                               shape=[-1, line_nodes],
                               name="line_reshaped")

with tf.variable_scope("DNN"):
    # layer 1 size:line_nodes->1024
    fc_layer1_weight = tf.get_variable(name="fc_layer1_weight",
                                       shape=[line_nodes, 1024],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
    fc_layer1_init = tf.matmul(line_reshaped, fc_layer1_weight)
    fc_layer1_active = tf.nn.relu(fc_layer1_init)
    # layer 2 size:1024->10(out)
    fc_layer2_weight = tf.get_variable(name="fc_layer2_weight",
                                       shape=[1024, 10],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
    fc_layer2_init = tf.matmul(fc_layer1_active, fc_layer2_weight)
    fc_layer2_active = tf.nn.relu(fc_layer2_init)

with tf.variable_scope("Train_model"):
    LEARNING_RATE_BASE = 0.001
    LEARNING_RATE_DECAY_STEP = 0
    LEARNING_RATE_DECAY_RATE = 0
    global_step = tf.Variable(initial_value=0,
                              trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,
                                               global_step=global_step,
                                               decay_steps=LEARNING_RATE_DECAY_STEP,
                                               decay_rate=LEARNING_RATE_DECAY_RATE,
                                               name="Learning_rate")
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_,
                                                                                  logits=fc_layer2_active))
    loss = cross_entropy
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss,
                                                                                         global_step=global_step)

with tf.variable_scope("Analyze_and_save"):
    saver = tf.train.Saver()

# 数据集读取
print("Loading data set ... ", end='')
CIFAR10_DATA_SET = CIFAR10(DATASET_DIR_PATH)
train_data_loader, test_data_loader = CIFAR10_DATA_SET.Create_dataloader(100)
print("Finished")

with tf.Session() as sess:
    # initialize
    sess.run(tf.global_variables_initializer())
    # summary
    summaries = tf.summary.merge_all()
    # write graph
    writer = tf.summary.FileWriter("./.log/", tf.get_default_graph())
