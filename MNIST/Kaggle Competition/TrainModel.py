import numpy as np
import tensorflow as tf

import NeuralNetwork as net
import ReadKaggleDataset as MNIST

CHECK_FREQUNCY=500
ACCURACY_TEST_DATA_UPDATE_FREQUENCY=10
BATCH_SIZE=1
TRAIN_STEPS=100000
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=5000
REGULARIZATION_RATE=0.001
CNN_KEEP_PROB=0.6
IMAGE_SIZE=[28,28,1]

def trainbatch(batchsize=1):
    image,label=a.nextbatch(batchsize)
    image=np.reshape(image,[-1,28,28,1])
    return {x:image,y_:label,keep_prob_cnn:CNN_KEEP_PROB}

def testbatch(batchsize=TEST_BATCH_SIZE):
    image,label=a.testbatch(batchsize)
    image=np.reshape(image,[-1,28,28,1])
    return {x:image,y_:label,keep_prob_cnn:1}

with tf.variable_scope("Input"):
    x=tf.placeholder(tf.float32,shape=[None,]+IMAGE_SIZE,name="image")
    y_=tf.placeholder(tf.int64)
    keep_prob_cnn=tf.placeholder(tf.float32)
    GLOBAL_STEP=tf.Variable(0,trainable=False)
    LEARNING_RATE_BASE=tf.constant(0.001)
    LEARNING_RATE_DECAY_STEP=500
    LEARNING_RATE_DECAY_RATE=0.99
with tf.variable_scope("Network"):
    CNN_LAYER=[
        ["conv",[4,4,1,16],[1,1,1,1],0.1,"SAME",True],
        ["dropout",keep_prob_cnn],
        ["maxpool",[1,3,3,1],[1,2,2,1]],
        ["conv",[5,5,16,32],[1,1,1,1],0.1,"SAME",True],
        ["dropout",keep_prob_cnn],
        ["avgpool",[1,3,3,1],[1,2,2,1]],
        ["conv",[5,5,32,64],[1,1,1,1],0.1,"SAME",True],
        ["dropout",keep_prob_cnn],
        ["avgpool",[1,3,3,1],[1,2,2,1]],
    ]
    lineshape_result,nodes_num=net.CNN_Interface(x,CNN_LAYER,output_each_layer=True)
    DNN_LAYERS=[nodes_num,256,10]
    y=net.DNN_Interface(lineshape_result,DNN_LAYERS)
    y_maxed=tf.argmax(y,1)

with tf.variable_scope("TrainModel"):
    loss,cross_entropy,regularization=net.Softmax_Cross_Entropy_With_Regularization(y_,y,REGULARIZATION_RATE,True,["fc_weight",])
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,GLOBAL_STEP,LEARNING_RATE_DECAY_STEP,LEARNING_RATE_DECAY_RATE)
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,GLOBAL_STEP)

    #正确率
    crrect_prediction=tf.equal(tf.argmax(y,1),y_)
    accuracy=tf.reduce_mean(tf.cast(crrect_prediction,tf.float64))
    saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #写文件
    summaries = tf.summary.merge_all()
    writer=tf.summary.FileWriter("./.log",tf.get_default_graph())
    print("==================SET TIMER==================")
    timer=net.TrainTimer()
    timer.getinput()
    print("====================LOAD=====================")
    a=MNIST.MNIST_Dataset("./.dataset/")
    #初始化
    reply_load=input('Load model?(y/n): ')
    if reply_load=='y':
        saver.restore(sess,'./.model/model.ckpt')
    print("=================INPUT DATA==================")
    
    net.bar("Result")
    accuracy_test_count=0
    #lr_manual=tf.assign(LEARNING_RATE_BASE,1e-05)
    #sess.run(lr_manual)
    for i in range(TRAIN_STEPS): 
        try:
            #i=i+TURN*TRAINING_STEPS
            i=i+1000000
            #计时器
            if timer.check==1:
                break
            #训练a
            feed_dict_train=trainbatch(BATCH_SIZE)
            sess.run(train_step,feed_dict=feed_dict_train)
            #测试
            if i%CHECK_FREQUNCY==0:
                if accuracy_test_count==0:
                    accuracy_test_dict=testbatch(TEST_BATCH_SIZE)
                    accuracy_test_count=ACCURACY_TEST_DATA_UPDATE_FREQUENCY
                accuracy_test_count-=1
                print("Training step:",i,\
                        'Accuracy:',sess.run(accuracy,feed_dict=accuracy_test_dict),\
                        sess.run(loss,feed_dict=accuracy_test_dict),\
                        sess.run(y_maxed,feed_dict=accuracy_test_dict)[5],\
                        sess.run(y_,feed_dict=accuracy_test_dict)[5],\
                        sess.run(learning_rate))
                summ = sess.run(summaries, feed_dict=accuracy_test_dict)
                writer.add_summary(summ, global_step=i)
                #net.Matedata_Writer(writer,accuracy_test_dict,train_step,i)
        except KeyboardInterrupt:
            net.bar("Pause")
            new_lr=input("New learning rate base(-1 to quit): ")
            try:
                if float(new_lr)>0:
                    try:
                        set_init=input("Refresh Model?(y/n): ")
                        if set_init=="y":
                            sess.run(tf.global_variables_initializer())
                        change_lr=tf.assign(LEARNING_RATE_BASE,float(new_lr))
                        sess.run(change_lr)
                    except Exception as e:
                        print("[!]Failed to change learning rate!")
                        print(e)
                    net.bar("Continue")
                else:
                    break
            except:
                pass
    writer.close()
    print("===================================")
    reply=input('Save?(y/n): ')
    if reply=='y':
        saver.save(sess,'./.model/model.ckpt')
    print("===================================")