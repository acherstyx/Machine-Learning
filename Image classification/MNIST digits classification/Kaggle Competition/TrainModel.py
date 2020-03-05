import numpy as np
import tensorflow as tf

import NeuralNetwork as net
import ReadKaggleDataset as MNIST

CHECK_FREQUNCY=1000
ACCURACY_TEST_DATA_UPDATE_FREQUENCY=1

TRAIN_STEPS=100000

BATCH_SIZE=1
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=400

LEARNING_RATE_BASE=0.001*1e-1
LEARNING_RATE_DECAY_STEP=500
LEARNING_RATE_DECAY_RATE=0.99

REGULARIZATION_RATE=0.001
CNN_KEEP_PROB=0.6
IMAGE_SIZE=[28,28,1]

class MNIST_dataset():
    MNIST=None
    def __init__(self,rootpath):
        self.MNIST=MNIST.MNIST_Dataset(rootpath)
    def nextbatch(self,batchsize=1):
        image,label=self.MNIST.nextbatch(batchsize)
        image=np.reshape(image,[-1,28,28,1])
        return {x:image,y_:label,keep_prob_cnn:CNN_KEEP_PROB}
    def testbatch(self,batchsize=TEST_BATCH_SIZE):
        image,label=self.MNIST.testbatch(batchsize)
        image=np.reshape(image,[-1,28,28,1])
        return {x:image,y_:label,keep_prob_cnn:1}

with tf.variable_scope("Input"):
    x=tf.placeholder(tf.float32,shape=[None,]+IMAGE_SIZE,name="image")
    y_=tf.placeholder(tf.int64)
    keep_prob_cnn=tf.placeholder(tf.float32)
    GLOBAL_STEP=tf.Variable(0,trainable=False)
    LEARNING_RATE_BASE=tf.constant(LEARNING_RATE_BASE)

with tf.variable_scope("Network"):
    CNN_LAYER=[
        ["conv",[4,4,1,16],[1,1,1,1],0.1,"SAME",True],
        ["dropout",keep_prob_cnn],
        ["conv",[4,4,16,16],[1,1,1,1],0.1,"SAME",True],
        ["dropout",keep_prob_cnn],
        ["maxpool",[1,3,3,1],[1,2,2,1]],
        ["conv",[5,5,16,32],[1,1,1,1],0.1,"SAME",True],
        ["conv",[5,5,32,32],[1,1,1,1],0.1,"SAME",True],
        ["dropout",keep_prob_cnn],
        ["avgpool",[1,3,3,1],[1,2,2,1]],
        ["conv",[5,5,32,64],[1,1,1,1],0.1,"SAME",True],
        ["conv",[5,5,64,64],[1,1,1,1],0.1,"SAME",True],
        ["dropout",keep_prob_cnn],
        ["avgpool",[1,3,3,1],[1,2,2,1]],
    ]
    lineshape_result,nodes_num=net.CNN_Interface(x,CNN_LAYER,output_each_layer=True)
    DNN_LAYERS=[nodes_num,256,10]
    y=net.DNN_Interface(lineshape_result,DNN_LAYERS)
    y_maxed=tf.argmax(y,1)
    sampel=net.Sample_Output(y,y_,one_hot=False)

with tf.variable_scope("TrainModel"):
    loss,cross_entropy,regularization=net.Softmax_Cross_Entropy_With_Regularization(y_,y,REGULARIZATION_RATE,True,["fc_weight",])
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,GLOBAL_STEP,LEARNING_RATE_DECAY_STEP,LEARNING_RATE_DECAY_RATE)
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,GLOBAL_STEP)

    accuracy=net.Calc_Accuratcy(y,y_,one_hot=False)
    tf.summary.scalar("Accuracy",accuracy)
    saver=tf.train.Saver()

with tf.Session() as sess:
    dataset=MNIST_dataset("./.dataset/")
    sess.run(tf.global_variables_initializer())
    #写计算图
    summaries = tf.summary.merge_all()
    writer=tf.summary.FileWriter("./.log",tf.get_default_graph())
    print("==================SET TIMER==================")
    timer=net.TrainTimer()
    timer.getinput()
    print("====================LOAD=====================")
    #初始化
    reply_load=input('Load model?(y/n): ')
    if reply_load=='y':
        saver.restore(sess,'./.model/model.ckpt')
    print("=================INPUT DATA==================")
    net.bar("Result")
    
    #lr_manual=tf.assign(LEARNING_RATE_BASE,1e-05)
    #sess.run(lr_manual)

    accuracy_test_count=0
    try:
        for i in range(TRAIN_STEPS): 
            i=i+200000
            #计时器
            if timer.check==1:
                break
            #训练a
            feed_dict_train=dataset.nextbatch(BATCH_SIZE)
            sess.run(train_step,feed_dict=feed_dict_train)
            #测试
            if i%CHECK_FREQUNCY==0:
                if accuracy_test_count==0:
                    accuracy_test_dict=dataset.testbatch(TEST_BATCH_SIZE)
                    accuracy_test_count=ACCURACY_TEST_DATA_UPDATE_FREQUENCY
                accuracy_test_count-=1
                print("Training step:{i:09d} Accuracy:{acc:0.3f} \nloss:{l} learning rate:{lr}\n".format(
                    i=i,\
                    acc=sess.run(accuracy,feed_dict=accuracy_test_dict)*100,\
                    l=sess.run(loss,feed_dict=accuracy_test_dict),\
                    lr=sess.run(learning_rate)),\
                    sess.run(sampel[0],feed_dict=accuracy_test_dict),\
                    sess.run(sampel[1],feed_dict=accuracy_test_dict))
                summ = sess.run(summaries, feed_dict=accuracy_test_dict)
                writer.add_summary(summ, global_step=i)
                net.Matedata_Writer(writer,accuracy_test_dict,train_step,i)
    except KeyboardInterrupt:
        pass
    writer.close()

    net.bar("Generate upload file")
    reply=input("Continue?(y/n): ")
    if reply=="y":
        net.bar("Generating upload file")
        with open("upload_output.csv","w") as f:
            f.write("ImageId,Label\n")
            for index,image in enumerate(MNIST.exam_data("./.dataset/")):        
                image={x:np.reshape(image,[1,28,28,1]),keep_prob_cnn:1}
                f.write("{index},{prediction}\n".format(index=index+1,prediction=sess.run(y_maxed,feed_dict=image)[0]))
            f.close()

    net.bar()
    reply=input('Save?(y/n): ')
    if reply=='y':
        saver.save(sess,'./.model/model.ckpt')
    net.bar()