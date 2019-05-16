import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
import math

def CNN_Interface(data,Layers,lineshape=True,bias=True,order=0,const_init=0.0,active_function=tf.nn.relu,output_each_layer=False):
    '''
    传入初始的数据，以及按格式创建的各个层的参数，就可以完成卷积层和池化层的创建
    Layers示例:
    CNN_LAYERS=[
        ["conv",[3,3,3,32],[1,1,1,1],0.01,"SAME",True],
        ["pool",[1,8,8,1],[1,8,8,1]],
        ]
    卷积层：卷积层标识|过滤器大小|过滤器移动步长|stddev|边界处理方式|是否使用偏置项|
    池化层：池化层标识|池化层大小|池化层移动步长|
    其他参数：
    order 当前是第几次条用本函数，多以调用需要更改order以防止变量名重复
    stddev 卷积层初始化参数
    const_init 偏置项初始化值
    '''
    hidden_layer=data
    with tf.variable_scope("Convolutional_Neural_Networks_{order}".format(order=order)):
        if output_each_layer==True:
            tf.summary.image("CNN_Input",hidden_layer,max_outputs=1)
        for layer_order,layer in enumerate(Layers):
            if layer[0]=="conv":
                #过滤器
                cnn_filter=tf.get_variable("conv{layer_order}_filter".format(layer_order=layer_order),shape=layer[1],initializer=tf.random_normal_initializer(stddev=layer[3]))
                tf.summary.histogram("conv{layer_order}_filter".format(layer_order=layer_order), cnn_filter)
                temp_layer=tf.nn.conv2d(hidden_layer,cnn_filter,layer[2],padding="SAME")
                if layer[5]==True:
                    #偏置项
                    cnn_bias=tf.get_variable("conv{layer_order}_biase".format(layer_order=layer_order),shape=[layer[1][3]],initializer=tf.constant_initializer(const_init))
                    tf.summary.histogram("conv{layer_order}_bias".format(layer_order=layer_order), cnn_bias)
                    temp_bias=tf.nn.bias_add(temp_layer,cnn_bias)
                    hidden_layer=active_function(temp_bias)
                else:
                    hidden_layer=active_function(temp_layer)
            elif layer[0]=="maxpool":
                temp_layer=tf.nn.max_pool(hidden_layer,ksize=layer[1],strides=layer[2],padding="SAME")
                hidden_layer=active_function(temp_layer)
            elif layer[0]=="avgpool":
                temp_layer=tf.nn.avg_pool(hidden_layer,ksize=layer[1],strides=layer[2],padding="SAME")
                hidden_layer=active_function(temp_layer)
            elif layer[0]=="dropout":
                temp_layer=tf.nn.dropout(hidden_layer,layer[1])
                hidden_layer=temp_layer
            else:
                raise(ValueError)
            
            if output_each_layer==True:
                tf.summary.image("CNN_Layer_{ord}".format(ord=layer_order),hidden_layer[:,:,:,:1],max_outputs=1)
        
        if lineshape==True:
            pool_shape=hidden_layer.get_shape().as_list()
            nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
            reshaped=tf.reshape(hidden_layer,[-1,nodes],name='reshaped')
            return reshaped,nodes
        else:
            return hidden_layer

def DNN_Interface(data,Layer,order=0,keep_prob_layer=1,keep_prob_image=1,stddev=0.1,const_init=0.0,active_function=tf.nn.relu):
    '''
    自动创建全连接层
    Layer示例：
    [输入图片大小,隐藏层大小,...,输出层大小]
    其他参数：
    order：为防止重复预留的区分数字
    keep_prob_layer,keep_prob_image：dropout操作时图像和隐藏层保留的比例，默认为1
    '''
    with tf.variable_scope("Fully_Lincked_Networks_{order}".format(order=order)):
        #建立权重
        weight=[]
        bias=[]
        for i in range(len(Layer)-1):
            weight.append(tf.get_variable("fc{layer_order}_weight".format(layer_order=i),shape=[Layer[i],Layer[i+1]],initializer=tf.random_normal_initializer(stddev=stddev)))
            #bias.append(tf.get_variable("fc{layer_order}_bias".format(layer_order=i),shape=[Layer[i+1]],initializer=tf.constant_initializer(const_init)))
            #加入到集合以供计算正则化
            tf.add_to_collection("fc_weight",weight[i])
            #添加到Tensorboard
            tf.summary.histogram("fc{layer_order}_weight".format(layer_order=i), weight[i])
            #tf.summary.histogram("fc{layer_order}_bias".format(layer_order=i), bias[i])
        #计算
        data=tf.nn.dropout(data,keep_prob=keep_prob_image)
        for i in range(len(Layer)-1):
            if i==0:
                result=active_function(tf.matmul(data,weight[i]))
            else:
                result_droped=tf.nn.dropout(result,keep_prob=keep_prob_layer)
                if not i==(len(Layer)-2):
                    result=active_function(tf.matmul(result_droped,weight[i]))
                else:
                    result=tf.matmul(result_droped,weight[i])
    return result

def Matedata_Writer(writer,feed_dict,train_op,step,checkfreq=1,sess="default",name="step{global_step}"):
    '''
    用于写入运行时的资源占用等源数据
    需要传入：已经定义好的Writer，测试运行时间用的feed_dict，训练操作train_op，当前训练轮数
    检查频率标志了多少次step时进行一次记录
    可利用name自定义当前次训练的命名
    '''
    with tf.variable_scope("Matedata_Writer"):
        if step%checkfreq != 0:
            return
        if sess=="default":
            sess=tf.get_default_session()
        summary = tf.summary.merge_all()
        runop=tf.RunOptions(trace_level=3)
        runme=tf.RunMetadata()
        summary_ , _ =sess.run([summary,train_op],feed_dict=feed_dict,run_metadata=runme,options=runop)
        writer.add_run_metadata(runme,name.format(global_step=step))
        writer.add_summary(summary_,step)

def Softmax_Cross_Entropy_With_Regularization(label,logits,regularization_rate=0.0,single_index=False,collections=None,):
    '''
    接受label和logit计算交叉熵
    在collections不为空的时候会同时计算正则化的损失，需要传入进行正则化计算的collenction
    collections格式为：
    [["collection_name_1"],
    ["collection_name_1"],
    ...]
    '''
    with tf.variable_scope("Loss"):
        if single_index==True:
            cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits))
        elif single_index==False:
            cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits))
        regu=tf.constant(0.0)
        regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
        if collections!=None:
            for collection in collections:
                for weight in tf.get_collection(collection):
                    regu+=regularizer(weight)
        return regu+cross_entropy,cross_entropy,regu

def Info_Printer(learning_rate_base=None,learning_rate_decay_rate=None,learning_rate_decay_step=None,regularization_rate=None,keep_prob_layer=None,keep_prob_image=None):
    if learning_rate_base!=None:
        print("Learning rate base:{lrb}".format(lrb=learning_rate_base))
    if learning_rate_decay_rate!=None:
        print("Learning rate decay rate:{lrdr}".format(lrdr=learning_rate_decay_rate))
    if learning_rate_decay_step!=None:
        print("Learning rate decay step:{lrds}".format(lrds=learning_rate_decay_step))
    if regularization_rate!=None:
        print("Regularization rate:{regu}".format(regu=regularization_rate))
    if keep_prob_layer!=None:
        print("Keep prob layer:{drop}".format(drop=keep_prob_layer))
    if keep_prob_image!=None:
        print("Keep prob image:{drop}".format(drop=keep_prob_image))

def Learning_Rate_Search(lr_tf_variable,train_op,aim_test_op,dataset,lr_base,lr_upper_bond=1,lr_raise_rate=2,train_step=5000,train_batch_size=5,test_batch_size=2000,show_graphe=True,print_data=False,sess="default",restore=False):
    with tf.variable_scope("Learning_Rate_Search"):
        if lr_base>lr_upper_bond:
            return
        if sess=="default":
            sess=tf.get_default_session()
        lr_plotline=[]
        lr_plotaxis=[]
        feed_dict_test=dataset.nextbatch(test_batch_size,type="Test")
        while(1):
            try:
                if restore==False:
                    sess.run(tf.global_variables_initializer())
                else:
                    saver=tf.train.Saver()
                    saver.restore(sess,restore)
                update_lr=tf.assign(lr_tf_variable,lr_base)
                sess.run(update_lr)

                for step in range(train_step):
                    feed_dict_train=dataset.nextbatch(train_batch_size)
                    sess.run(train_op,feed_dict=feed_dict_train)

                acc=sess.run(aim_test_op,feed_dict=feed_dict_test)
                
                lr_plotline.append(acc)
                if print_data==True:
                    print(lr_plotline[-1],sess.run(lr_tf_variable))
                lr_plotaxis.append(lr_base)
                if lr_base > lr_upper_bond:
                    break
                lr_base=lr_base*lr_raise_rate
            except KeyboardInterrupt:
                bar("New Test Case")
                interrupt_change=input("New lr base:")
                try:
                    lr_base=float(interrupt_change)
                except Exception as e:
                    print(e)
                    print("Failed")
                bar("New Test Case")
                
    if show_graphe==True:
        x=np.array(lr_plotaxis)
        x=np.log10(x)
        y=np.array(lr_plotline)
        plt.plot(x,y,'r--',label='Learning rate')
        plt.legend()
        plt.show()
    
    return lr_plotaxis[np.argmax(lr_plotline)]



class TrainTimer():
    """
    训练时计时用，setting设置时限
    在到达时限时，返回1，否则0
    若设置为-1.则无限制
    """
    limit=0
    start_time=0
    def __init__(self,timelimit=-1):
        self.limit=timelimit
    def setting(self,timelimit=-1):
        if timelimit!=-1:
            self.limit=timelimit
        self.start_time=time.time()
    def check(self):
        time_check=time.time()
        if (time_check-self.start_time) > self.limit | self.limit!=-1:
            return False
        else:
            return True
    def getinput(self):
        reply=input('Set time limit(s, unlimited: -1): ')
        try:
            self.limit=int(reply)
        except ValueError:
            self.limit=-1

def bar(string="",length=40,with_fram=False):
        strlength=len(string)
        if with_fram==True:
            print("="*length)
        print("="*int((length-strlength)/2)+string+"="*int(length-strlength-(length-strlength)/2))
        if with_fram==True:
            print("="*length)

if __name__ == "__main__":
    import random
    import numpy as np
    TEST_LAYERS=[
        ["conv",[5,5,3,32],[1,1,1,1],0.0001,"SAME",True],
        ["maxpool",[1,3,3,1],[1,2,2,1]],
        ["conv",[5,5,32,32],[1,1,1,1],0.01,"SAME",True],
        ["avgpool",[1,3,3,1],[1,2,2,1]],
        ["conv",[5,5,32,64],[1,1,1,1],0.01,"SAME",True],
        ["avgpool",[1,3,3,1],[1,2,2,1]],
    ]
    image=tf.Variable(tf.random_normal([1000,32,32,3],mean=1.0))
    result,index=CNN_Interface(image,TEST_LAYERS)
    FC_LAYER=[index,100,10]
    fcout=DNN_Interface(result,FC_LAYER)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        bar("Image Shape")
        print(image.shape)
        bar("CNN Network Output Shape")
        print(result.shape)
        bar("FC Network Output Shape")
        print(fcout.shape)
        bar("Network Output")
        print(sess.run(result))
        bar("Summary Writer Check")
        writer=tf.summary.FileWriter("./log/NeuralNetworkDebug",tf.get_default_graph())
        Matedata_Writer(writer,{},fcout,1)
    bar("Info Printer Check")
    Info_Printer(learning_rate_base=0.2)
    bar("Timer Check")
    timer=TrainTimer(2)
    timer.setting()
    while timer.check():
        pass
    bar("Timer Check Exit")
