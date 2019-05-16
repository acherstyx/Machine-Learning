import numpy as np
import os
import random
import pandas

#数据集封装
class MNIST_Dataset:
    train={}
    test={}
    batch_mark=0
    def __init__(self,root):
        train_file=os.path.join(root,"train.csv")
        image=pandas.read_csv(train_file,skiprows=1,delimiter=',',usecols=list(range(28*28+1)[1:])).values
        label=np.reshape(pandas.read_csv(train_file,skiprows=1,delimiter=',',usecols=(0,),dtype=int).values,[-1])
        seed=random.randint(0,10000)
        np.random.seed(seed)
        np.random.shuffle(image)
        np.random.seed(seed)
        np.random.shuffle(label)       
        self.train["image"]=image[:40000]
        self.train["label"]=label[:40000]
        self.test["image"]=image[40000:]
        self.test["label"]=label[40000:]
    def show_image(self,indexs=(0,)):
        for index in indexs:
            print(self.train["label"][index])
            for i in range(28):
                for ii in range(28):
                    if 0!=self.train["image"][index][i*28+ii]:
                        print(1,end='')
                    else:
                        print(0,end='')
                print(" ")
    def nextbatch(self,BATCH_SIZE=1):
        out= [self.train["image"][self.batch_mark%40000:min(self.batch_mark%40000+BATCH_SIZE,40000)],\
            self.train["label"][self.batch_mark%40000:min(self.batch_mark%40000+BATCH_SIZE,40000)]]
        self.batch_mark+=1
        return out
    def testbatch(self,BATCH_SIZE=10,start=0):
        seed=random.randint(0,10000)
        np.random.seed(seed)
        np.random.shuffle(self.test["image"])
        np.random.seed(seed)
        np.random.shuffle(self.test["label"])
        out= [self.test["image"][start:BATCH_SIZE],\
            self.test["label"][start:BATCH_SIZE]]
        return out

if __name__=="__main__":
    ROOTPATH='./.dataset/'
    a=MNIST_Dataset(ROOTPATH)
    a.show_image(range(5))
    image,label=a.nextbatch()
    print(image)
    print(label)
    pass