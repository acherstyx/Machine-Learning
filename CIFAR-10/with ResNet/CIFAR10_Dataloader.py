import numpy
import os
import platform
from six.moves import cPickle as pickle
from mxnet.gluon import data as gdata
import numpy as np
import cv2.cv2 as cv


class CIFAR10:
    Train = {}
    Train_Size = 0
    Test = {}
    Test_Size = 0

    def __load_pickle(self, f):
        version = platform.python_version_tuple()  # 取python版本号
        if version[0] == '2':
            return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

    def __load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = self.__load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        if filename=="test_batch":
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int")
        else:
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

    def __init__(self, Data_file_dir, batch_index=[1, 2, 3, 4, 5], load_test=True):
        """ load all of cifar """
        xs = []  # list
        ys = []
        train_loder = test_loader = None
        # 训练集batch 1～5
        for b in batch_index:
            f = os.path.join(Data_file_dir, 'data_batch_%d' % (b,))
            x, y = self.__load_CIFAR_batch(f)
            xs.append(x)  # 在list尾部添加对象X, x = [..., [X]]
            ys.append(y)
            self.Train_Size += 10000
        self.Train['image'] = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
        self.Train['label'] = np.concatenate(ys)
        del x, y
        # 测试集
        if load_test == True:
            self.Test['image'], self.Test['label'] = self.__load_CIFAR_batch(os.path.join(Data_file_dir, 'test_batch'))
            self.Test_Size = 10000

    def Create_dataloader(self, batch_size, shuffle=True):
        test_dataset = gdata.ArrayDataset(self.Test["image"], self.Test['label'])
        test_loader = gdata.DataLoader(test_dataset, batch_size, shuffle=shuffle)
        train_dataset = gdata.ArrayDataset(self.Train["image"], self.Train["label"]);
        train_loader = gdata.DataLoader(train_dataset, batch_size, shuffle=shuffle)
        return train_loader, test_loader


if __name__ == "__main__":
    dataset = CIFAR10(".\.dataset")
    train, test = dataset.Create_dataloader(10)
    print(np.shape(train))
    for x, y in train:
        print(x, y)
        break
