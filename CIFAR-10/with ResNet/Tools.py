import numpy
import os
import platform
from six.moves import cPickle as pickle
from mxnet.gluon import data as gdata
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
import time
import matplotlib.pyplot as plt
import cv2.cv2 as cv
import random


# crop image
def data_augmentation(image_list, size_cut):
    image_out = []
    for image in image_list:
        augmenter = mx.image.CreateAugmenter(data_shape=(3, 32, 32),
                                             rand_crop=True,
                                             rand_resize=random.randint(28, 32),
                                             rand_gray=random.random() * 0.2,
                                             rand_mirror=True,
                                             brightness=random.random() * 0.5,
                                             contrast=random.random() * 0.8,
                                             saturation=random.random() * 0.4,
                                             pca_noise=random.random() * 1.0
                                             )
        temp = nd.array(image)
        for aug in augmenter:
            temp = aug(temp)
        temp.transpose((2, 0, 1))
        image_out.append(temp.asnumpy())
    return nd.array(image_out)


class CIFAR10:
    Train = {}
    Train_Size = 0
    Test = {}
    Test_Size = 0

    @staticmethod
    def __load_pickle(f):
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
        if filename == "test_batch":
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


def Create_dataloader(path, train_batch_size, test_batch_size, shuffle=True, dataAug=True):
    data_set = CIFAR10(path)
    transform = lambda data, label: (data.astype(np.float32) / 255, label)
    # test data loader
    test_data_set = gdata.ArrayDataset(nd.array(data_set.Test["image"]), nd.array(data_set.Test['label']))
    test_loader = gdata.DataLoader(test_data_set.transform(transform),
                                   test_batch_size,
                                   shuffle=shuffle)
    # train data loader
    if dataAug:
        dataAug_data_set = data_augmentation(nd.array(data_set.Train["image"]), (24, 24))
        train_data_set = gdata.ArrayDataset(nd.concat(nd.array(data_set.Train["image"]), dataAug_data_set,
                                                      dim=0),
                                            nd.concat(nd.array(data_set.Train["label"]),
                                                      nd.array(data_set.Train["label"]),
                                                      dim=0))
    else:
        train_data_set = gdata.ArrayDataset(nd.array(data_set.Train["image"]), nd.array(data_set.Train["label"]))
    train_loader = gdata.DataLoader(train_data_set.transform(transform),
                                    train_batch_size,
                                    shuffle=shuffle)
    del data_set
    return train_loader, test_loader


class TrainTimer:
    """
    训练时计时用，setting设置时限
    在到达时限时，返回1，否则0
    若设置为-1.则无限制
    """
    limit = 0
    start_time = 0

    def __init__(self, time_limit=-1):
        self.limit = time_limit

    def start(self):
        self.start_time = time.time()

    def reset(self, time_limit=-1):
        if time_limit != -1:
            self.limit = time_limit
        self.start_time = time.time()

    def check(self):
        time_check = time.time()
        if (time_check - self.start_time) > self.limit | self.limit != -1:
            return False
        else:
            return True

    def read(self):
        return time.time() - self.start_time


if __name__ == "__main__":
    #    train, test = Create_dataloader("./.dataset", 2, 100)
    #    for x, y in train:
    #        print(x, y)
    #        break

    #    print("Batch shape: ", end="")
    #    for x, y in train:
    #        print(x.shape)
    #        break

    print("Test timer ... ", end="")
    timer = TrainTimer()
    timer.start()
    time.sleep(0.1)
    print(timer.read())

    #    print("Testing data load speed: ", end="")
    #    timer.reset()
    #    for x, y in train:
    #        feeddict=[x.asnumpy(),y.asnumpy()]
    #    print(timer.read())

    print("Test image augmentation")
    cifar10 = CIFAR10("./.dataset/")

    image_init = cifar10.Train["image"][1400]
    auged = data_augmentation([image_init, ], (24, 24))

    image_init = np.float32(image_init)
    image_init = image_init / np.max(image_init)
    image_out = auged[0].asnumpy()
    image_out = image_out / np.max(image_out)

    plt.imshow(image_init)
    plt.pause(2)
    plt.imshow(image_out)
    plt.pause(2)
