import mxnet.gluon.data as gdata
import tensorflow as tf
import struct
import os
import matplotlib.pyplot as plt
import random
import mxnet.ndarray as nd
import mxnet as mx
from PIL import Image

# 训练集文件
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'
# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'


class MNIST_Dataset():
    train = {}
    test = {}
    train_batch_index = 0
    test_batch_index = 0

    def __read_image(self, path_to_file, show_process):
        bin_data = open(path_to_file, 'rb').read()
        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。
        # 我们后面会看到标签集中，只使用2个ii。
        fmt_header = '>iiii'
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
        offset += struct.calcsize(fmt_header)
        # print(offset)
        # 图像数据像素值的类型为unsigned char型，对应的format格式为B。
        # 这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
        fmt_image = '>' + str(image_size) + 'B'
        # print(fmt_image,offset,struct.calcsize(fmt_image))
        images = nd.empty((num_images, num_rows, num_cols, 3))
        # plt.figure()
        print("Loading...", end="")
        for i in range(num_images):
            if show_process == True and (i + 1) % 5000 == 0:
                print(' %.0f' % ((i + 1) / num_images * 100), end='%')
                # print(offset)
            images[i, :, :, 0] = images[i, :, :, 1] = images[i, :, :, 2] = nd.array(
                struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols)) / 255
            # print(images[i])
            offset += struct.calcsize(fmt_image)
            # plt.imshow(images[i],'gray')
            # plt.pause(0.00001)
            # plt.show()
        print("")
        # plt.show()
        return images

    def __read_label(self, path_to_file, show_process):
        # 读取二进制数据
        bin_data = open(path_to_file, 'rb').read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = nd.empty(num_images)
        print("Loading...", end="")
        for i in range(num_images):
            if show_process == True and (i + 1) % 5000 == 0:
                print(' %.0f' % ((i + 1) / num_images * 100), end='%')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        print("")
        labels = nd.cast(labels, int)
        return labels

    def __init__(self, root_path, show_process=False):
        # train image
        print(">>> ", train_images_idx3_ubyte_file)
        file_path = os.path.join(root_path, train_images_idx3_ubyte_file)
        self.train["image"] = self.__read_image(file_path, show_process)
        print(">>> ", train_labels_idx1_ubyte_file)
        file_path = os.path.join(root_path, train_labels_idx1_ubyte_file)
        self.train["label"] = self.__read_label(file_path, show_process)
        print(">>> ", test_images_idx3_ubyte_file)
        file_path = os.path.join(root_path, test_images_idx3_ubyte_file)
        self.test["image"] = self.__read_image(file_path, show_process)
        print(">>> ", test_labels_idx1_ubyte_file)
        file_path = os.path.join(root_path, test_labels_idx1_ubyte_file)
        self.test["label"] = self.__read_label(file_path, show_process)

    def show_sample(self, dataset="train", amount=5, index=None, pause=0.5):
        if index is not None:
            if dataset == "train":
                plt.figure()
                plt.imshow(self.train["image"][index], "gray")
                plt.text(1, -0.6,
                         "Number: {num}  Label: {label}".format(num=int(self.train["label"][index]), label=index),
                         fontsize=15)
                plt.show()
                return
            elif dataset == "test":
                plt.figure()
                plt.imshow(self.test["image"][index], "gray")
                plt.text(1, -0.6,
                         "Number: {num}  Label: {label}".format(num=int(self.test["label"][index]), label=index),
                         fontsize=15)
                plt.show()
                return
            else:
                raise ValueError
        if dataset == "train":
            for i in range(amount):
                index = random.randint(0, 60000)
                plt.ion()
                plt.figure(i)
                plt.imshow(self.train["image"][index], "gray")
                plt.text(1, -0.6,
                         "Number: {num}  Label: {label}".format(num=int(self.train["label"][index]), label=index),
                         fontsize=15)
                plt.show()
                plt.pause(0.1)
            plt.pause(pause)
            for i in range(amount):
                plt.close(i)
        elif dataset == "test":
            for i in range(amount):
                index = random.randint(0, 10000)
                plt.ion()
                plt.figure(i)
                plt.imshow(self.test["image"][index], "gray")
                plt.text(1, -0.6,
                         "Number: {num}  Label: {label}".format(num=int(self.test["label"][index]), label=index),
                         fontsize=15)
                plt.show()
                plt.pause(0.07)
            plt.pause(0.5)
            for i in range(amount):
                plt.close(i)
        else:
            raise ValueError


def Image_preprocessed(image_list):
    return image_list


def Create_dataloader(data_path, train_batch_size, test_batch_size, shuffle=True, show_process=False):
    data = MNIST_Dataset(data_path, show_process=show_process)
    data_set = gdata.ArrayDataset(Image_preprocessed(data.train["image"]),
                                  Image_preprocessed(data.train["label"]))
    train_dataloader = gdata.DataLoader(data_set, train_batch_size, shuffle=shuffle)
    data_set = gdata.ArrayDataset(Image_preprocessed(data.test["image"]),
                                  Image_preprocessed(data.test["label"]))
    test_dataloader = gdata.DataLoader(data_set, test_batch_size, shuffle=shuffle)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    # dataset = MNIST_Dataset("./.dataset", show_process=True)
    train_data_loader, test_data_loader = Create_dataloader("./.dataset", 10, 100, show_process=True)
    for x, y in train_data_loader:
        print(x.shape)
        break
