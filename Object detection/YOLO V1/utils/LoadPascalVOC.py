import xml.etree.cElementTree as ET
import pickle
import cv2 as cv
import numpy as np
import random
from utils import Config
import os

PASCAL_VOC_PATH = "./.VOCdevkit/"


class PascalVOC:
    def __init__(self):
        # path
        #   data
        self.ImagePath = Config.ImagePath
        self.AnnotationPath = Config.AnnotationsPath
        #   save
        self.SaveTrain = Config.TrainSavePath
        self.SaveVal = Config.ValSavePath
        #   image list
        self.ImageList = os.listdir(self.ImagePath)
        self.ImageList = [i.replace(".jpg", "") for i in self.ImageList]
        random.shuffle(self.ImageList)

        # image settings
        self.ImageSize = Config.ImageSize
        self.CellSize = Config.CellSize

        # class
        self.Classes = Config.Classes
        self.ClassesDict = Config.ClassesDict

        # divide data set to train and val
        self.TrainPercentage = Config.TrainPercentage
        self.TrainNum = int(len(self.ImageList) * self.TrainPercentage)
        self.ValNum = len(self.ImageList) - self.TrainNum
        self.TrainImageList = self.ImageList[:self.TrainNum]
        self.ValImageList = self.ImageList[self.TrainNum:]

        # start loading data
        if Config.LoadSavedData and os.path.isfile(self.SaveTrain) and os.path.isfile(self.SaveVal):
            self.TrainData = self.__LoadFromPickle__(self.SaveTrain)
            self.ValidData = self.__LoadFromPickle__(self.SaveVal)
        else:
            # load label
            self.__LoadImageLabel__()
            self.__SaveToPickle__(self.SaveTrain, self.TrainData)
            self.__SaveToPickle__(self.SaveVal, self.ValidData)

    # accelerated loading save
    @staticmethod
    def __SaveToPickle__(file_path, data_to_save):
        print("Save data to: " + file_path)
        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)

    # accelerated loading load
    @staticmethod
    def __LoadFromPickle__(file_path):
        print("Load data from cache: " + file_path)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    # load from raw pascal voc data
    def __LoadImageLabel__(self):
        print("Loading labels from annotation directory: " + self.AnnotationPath)

        self.TrainData = []
        # load train data set
        for TrainImage in self.TrainImageList:
            current_label, has_object = self.__LoadLabel__(TrainImage)
            if has_object == 0:
                continue
            current_image_path = os.path.join(self.ImagePath, TrainImage + ".jpg")
            self.TrainData.append({"ImagePath": current_image_path, "Label": current_label})

        self.ValidData = []
        # load val data set
        for ValImage in self.ValImageList:
            current_label, has_object = self.__LoadLabel__(ValImage)
            if has_object == 0:
                continue
            current_image_path = os.path.join(self.ImagePath, ValImage + ".jpg")
            self.ValidData.append({"ImagePath": current_image_path, "Label": current_label})

    def __LoadLabel__(self, image_name):
        annotation_path = os.path.join(self.AnnotationPath, image_name + ".xml")

        label_temp = np.zeros((self.CellSize, self.CellSize, 6))

        # xml preload
        xml_tree = ET.parse(annotation_path)
        xml_object = xml_tree.findall("object")

        image_size = xml_tree.find("size")
        image_size = [int(image_size.find("height").text),
                      int(image_size.find("width").text)]

        h_ratio = 1.0 * self.ImageSize / image_size[0]
        w_ratio = 1.0 * self.ImageSize / image_size[1]

        object_counter = 0
        for Object in xml_object:
            Box = Object.find("bndbox")
            x1 = max(min((float(Box.find('xmin').text) - 1) * w_ratio, self.ImageSize - 1), 0)
            x2 = max(min((float(Box.find('xmax').text) - 1) * w_ratio, self.ImageSize - 1), 0)
            y1 = max(min((float(Box.find('ymin').text) - 1) * h_ratio, self.ImageSize - 1), 0)
            y2 = max(min((float(Box.find('ymax').text) - 1) * h_ratio, self.ImageSize - 1), 0)

            class_id = self.ClassesDict[Object.find("name").text.lower().strip()]
            box_info = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]  # x,y,w,h

            cell_x = int(box_info[0] * self.CellSize / self.ImageSize)
            cell_y = int(box_info[1] * self.CellSize / self.ImageSize)

            if label_temp[cell_y, cell_x, 0] == 1:  # this cell already has an object
                continue
            else:
                object_counter += 1
            # format: confidence[0] x,y,w,h[0:4] classes[5]
            label_temp[cell_y, cell_x, 0] = 1
            label_temp[cell_y, cell_x, 1:5] = [i / Config.ImageSize for i in box_info]
            label_temp[cell_y, cell_x, 5] = class_id

        return label_temp, object_counter

    def train_generator(self, batch_size):
        while True:
            for i in range(batch_size, self.TrainNum, batch_size):
                batch_img = []
                batch_label = []
                batch_data = self.TrainData[i - batch_size: i]
                for single_sample in batch_data:
                    batch_img.append(
                        cv.resize(cv.imread(single_sample["ImagePath"]), (Config.ImageSize, Config.ImageSize)))
                    batch_label.append(single_sample["Label"])
                yield {"input": np.array(batch_img, dtype=np.float) / 255.0}, \
                      {"output": np.array(batch_label, dtype=np.float)}

    def valid_generator(self, batch_size):
        while True:
            for i in range(batch_size, self.ValNum, batch_size):
                batch_img = []
                batch_label = []
                batch_data = self.ValidData[i - batch_size: i]
                for single_sample in batch_data:
                    batch_img.append(
                        cv.resize(cv.imread(single_sample["ImagePath"]), (Config.ImageSize, Config.ImageSize)))
                    batch_label.append(single_sample["Label"])
                batch_img = np.array(batch_img)
                batch_label = np.array(batch_label)

                yield {"input": np.array(batch_img, dtype=np.float) / 255.0}, \
                      {"output": np.array(batch_label, dtype=np.float)}


if __name__ == "__main__":
    data = PascalVOC()

    # test val generator
    print(">>> Test output shape of validation data generator")
    # get generator
    val_iter = data.valid_generator(10)
    # print the data shape of validation generator
    for image, label in val_iter:
        print("Data shape:", np.shape(image["input"]), np.shape(label["output"]))
        break

    # test the amount of train and validation data
    print(">>> Test the amount of train and validation data")
    print("The length of train data is:", len(data.TrainData))
    print("The length of valid data is:", len(data.ValidData))

    # get image sample from train generator
    print(">>> Show sample image of train data without using generator")
    index = random.randint(0, len(data.TrainData) - 11)
    print("Index of image to show:", index)
    # show 10 sample image
    for _ in range(10):
        # dictionary of data
        sample = data.TrainData[index]
        # separate data from the dict
        image_path = sample["ImagePath"]
        label = sample["Label"]
        # read image
        image = cv.imread(image_path)
        # resize to the shape used in neural network model
        image = cv.resize(image, (Config.ImageSize, Config.ImageSize))
        # make a copy
        image_old = image.copy()
        # use label to add bbox and class name
        for i in label:
            for ii in i:
                if ii[0] != 0.0:
                    # draw bound box on image
                    cv.rectangle(image,
                                 (int((ii[1] - ii[3] / 2) * Config.ImageSize),
                                  int((ii[2] - ii[4] / 2) * Config.ImageSize)),
                                 (int((ii[1] + ii[3] / 2) * Config.ImageSize),
                                  int((ii[2] + ii[4] / 2) * Config.ImageSize)),
                                 (0, 255, 0),
                                 2)
                    # put class name on image
                    cv.putText(image,
                               Config.Classes[int(ii[5])],
                               (int((ii[1] - ii[3] / 2) * Config.ImageSize + 2),
                                int((ii[2] - ii[4] / 2) * Config.ImageSize + 15)),
                               cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
        # stack labeled and unlabeled image
        image = np.hstack((image_old, image))
        # put the path info of the image
        cv.putText(image,
                   "index: " + str(index) + " path: " + image_path,
                   (0, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv.imshow("Labeled Train Image", image)
        cv.waitKey(500)
        # update index of the showing image
        index += 1

    # get image sample from valid generator
    print(">>> Show sample image of valid data without using generator")
    index = random.randint(0, len(data.ValidData) - 11)
    print("Index of image to show:", index)
    # show 10 sample image
    for _ in range(10):
        # dictionary of data
        sample = data.ValidData[index]
        # separate data from the dict
        image_path = sample["ImagePath"]
        label = sample["Label"]
        # read image
        image = cv.imread(image_path)
        # resize to the shape used in neural network model
        image = cv.resize(image, (Config.ImageSize, Config.ImageSize))
        # make a copy
        image_old = image.copy()
        # use label to add bbox and class name
        for i in label:
            for ii in i:
                if ii[0] != 0.0:
                    # draw bound box on image
                    cv.rectangle(image,
                                 (int((ii[1] - ii[3] / 2) * Config.ImageSize),
                                  int((ii[2] - ii[4] / 2) * Config.ImageSize)),
                                 (int((ii[1] + ii[3] / 2) * Config.ImageSize),
                                  int((ii[2] + ii[4] / 2) * Config.ImageSize)),
                                 (0, 255, 0),
                                 2)
                    # put class name on image
                    cv.putText(image,
                               Config.Classes[int(ii[5])],
                               (int((ii[1] - ii[3] / 2) * Config.ImageSize + 2),
                                int((ii[2] - ii[4] / 2) * Config.ImageSize + 15)),
                               cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
        # stack labeled and unlabeled image
        image = np.hstack((image_old, image))
        # put the path info of the image
        cv.putText(image,
                   "index: " + str(index) + " path: " + image_path,
                   (0, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv.imshow("Labeled Valid Image", image)
        cv.waitKey(500)
        # update index of the showing image
        index += 1
