import xml.etree.cElementTree as ET
import pickle
import cv2.cv2 as cv
import numpy as np
import random
from utils import Config as cfg
import os

PASCAL_VOC_PATH = "./.VOCdevkit/"


class PascalVOC:
    def __init__(self):
        # path
        #   data
        self.ImagePath = cfg.ImagePath
        self.AnnotationPath = cfg.AnnotationsPath
        #   save
        self.SaveTrain = cfg.TrainSavePath
        self.SaveVal = cfg.ValSavePath
        #   image list
        self.ImageList = os.listdir(self.ImagePath)
        self.ImageList = [i.replace(".jpg", "") for i in self.ImageList]
        random.shuffle(self.ImageList)

        # image settings
        self.ImageSize = cfg.ImageSize
        self.CellSize = cfg.CellSize

        # class
        self.Classes = cfg.Classes
        self.ClassesDict = cfg.ClassesDict

        # divide data set to train and val
        self.TrainPercentage = cfg.TrainPercentage
        self.TrainNum = int(len(self.ImageList) * self.TrainPercentage)
        self.ValNum = len(self.ImageList) - self.TrainNum
        self.TrainImageList = self.ImageList[:self.TrainNum]
        self.ValImageList = self.ImageList[self.TrainNum:]

        # start loading data
        if cfg.LoadSavedData and os.path.isfile(self.SaveTrain) and os.path.isfile(self.SaveVal):
            self.TrainData = self.__LoadPickle__(self.SaveTrain)
            self.ValData = self.__LoadPickle__(self.SaveVal)
        else:
            # load label
            self.__LoadImageLabel__()
            self.__SaveToPickle__(self.SaveTrain, self.TrainData)
            self.__SaveToPickle__(self.SaveVal, self.ValData)

    # accelerated loading save
    @staticmethod
    def __SaveToPickle__(file_path, data_to_save):
        print("Save data to: " + file_path)
        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)

    # accelerated loading load
    @staticmethod
    def __LoadPickle__(file_path):
        print("Load data from: " + file_path)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    # load from raw pascal voc data
    def __LoadImageLabel__(self):
        print("Loading labels from: " + self.AnnotationPath)

        self.TrainData = []
        # load train data set
        for TrainImage in self.TrainImageList:
            current_label, has_object = self.__LoadLabel__(TrainImage)
            if has_object == 0:
                continue
            current_image_path = os.path.join(self.ImagePath, TrainImage + ".jpg")
            self.TrainData.append({"ImagePath": current_image_path, "Label": current_label})
        self.ValData = []
        # load val data set
        for ValImage in self.ValImageList:
            current_label, has_object = self.__LoadLabel__(ValImage)
            if has_object == 0:
                continue
            current_image_path = os.path.join(self.ImagePath, ValImage + ".jpg")
            self.ValData.append({"ImagePath": current_image_path, "Label": current_label})

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
            box_info = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]

            cell_x = int(box_info[0] * self.CellSize / self.ImageSize)
            cell_y = int(box_info[1] * self.CellSize / self.ImageSize)

            if label_temp[cell_x, cell_y, 0] == 1:  # this cell already has an object
                continue
            else:
                object_counter += 1
            # format: hasObject[0] boxinfo[1:5] class_id[5]
            label_temp[cell_x, cell_y, 4] = 1
            label_temp[cell_x, cell_y, 0:4] = [i / cfg.ImageSize for i in box_info]
            label_temp[cell_x, cell_y, 5] = class_id

        return label_temp, object_counter

    def train_generator(self, batch_size):
        for i in range(batch_size, self.TrainNum, batch_size):
            batch_img = []
            batch_label = []
            batch_data = self.TrainData[i - batch_size: i]
            for single_sample in batch_data:
                batch_img.append(cv.resize(cv.imread(single_sample["ImagePath"]), (cfg.ImageSize, cfg.ImageSize)))
                batch_label.append(single_sample["Label"])
            batch_img = np.array(batch_img)
            batch_label = np.array(batch_label)
            yield {"input": np.array(batch_img, dtype=np.float)}, {"output": np.array(batch_label)}

    def val_generator(self, batch_size):
        for i in range(batch_size, self.ValNum, batch_size):
            batch_img = []
            batch_label = []
            batch_data = self.ValData[i - batch_size: i]
            for single_sample in batch_data:
                batch_img.append(cv.resize(cv.imread(single_sample["ImagePath"]), (cfg.ImageSize, cfg.ImageSize)))
                batch_label.append(single_sample["Label"])
            batch_img = np.array(batch_img)
            batch_label = np.array(batch_label)
            yield {"input": np.array(batch_img, dtype=np.float)}, {"output": np.array(batch_label)}


if __name__ == "__main__":
    data = PascalVOC()

    val_iter = data.val_generator(10)

    for image, label in val_iter:
        print("Data shape:", np.shape(image["input"]), np.shape(label["output"]))
        break

    # get image sample
    index = int(input("Index of image to show: "))
    while True:
        sample = data.TrainData[index]
        image_path = sample["ImagePath"]
        label = sample["Label"]

        image = cv.imread(image_path)
        image = cv.resize(image, (cfg.ImageSize, cfg.ImageSize))
        cv.putText(image,
                   str(index) + image_path,
                   (0, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv.imshow("InitImage", image)
        for i in label:
            for ii in i:
                if ii[4] != 0.0:
                    cv.rectangle(image,
                                 (int((ii[0] - ii[2] / 2) * cfg.ImageSize), int((ii[1] - ii[3] / 2) * cfg.ImageSize)),
                                 (int((ii[0] + ii[2] / 2) * cfg.ImageSize), int((ii[1] + ii[3] / 2) * cfg.ImageSize)),
                                 (0, 255, 0),
                                 2)
                    cv.putText(image,
                               cfg.Classes[int(ii[5])],
                               (int((ii[0] - ii[2] / 2) * cfg.ImageSize + 2),
                                int((ii[1] - ii[3] / 2) * cfg.ImageSize + 12)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv.imshow("LabeledImage", image)
        cv.waitKey()
        index += 1
