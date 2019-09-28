import cv2.cv2 as cv2
import numpy as np
import os
import random
from math import *

INIT_DATASET_PATH = "./.init_dataset/"
BACKGROUND_IMAGE = "./.background_image/"
OUTPUT_DATASET_PATH = "./.dataset/"
STYLES = ["Clubs", "Diamonds", "Hearts", "Spades"]


def all_path(path):
    result = []  # 所有的文件

    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            result.append(apath)
    print(result)
    return result


def __show(image):
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(all_path(BACKGROUND_IMAGE))

    background_img = []
    for image_file in all_path(BACKGROUND_IMAGE):
        background_img.append(cv2.resize(cv2.imread(image_file, 1), (299, 299)))

    # for image in background_img:
    #    cv2.imshow("image", image)
    #    cv2.waitKey()

    print(background_img[0])

    poker_image = []
    for style in STYLES:
        style_image = []
        for image_file in all_path(INIT_DATASET_PATH + style):
            style_image.append(cv2.imread(image_file, 1))
        poker_image.append(style_image)

    # for style in poker_image:
    #    for style_image in style:
    #        cv2.imshow("image", style_image)
    #        cv2.waitKey()

    output_image = []
    for style, style_name in zip(poker_image, STYLES):
        for style_image, number in zip(style, range(13)):
            # pick a image and start to add to a background:
            for index, background in enumerate(background_img):
                while True:
                    style_image_resized = cv2.resize(style_image, (random.randint(40, 200), random.randint(50, 200)))
                    size = style_image_resized.shape
                    if abs(size[0] - size[1]) < 100:
                        break
                # __show(style_image_resized)

                degree = random.randint(-5, 5)
                # 旋转后的尺寸
                widthNew = heightNew = max(int(size[0] * fabs(sin(radians(degree))) +
                                               size[1] * fabs(cos(radians(degree)))),
                                           int(size[1] * fabs(sin(radians(degree))) +
                                               size[0] * fabs(cos(radians(degree)))))
                matRotation = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), degree, 1)
                matRotation[0, 2] += (widthNew - size[0]) / 2  # 重点在这步，目前不懂为什么加这步
                matRotation[1, 2] += (heightNew - size[1]) / 2  # 重点在这步
                style_image_resized = cv2.warpAffine(style_image_resized, matRotation, (widthNew, heightNew),
                                                     borderValue=(random.randint(0, 255),
                                                                  random.randint(0, 255),
                                                                  random.randint(0, 255)))
                size = style_image_resized.shape
                # __show(style_image_resized)

                limit = (299 - size[0], 299 - size[1])
                position = (random.randint(0, limit[0]), random.randint(0, limit[1]))
                background = cv2.copyTo(background, background)
                background[position[0]:position[0] + size[0], position[1]:position[1] + size[1]] = style_image_resized
                # __show(background)
                image_file_name = OUTPUT_DATASET_PATH + style_name + "/" + str(number) + "_" + str(index) + ".jpg"
                cv2.imwrite(image_file_name, background)
