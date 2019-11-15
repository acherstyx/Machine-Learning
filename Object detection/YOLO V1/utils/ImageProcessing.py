import cv2 as cv
import numpy as np
import utils.Config as Config
from math import sqrt


def DrawBoundingBox(predict_result, image, bbox_Num, Threshold=0.99):
    for rows, i in enumerate(predict_result):  # row
        for cols, ii in enumerate(i):  # col
            # [confidence,confidence,x,y,w,h,x,y,w,h] x,y is cell-based w,h is image-based
            if np.amax(ii[:bbox_Num]) < Threshold:
                continue
            else:
                box_index = np.argmax(ii[:bbox_Num])
                offset = box_index * 4 + bbox_Num

                print(ii[offset:offset + 4])
                center_x = int((ii[offset + 0] + rows) / Config.CellSize * Config.ImageSize)
                center_y = int((ii[offset + 1] + cols) / Config.CellSize * Config.ImageSize)

                w = int(ii[offset + 2] * Config.ImageSize)
                h = int(ii[offset + 3] * Config.ImageSize)

                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                x2 = int(center_x + w / 2)
                y2 = int(center_y + h / 2)
                print(x1, y1, x2, y2)
                cv.rectangle(image,
                             (x1, y1),
                             (x2, y2),
                             (0, 255, 0),
                             2)
    print("====================")
    return image


if __name__ == "__main__":
    import utils.LoadPascalVOC as voc

    data = voc.PascalVOC()
    iter = data.train_generator(1)

    for image, label in iter:
        image = image["input"]
        label = label["output"]
        result = np.zeros([7, 7, 30])
        result[1, 1, :] = [0.5, 0.99,
                           0.99, 0.99, 0.1, 0.1,
                           0.5, 0.5, 0.1, 0.1,
                           0.5, 0.5, 0.4, 0.8, 0.9, 0.6, 0.4, 0.4, 0.4, 0.5,
                           0.55, 0.88, 0.99, 0.99, 0.14, 0.66, 0.54, 0.55, 0.66, 0.55]
        image_result = DrawBoundingBox(result, image[0], 2, 0.8)
        cv.imshow("out", image_result)
        cv.waitKey()
        break
