import cv2 as cv
import numpy as np
import utils.Config as Config
from math import sqrt


def DrawBoundingBox(predict_result, current_image, bbox_num, base_coordinate, is_logits,
                    threshold=0.8,
                    color=(0, 255, 0), ):
    """

    @param predict_result: the information of boundary box
    @param current_image: image
    @param bbox_num: number of boundary box
    @param base_coordinate: "IMAGE" based or "CELL" based
    @param is_logits: the format of class label, True or False
    @param threshold:
    @param color:
    @return: image
    """
    for rows, i in enumerate(predict_result):  # row
        for cols, ii in enumerate(i):  # col
            # [confidence,confidence,x,y,w,h,x,y,w,h] x,y is cell-based w,h is image-based
            if np.amax(ii[:bbox_num]) < threshold:
                continue
            else:
                box_index = np.argmax(ii[:bbox_num])
                offset = box_index * 4 + bbox_num

                if Config.DebugOutput_ImageShow_Point:
                    print(base_coordinate, ii[offset:offset + 4])

                if base_coordinate == "CELL":
                    center_x = int((ii[offset + 0] + rows) / Config.CellSize * Config.ImageSize)
                    center_y = int((ii[offset + 1] + cols) / Config.CellSize * Config.ImageSize)
                elif base_coordinate == "IMAGE":
                    center_x = int(ii[offset + 0] * Config.ImageSize)
                    center_y = int(ii[offset + 1] * Config.ImageSize)
                else:
                    raise AttributeError

                w = int(ii[offset + 2] * Config.ImageSize)
                h = int(ii[offset + 3] * Config.ImageSize)

                x1 = int(center_x - h / 2)
                y1 = int(center_y - w / 2)
                x2 = int(center_x + h / 2)
                y2 = int(center_y + w / 2)

                if Config.DebugOutput_ImageShow_Point:
                    print("Point info in DrawBoundingBox():", x1, y1, x2, y2)

                cv.rectangle(current_image,
                             (x1, y1),
                             (x2, y2),
                             color=color,
                             thickness=1)
                if is_logits:
                    class_index = np.argmax(ii[5 * bbox_num:])
                else:
                    class_index = ii[5 * bbox_num]
                cv.putText(current_image,
                           Config.Classes[int(class_index)] + " {:.2f}".format(np.amax(ii[:bbox_num])),
                           (x1 + 10, y1 + 15),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           color)

    print("====================")
    return current_image


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
        image_result = DrawBoundingBox(result, image[0], 2, "CELL", True, 0.8)
        cv.imshow("out", image_result)
        cv.waitKey()
        break

    iter = data.train_generator(1)
    for image, label in iter:
        image = image["input"][0]
        label = label["output"][0]

        image_result = DrawBoundingBox(label, image, 1, "IMAGE", False)
        cv.imshow("out", image_result)

        result = np.zeros([7, 7, 30])
        result[:, :, 0] = result[:, :, 1] = label[:, :, 0]
        result[:, :, 2:6] = result[:, :, 6:10] = label[:, :, 1:5]
        image_result = DrawBoundingBox(result, image, 2, "IMAGE", True, 0.8)
        cv.imshow("out2", image_result)

        cv.waitKey()
