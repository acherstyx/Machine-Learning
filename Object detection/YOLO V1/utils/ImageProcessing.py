import cv2 as cv
import numpy as np
import utils.Config as Config


def draw_bounding_box(label_array,
                      image_to_process,
                      bbox_number,
                      base_coordinate_of_xy,
                      is_logits,
                      threshold=0.8,
                      color=(0, 255, 0)):
    """
    Draw the bounding box based on the predict result or label
    Note that the function do 'in-place editing' to the image
    @param label_array: the information of boundary box
    [bbox:{confidence, x, y, w, h} * bbox_number, logits/index of predict result]
    @param image_to_process: image
    @param bbox_number: number of boundary box
    @param base_coordinate_of_xy: "IMAGE" based or "CELL" based
    whether to add offset to xy coordinate
    @param is_logits: the format of class label, True or False
    @param threshold: the bbox will only be drawn on the image if the confidence of bbox is over threshold
    @param color: the color of box and text ( , , )
    @return: image
    """
    for rows, i in enumerate(label_array):  # row
        for cols, ii in enumerate(i):  # col
            # [confidence,confidence,x,y,w,h,x,y,w,h] x,y is cell-based/image-based, w,h is all image-based
            if np.amax(ii[:bbox_number]) < threshold:
                continue
            else:
                box_index = np.argmax(ii[:bbox_number])
                offset = bbox_number + box_index * 4

                if Config.DebugOutput_ImageShow_Point:
                    print("Point info in DrawBoundingBox() - 1:", base_coordinate_of_xy, ii[offset:offset + 4])

                if base_coordinate_of_xy == "CELL":
                    # add offset to the image
                    center_x = int((ii[offset + 0] + cols) / Config.CellSize * Config.ImageSize)
                    center_y = int((ii[offset + 1] + rows) / Config.CellSize * Config.ImageSize)
                    # use square of w,h as the real w,h
                    w = int((ii[offset + 2] ** 2) * Config.ImageSize)
                    h = int((ii[offset + 3] ** 2) * Config.ImageSize)
                elif base_coordinate_of_xy == "IMAGE":
                    center_x = int(ii[offset + 0] * Config.ImageSize)
                    center_y = int(ii[offset + 1] * Config.ImageSize)
                    w = int(ii[offset + 2] * Config.ImageSize)
                    h = int(ii[offset + 3] * Config.ImageSize)
                else:
                    raise AttributeError

                # left up point
                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                # right down point
                x2 = int(center_x + w / 2)
                y2 = int(center_y + h / 2)

                if Config.DebugOutput_ImageShow_Point:
                    print("Point info in DrawBoundingBox() - 2:", x1, y1, x2, y2)

                cv.rectangle(img=image_to_process,
                             pt1=(x1, y1),
                             pt2=(x2, y2),
                             color=color,
                             thickness=1)

                if is_logits:
                    class_index = np.argmax(ii[5 * bbox_number:])
                else:
                    class_index = ii[5 * bbox_number]
                cv.putText(img=image_to_process,
                           text=Config.Classes[int(class_index)] + " {:.2f}".format(np.amax(ii[:bbox_number])),
                           org=(x1 + 5, y1 + 10),
                           fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.5,
                           color=color)

                if Config.DebugOutput_ImageShow_Logits:
                    print("Logits info in DrawBoundingBox():", ii[5 * bbox_number:])

    return image_to_process


if __name__ == "__main__":
    # load image data from Pascal VOC data set
    import utils.LoadPascalVOC as LoadPascalVoc

    # create data loader
    data_loader = LoadPascalVoc.PascalVOC()
    # get train data generator
    train_iter = data_loader.train_generator(1)
    # get valid data generator
    valid_iter = data_loader.valid_generator(1)

    # draw bbox based on label data
    print(">>> Draw bbox based on label data")
    current_image, current_label = train_iter.__next__()
    current_image = current_image["input"][0]
    current_label = current_label["output"][0]
    # get labeled image
    labeled_image = draw_bounding_box(label_array=current_label,
                                      image_to_process=current_image.copy(),
                                      bbox_number=1,
                                      base_coordinate_of_xy="IMAGE",
                                      is_logits=False,
                                      threshold=1.0,
                                      color=(255, 255, 255),
                                      )
    # show image
    cv.imshow("Deal with label from dataset", np.hstack((current_image, labeled_image)))
    cv.waitKey(2000)

    # generate output of neural network based on label
    print(">>> Show simulate label (logits, xy is image based)")
    simulate_label = np.zeros([7, 7, 30], dtype=np.float)
    simulate_label[..., 1] = current_label[..., 0]
    simulate_label[..., 6:10] = current_label[..., 1:5]
    for i in range(7):
        for ii in range(7):
            if (np.asscalar(current_label[i, ii, 0]) == 1):
                simulate_label[i, ii, 10 + int(np.asscalar(current_label[i, ii, 5]))] = 1.0
    # get labeled image
    labeled_image = draw_bounding_box(label_array=simulate_label,
                                      image_to_process=current_image.copy(),
                                      bbox_number=2,
                                      base_coordinate_of_xy="IMAGE",
                                      is_logits=True,
                                      threshold=1.0,
                                      color=(0, 255, 0),
                                      )
    # show image
    cv.imshow("Deal with simulated label", np.hstack((current_image, labeled_image)))
    cv.waitKey(2000)
