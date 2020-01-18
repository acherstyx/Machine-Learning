import numpy as np
import tensorflow as tf
import utils.Config as Config
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def calc_iou(box1, box2):
    """
    calculate IoU of 2 bound box
    box1 and box2 have the same shape because pre-process in yolo_loss, don't worry!
    :param box1: [batch_size,cell_size,cell_size,box_per_cell(2),4] (x,y,w,h)
    :param box2: [batch_size,cell_size,cell_size,box_per_cell(2),4] (x,y,w,h)
    return
        iou: [batch_size,cell_size,cell_size,box_pre_cell]
    """
    box1_point = tf.stack([box1[..., 0] - box1[..., 2] / 2.0,
                           box1[..., 1] - box1[..., 3] / 2.0,
                           box1[..., 0] + box1[..., 2] / 2.0,
                           box1[..., 1] + box1[..., 3] / 2.0],
                          axis=-1)
    box2_point = tf.stack([box2[..., 0] - box2[..., 2] / 2.0,
                           box2[..., 1] - box2[..., 3] / 2.0,
                           box2[..., 0] + box2[..., 2] / 2.0,
                           box2[..., 1] + box2[..., 3] / 2.0],
                          axis=-1)

    # left up point and right down point of intersection

    left_up = tf.maximum(box1_point[..., :2], box2_point[..., :2])
    right_down = tf.minimum(box1_point[..., 2:], box2_point[..., 2:])
    # get area of intersection
    intersection_wh = tf.maximum(0.0, right_down - left_up)
    intersection_area = tf.maximum(intersection_wh[..., 0] * intersection_wh[..., 1], 1e-10)

    # calculate the area of box1 and box2
    box1_area = box1[..., 2] * box1[..., 3]
    box2_area = box2[..., 2] * box2[..., 3]

    union_area = tf.maximum(box1_area + box2_area - intersection_area, 1e-10)
    out = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0, name="current_iou")
    tf.keras.backend.log(out)
    return out


def yolo_loss(y_true, y_pred, **kwargs):
    """
    calculate loss for yolo model output
    :param y_true: has a shape of batch_size*cell_size*cell_size*6 [x,y,w,h,conf,classes]
    :param y_pred: has a shape of batch_size*cell_size*cell_size*30 [conf,conf,x,y,w,h,x,y,w,h,classes...]
    :return: loss
    """
    batch_size = y_true.get_shape()[0]
    # unpack data
    #   pred
    pred_classes = y_pred[..., 10:]  # [_,7,7,20]
    pred_confidence = y_pred[..., :2]  # [_,7,7,2]
    pred_bbox_cell_base = tf.reshape(y_pred[..., 2:10],
                                     [-1, Config.CellSize, Config.CellSize, Config.BoxPerCell, 4])  # [_,7,7,2,4]

    # TODO: remove debug output
    if Config.DebugOutput_PredBox:
        try:
            print("\n - pred of box in yolo loss function input: ", y_pred[..., :10].numpy())
        except AttributeError:
            pass

    #  true
    true_classes = y_true[..., 5]  # [_,7,7,1] use index instead of probabilities
    true_confidence = tf.reshape(y_true[..., 0],
                                 (-1, Config.CellSize, Config.CellSize, 1))  # [_,7,7,1]
    true_bbox_image_base = tf.reshape(y_true[..., 1:5],
                                      [-1, Config.CellSize, Config.CellSize, 1, 4])  # [_,7,7,1,4]
    true_bbox_image_base = tf.tile(true_bbox_image_base,
                                   [1, 1, 1, Config.BoxPerCell, 1])  # fit bbox number in prediction

    # get offset
    offset = tf.constant(Config.Offset, dtype=tf.float32)
    offset = tf.reshape(offset, (1, Config.CellSize, Config.CellSize, Config.BoxPerCell))
    try:
        offset = tf.tile(offset, [batch_size, 1, 1, 1])
    except TypeError:
        pass
    offset_t = tf.transpose(offset, (0, 2, 1, 3))  # transpose of offset

    # Cell-based coordinates -> Image-based coordinates, for IoU(
    pred_bbox_image_base = tf.stack([(pred_bbox_cell_base[..., 0] + offset) / Config.CellSize,
                                     (pred_bbox_cell_base[..., 1] + offset_t) / Config.CellSize,
                                     tf.square(pred_bbox_cell_base[..., 2]),
                                     tf.square(pred_bbox_cell_base[..., 3])],
                                    axis=-1
                                    )
    # Image-based coordinates -> Cell-based coordinates, for loss
    true_bbox_cell_base = tf.stack([true_bbox_image_base[..., 0] * Config.CellSize - offset,
                                    true_bbox_image_base[..., 1] * Config.CellSize - offset_t,
                                    tf.sqrt(true_bbox_image_base[..., 2]),
                                    tf.sqrt(true_bbox_image_base[..., 3])],
                                   axis=-1)

    # get iou of box
    iou_of_pred_and_true = calc_iou(pred_bbox_image_base, true_bbox_image_base)

    # TODO: remove debug output
    if Config.DebugOutput_IOU:
        try:
            print(" - all iou in yolo loss function: ", iou_of_pred_and_true.numpy)
            print(" - average iou in yolo loss function: ", tf.reduce_mean(iou_of_pred_and_true).numpy())
        except AttributeError:
            pass

    # get mask
    obj_mask = tf.reduce_max(iou_of_pred_and_true,
                             axis=3,
                             keepdims=True)  # choose the larger one as the result of prediction
    obj_mask = tf.cast(
        (iou_of_pred_and_true >= obj_mask),
        tf.float32
    ) * true_confidence
    no_obj_mask = tf.ones_like(obj_mask, dtype=tf.float32) - obj_mask

    # classify loss
    true_classes_one_hot = tf.one_hot(indices=tf.cast(true_classes, dtype=tf.uint8),
                                      depth=Config.ClassesNum,
                                      on_value=1.0,
                                      off_value=0.0,
                                      axis=-1,
                                      dtype=tf.float32)
    classes_delta = tf.square((pred_classes - true_classes_one_hot) * true_confidence)
    classes_loss = tf.reduce_mean(
        tf.reduce_sum(classes_delta, axis=[1, 2, 3])
    )

    # has object loss
    object_delta = tf.square((pred_confidence - iou_of_pred_and_true) * obj_mask)
    object_loss = tf.reduce_mean(
        tf.reduce_sum(object_delta, axis=[1, 2, 3])
    )

    # TODO: remove debug output
    if Config.DebugOutput_ObjectDelta:
        try:
            print(" - object delta in yolo loss function: ", object_delta[0].numpy)
        except AttributeError:
            pass

    # no object loss
    no_obj_delta = tf.square(no_obj_mask * pred_confidence)
    no_obj_loss = tf.reduce_mean(
        tf.reduce_sum(no_obj_delta, axis=[1, 2, 3])
    )

    # TODO: remove debug output
    if Config.DebugOutput_NoObjectDelta:
        try:
            print(" - no_object delta in yolo loss function: ", no_obj_delta.numpy())
        except AttributeError:
            pass

    # box loss
    coordinate_mask = tf.expand_dims(obj_mask, 4)
    bbox_delta = tf.square(coordinate_mask * (pred_bbox_cell_base - true_bbox_cell_base))
    bbox_loss = tf.reduce_mean(
        tf.reduce_sum(bbox_delta, axis=[1, 2, 3, 4])
    )

    # TODO: remove debug output
    if Config.DebugOutput_loss:
        try:
            print("- obj_loss     ", object_loss.numpy(),
                  "\n- no_obj_loss  ", no_obj_loss.numpy(),
                  "\n- bbox_loss    ", bbox_loss.numpy(),
                  "\n- classes_loss ", classes_loss.numpy())
        except AttributeError:
            pass

    return object_loss * Config.LossWeight_Object + \
           no_obj_loss * Config.LossWeight_NoObject + \
           bbox_loss * Config.LossWeight_Coordinate + \
           classes_loss * Config.LossWeight_Classes


if __name__ == "__main__":
    Config.DebugOutput_loss = True
    Config.DebugOutput_IOU = True
    Config.DebugOutput_ObjectDelta = True
    Config.DebugOutput_NoObjectDelta = True
    Config.DebugOutput_PredBox = True
    Config.DebugOutput_loss = True

    print(">>> loss test - pred and true is all zero")
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    print("loss: ", yolo_loss(tf_y_true, tf_y_pred).numpy())

    print(">>> loss test - pred:confidence all=1 true:confidence all=0")
    # the truth is "no object", but the predict result is "has object",
    # so calculate "no object loss".
    # The class loss and coordinate loss is ignored by the truth that no object in the box.
    # the result should be: 2*CellSize*CellSize*Weight 2*7*7*0.5=49
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    y_pred[..., :2] = 1
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    print("loss: ", yolo_loss(tf_y_true, tf_y_pred).numpy())

    print(">>> loss test - pred:confidence all=0 true:confidence all=1, all coordinate is same")
    # the truth is has object, the predict is no object
    # so the no object loss = 0
    # the bbox loss = 0 (all zero, same)
    # then calculate the iou = 1, so the "has object" loss = 2(bbox number)*CellSize*CellSize
    # the class[0] should be 1, so "class loss" = CellSize*CellSize
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    y_true[..., :1] = 1
    for i in range(7):
        for ii in range(7):
            # conf,x,y,w,h,class
            #   0  1 2 3 4   5
            y_true[:, i, ii, 1] = ii / 7
            y_true[:, i, ii, 2] = i / 7
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    print("loss: ", yolo_loss(tf_y_true, tf_y_pred).numpy())

    print(">>> loss test - pred:confidence all=0 true:confidence all=1, change coordinate")
    # the same with previous test case
    # except coordinnate
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    y_true[..., :1] = 1
    for i in range(7):
        for ii in range(7):
            # conf,x,y,w,h,class
            #   0  1 2 3 4   5
            y_true[:, i, ii, 1] = ii / 7
            y_true[:, i, ii, 2] = i / 7
    y_pred[..., 2:10] = 0.1
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    print("loss: ", yolo_loss(tf_y_true, tf_y_pred).numpy())

    print(">>> iou test: same bbox , iou should = 1")
    # the same with previous test case
    # except coordinnate
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    y_true[..., :1] = 1

    y_pred[..., 4:6] = y_pred[..., 8:10] = 0.1
    y_true[..., 3:5] = 0.1
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    Config.BoxPerCell = 1
    print("iou: ", calc_iou(tf.reshape(tf_y_true[..., 1:5], (2, 7, 7, 1, 4)),
                            tf.reshape(tf_y_pred[..., 2:6], (2, 7, 7, 1, 4))).numpy().tolist())

    print(">>> iou test: different bbox, iou should = 81/(200-81) = 0.68")
    # the same with previous test case
    # except coordinnate
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    y_true[..., :1] = 1

    y_pred[..., 2:4] = y_pred[..., 6:8] = 0.01
    y_pred[..., 4:6] = y_pred[..., 8:10] = 0.1
    y_true[..., 3:5] = 0.1
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    Config.BoxPerCell = 1
    print("iou: ", calc_iou(tf.reshape(tf_y_true[..., 1:5], (2, 7, 7, 1, 4)),
                            tf.reshape(tf_y_pred[..., 2:6], (2, 7, 7, 1, 4))).numpy().tolist())

    # print(">>> iou test - same bbox")
    # y_cellbase = np.random.uniform(0, 1, size=[2, 7, 7, 2, 4])
    # x_cellbase = np.random.uniform(0, 1, size=[2, 7, 7, 2, 4])
    # print("iou x: ", x_cellbase, "iou y: ", y_cellbase)
    # print(calc_iou(y_cellbase, x_cellbase).numpy())
