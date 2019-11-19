import numpy as np
import tensorflow as tf
import utils.Config as Config


def calc_iou(box1, box2):
    """
    calculate IoU of 2 bound box
    box1 and box2 have the same shape because pre-process in yolo_loss, don't worry!
    :param box1: [batch_size,cell_size,cell_size,box_per_cell(2),4] (x,y,w,h)
    :param box2: [batch_size,cell_size,cell_size,box_per_cell(2),4] (x,y,w,h)
    return
        iou: [batch_size,cell_size,cell_size,box_pre_cell]
    """
    box1_point = tf.stack([box1[..., 0] - box1[..., 3] / 2.0,
                           box1[..., 1] - box1[..., 2] / 2.0,
                           box1[..., 0] + box1[..., 3] / 2.0,
                           box1[..., 1] + box1[..., 2] / 2.0],
                          axis=-1)
    box2_point = tf.stack([box2[..., 0] - box2[..., 3] / 2.0,
                           box2[..., 1] - box2[..., 2] / 2.0,
                           box2[..., 0] + box2[..., 3] / 2.0,
                           box2[..., 1] + box2[..., 2] / 2.0],
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
    out = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

    return out


def yolo_loss(y_true, y_pred):
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
    pred_bbox_cell_base = tf.stack([pred_bbox_cell_base[..., 0],
                                    pred_bbox_cell_base[..., 1],
                                    tf.sqrt(pred_bbox_cell_base[..., 2]),
                                    tf.sqrt(pred_bbox_cell_base[..., 3])],
                                   axis=-1)

    # TODO: remove debug output
    if Config.DebugOutput_PredBox:
        print(" - pred of box in loss input: ", y_pred[..., :10])

    #  true
    true_classes = y_true[..., 5]  # [_,7,7,1] use index instead of probabilities
    true_confidence = tf.reshape(y_true[..., 0],
                                 (-1, Config.CellSize, Config.CellSize, 1))  # [_,7,7,1]
    true_bbox_image_base = tf.reshape(y_true[..., 1:5],
                                      [-1, Config.CellSize, Config.CellSize, 1, 4])  # [_,7,7,1,4]
    true_bbox_image_base = tf.tile(true_bbox_image_base,
                                   [1, 1, 1, Config.BoxPerCell, 1])  # fit bbox number in prediction

    # get offset
    offset_t = tf.constant(Config.Offset, dtype=tf.float32)
    offset_t = tf.reshape(offset_t, (1, Config.CellSize, Config.CellSize, Config.BoxPerCell))
    try:
        offset_t = tf.tile(offset_t, [batch_size, 1, 1, 1])
    except TypeError:
        pass
    offset = tf.transpose(offset_t, (0, 2, 1, 3))  # transpose of offset

    # Cell-based coordinates -> Image-based coordinates, for IoU
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
        print(" - iou in loss: ", iou_of_pred_and_true)

    # get mask
    obj_mask = tf.reduce_max(iou_of_pred_and_true,
                             axis=3,
                             keepdims=True)  # choose the larger one as the result of prediction
    obj_mask = tf.cast(
        (iou_of_pred_and_true >= obj_mask),
        tf.float32
    ) * true_confidence
    no_obj_mask = 1.0 - obj_mask

    # classify loss
    classes_loss = tf.squeeze(true_confidence, axis=-1) * tf.keras.losses.sparse_categorical_crossentropy(true_classes,
                                                                                                          pred_classes)
    classes_loss = tf.reduce_mean(
        tf.reduce_sum(classes_loss, axis=[1, 2])
    )

    # has object loss
    object_delta = tf.square((iou_of_pred_and_true - pred_confidence) * obj_mask)
    object_loss = tf.reduce_mean(
        tf.reduce_sum(object_delta, axis=[1, 2, 3])
    )

    # TODO: remove debug output
    if Config.DebugOutput_ObjectDelta:
        print(" - object delta in loss: ", object_delta[0])

    # no object loss
    no_obj_delta = tf.square(no_obj_mask * pred_confidence)
    no_obj_loss = tf.reduce_mean(
        tf.reduce_sum(no_obj_delta, axis=[1, 2, 3])
    )

    # TODO: remove debug output
    if Config.DebugOutput_NoObjectDelta:
        print(" - no_object delta in loss: ", no_obj_delta)

    # box loss
    coordinate_mask = tf.expand_dims(obj_mask, 4)
    bbox_delta = tf.square(coordinate_mask * (true_bbox_cell_base - pred_bbox_cell_base))
    bbox_loss = tf.reduce_mean(
        tf.reduce_sum(bbox_delta, axis=[1, 2, 3, 4])
    )

    # TODO: remove debug output
    if Config.DebugOutput_loss:
        print("- obj_loss", object_loss,
              "\n- no_obj_loss", no_obj_loss,
              "\n- bbox_loss", bbox_loss,
              "\n- classes_loss", classes_loss)

    return object_loss * Config.LossWeight_Object + \
           no_obj_loss * Config.LossWeight_NoObject + \
           bbox_loss * Config.LossWeight_Coordinate + \
           classes_loss * Config.LossWeight_Classes


if __name__ == "__main__":
    # y_true = tf.random.normal([2, 7, 7, 6])
    # y_pred = tf.random.normal([2, 7, 7, 30])
    # print(yolo_loss(y_true, y_pred))

    # print(">>> Loss sample")
    # y_true = np.zeros([2, 7, 7, 6])
    # y_pred = np.zeros([2, 7, 7, 30])
    # y_true[0, 0, 0, :] = [1.0, 0.5, 0.5, 0.05, 0.05, 2.0]
    # y_pred[0, 0, 0, :11] = [1000.0, 0.000000000000001,
    #                         0.5, 0.5, 0.05, 0.05,
    #                         0.1, 0.1, 0.1, 0.1,
    #                         1.000]
    # y_true = tf.Variable(y_true, dtype=tf.float32)
    # y_pred = tf.Variable(y_pred, dtype=tf.float32)
    # print("loss ", yolo_loss(y_true, y_pred))

    print(">>> loss test - all zero")
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    print("loss ", yolo_loss(tf_y_true, tf_y_pred))

    print(">>> loss test - pred:conf all=1 true:conf all=0")
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    y_pred[..., :2] = 1
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    print("loss ", yolo_loss(tf_y_true, tf_y_pred))

    print(">>> loss test - pred:conf=0 true:conf=1")
    y_true = np.zeros([2, 7, 7, 6])
    y_pred = np.zeros([2, 7, 7, 30])
    y_true[..., 0] = 1
    tf_y_true = tf.Variable(y_true, dtype=tf.float32)
    tf_y_pred = tf.Variable(y_pred, dtype=tf.float32)
    print("loss ", yolo_loss(tf_y_true, tf_y_pred))

    print(">>> iou test - same bbox")
    y_cellbase = np.random.uniform(0, 1, size=[2, 7, 7, 2, 4])
    x_cellbase = np.random.uniform(0, 1, size=[2, 7, 7, 2, 4])
    print("iou x: ", x_cellbase, "iou y: ", y_cellbase)
    print(calc_iou(y_cellbase, x_cellbase))
