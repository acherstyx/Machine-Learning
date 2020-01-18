import tensorflow as tf
from utils.YoloLoss import calc_iou
import utils.Config as Config


class RealTimeIOU(tf.keras.metrics.Metric):

    def __init__(self, name="iou", **kwargs):
        super(RealTimeIOU, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name="iou", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        batch_size = y_true.get_shape()[0]
        pred_bbox_cell_base = tf.reshape(y_pred[..., 2:10],
                                         [-1, Config.CellSize, Config.CellSize, Config.BoxPerCell, 4])  # [_,7,7,2,4]

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

        # get iou of box
        iou_of_pred_and_true = calc_iou(pred_bbox_image_base, true_bbox_image_base)

        # get mask
        obj_mask = tf.reduce_max(iou_of_pred_and_true,
                                 axis=3,
                                 keepdims=True)  # choose the larger one as the result of prediction
        obj_mask = tf.cast(
            (iou_of_pred_and_true >= obj_mask),
            tf.float32
        ) * true_confidence

        self.iou.assign(tf.reduce_sum(iou_of_pred_and_true * obj_mask) / tf.reduce_sum(obj_mask))

    def result(self):
        return self.iou
