import os
import tensorflow as tf
import utils.Config as Config
from utils.LoadPascalVOC import PascalVOC
from utils.Model import yolo_model
from utils.ImageProcessing import DrawBoundingBox
import cv2 as cv
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# create data loader
data_set = PascalVOC()
train_iter = data_set.train_generator(Config.TrainBatchSize)
val_iter = data_set.val_generator(Config.ValBatchSize)
# create model
model = yolo_model(show_summary=False)

# train
model.fit_generator(generator=train_iter,
                    # steps_per_epoch=data_set.TrainNum / Config.TrainBatchSize,
                    steps_per_epoch=1000,
                    validation_freq=10,
                    epochs=1,
                    verbose=1)

predict_train_iter = data_set.train_generator(Config.TrainBatchSize)
for image, label in predict_train_iter:
    image = image["input"]
    label = label["output"]
    result = model.predict(image, batch_size=1)
    result[..., 10:] = tf.math.softmax(result[..., 10:])
    result[..., :10] = tf.math.sigmoid(result[..., :10])
    for single_result, current_image, single_pred in zip(result, image, label):
        image_out = DrawBoundingBox(single_result, current_image, Config.BoxPerCell, Threshold=0.51)
        cv.imshow("out", image_out)
        cv.waitKey()
    print(result[..., :2])
