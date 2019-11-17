import os
import tensorflow as tf
import utils.Config as Config
from utils.LoadPascalVOC import PascalVOC
from utils.Model import yolo_model
from utils.ImageProcessing import DrawBoundingBox
import cv2 as cv
from tensorflow.keras.callbacks import TensorBoard

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# create data loader
data_set = PascalVOC()
train_iter = data_set.train_generator(Config.TrainBatchSize)
val_iter = data_set.val_generator(Config.ValBatchSize)
# create model
model = yolo_model(show_summary=False)
# Tensorboard support
tensorboard = TensorBoard(log_dir=".log/{}".format(Config.TIMESTAMP))

while True:
    # train
    try:
        model.load_weights("./.save/my_model.h5")
    except OSError:
        print(">>> Can't load model!")
        pass

    model.fit_generator(generator=train_iter,
                        steps_per_epoch=data_set.TrainNum / Config.TrainBatchSize,
                        # steps_per_epoch=5,
                        epochs=10,
                        # callbacks=[tensorboard],
                        verbose=1)
    if input(">>> Finished. save ?(y/n): ") == "y":
        model.save("./.save/my_model.h5")
        print(">>> saved.")
    else:
        print(">>> abort.")

    predict_train_iter = data_set.train_generator(Config.TrainBatchSize)
    count = 0
    User_Threshold = input("Define current threshold: ")
    for image, label in predict_train_iter:
        image = image["input"]
        label = label["output"]
        result = model.predict(image, batch_size=1)
        for single_pred_result, single_true_result, current_image in zip(result, label, image):
            image_out = DrawBoundingBox(single_pred_result, current_image, Config.BoxPerCell,
                                        is_logits=True,
                                        base_coordinate="CELL",
                                        threshold=float(User_Threshold))
            image_out = DrawBoundingBox(single_true_result, image_out, 1,
                                        is_logits=False,
                                        base_coordinate="IMAGE",
                                        color=(0, 0, 255))
            # image_out = cv.resize(image_out, (900, 900))
            if Config.DebugOutput_Confidence:
                print(result[..., :Config.BoxPerCell])

            cv.imshow("out", image_out)
            cv.waitKey()

        count += 1
        if count == 100:
            break
