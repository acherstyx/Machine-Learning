import os
import tensorflow as tf
from utils.YoloLoss import yolo_loss
import numpy as np
from utils.LoadPascalVOC import PascalVOC
from utils.ImageProcessing import draw_bounding_box
import cv2 as cv
import utils.Config as Config
from utils.MyMetrics import RealTimeIOU
from utils.Model import yolo_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# create data loader
data_set = PascalVOC()
# restore entire model from saved data
model = yolo_model("ORIGINAL")
mymetric = RealTimeIOU()
model = tf.keras.models.load_model(Config.RestoreWeightPath_Test,
                                   compile=False)
print("Model restored from", Config.RestoreWeightPath_Test)

# predict
predict_data_iter = data_set.train_generator(batch_size=1)
count = 0
User_Threshold = input("Define current threshold: ")
for image, label in predict_data_iter:
    image = image["input"]
    label = label["output"]
    result = model.predict(image, batch_size=1)
    for single_pred_result, single_true_result, current_image in zip(result, label, image):
        image_out_pred = draw_bounding_box(single_pred_result,
                                           current_image.copy(),
                                           Config.BoxPerCell,
                                           is_logits=True,
                                           base_coordinate_of_xy="CELL",
                                           threshold=float(User_Threshold))
        image_out_true = draw_bounding_box(single_true_result,
                                           current_image.copy(),
                                           1,
                                           is_logits=False,
                                           base_coordinate_of_xy="IMAGE",
                                           color=(0, 0, 255))
        image_out = np.hstack((image_out_true, image_out_pred))
        if Config.DebugOutput_Confidence:
            print(result[..., :Config.BoxPerCell])

        cv.imshow("out", image_out)
        cv.waitKey()

    count += 1
    if count == 1000:
        break
