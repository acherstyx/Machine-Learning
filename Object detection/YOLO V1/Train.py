import os
import tensorflow as tf
import utils.Config as Config
from utils.LoadPascalVOC import PascalVOC
from utils.Model import yolo_model
from utils.ImageProcessing import DrawBoundingBox
import cv2 as cv
from tensorflow.keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Is there a GPU available: ", tf.test.is_gpu_available())

# create data loader
data_set = PascalVOC()
# create model
model = yolo_model(model_type="INCEPTION V3 KERAS", show_summary=False)
# Tensorboard support
log_dir = os.path.join(
    ".log",
    Config.TIMESTAMP,
)
tensorboard = TensorBoard(log_dir=log_dir)

while True:
    # load model
    try:
        model.load_weights("./.save/my_model.h5")
        print("Model restored.")
    except OSError:
        print(">>> Can't load model!")
        pass
    except ValueError:
        if input("Model don't match, continue?(y/n): ") != "y":
            exit()

    # TODO: Add Test.py
    # data generator
    train_iter = data_set.train_generator(Config.TrainBatchSize)
    val_iter = data_set.val_generator(Config.ValBatchSize)
    # train
    model.fit_generator(generator=train_iter,
                        steps_per_epoch=data_set.TrainNum / Config.TrainBatchSize,
                        # steps_per_epoch=5,
                        epochs=Config.Epochs,
                        validation_data=val_iter,
                        validation_steps=500,
                        validation_freq=1,
                        callbacks=[tensorboard],
                        verbose=1,
                        initial_epoch=Config.InitialEpoch)

    # save
    if input(">>> Finished. save ?(y/n): ") == "y":
        model.save("./.save/my_model.h5")
        print(">>> saved.")
    else:
        print(">>> abort.")

    # predict
    predict_data_iter = data_set.train_generator(batch_size=1)
    count = 0
    User_Threshold = input("Define current threshold: ")
    for image, label in predict_data_iter:
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
            image_out = cv.resize(image_out, (600, 600))
            if Config.DebugOutput_Confidence:
                print(result[..., :Config.BoxPerCell])

            cv.imshow("out", image_out)
            cv.waitKey()

        count += 1
        if count == 1000:
            break
