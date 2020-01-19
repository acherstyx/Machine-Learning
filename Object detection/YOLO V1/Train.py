import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import utils.Config as Config
from utils.LoadPascalVOC import PascalVOC
from utils.Model import yolo_model



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Is there a GPU available:", tf.test.is_gpu_available())
print("NameStamp:", Config.TrainNameStamp)

# create data loader
data_set = PascalVOC()
# create model
model = yolo_model(model_type="ORIGINAL", show_summary=False)

# tensorboard support
Callback_Tensorboard = TensorBoard(log_dir=Config.LogDirectory_Tensorboard,
                                   update_freq="batch",
                                   write_graph=True,
                                   write_images=True,
                                   histogram_freq=1
                                   )

# checkpoint support
Callback_Checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=Config.LogDirectory_Checkpoint,
                                                         save_best_only=False,
                                                         monitor="train_loss",
                                                         verbose=0,
                                                         )
# learning rate scheduler
Callback_Scheduler = tf.keras.callbacks.LearningRateScheduler(schedule=Config.scheduler,
                                                              verbose=1)

# load model
try:
    model.load_weights(Config.RestoreWeightPath)
    print("Model restored from", Config.RestoreWeightPath)
except OSError:
    print(">>> Can't load model!")
    pass
except ValueError:
    if input("Model don't match, continue?(y/n): ") != "y":
        exit()

# data generator
train_iter = data_set.train_generator(Config.TrainBatchSize)
val_iter = data_set.valid_generator(Config.ValBatchSize)
# train
model.fit_generator(generator=train_iter,
                    steps_per_epoch=data_set.TrainNum / Config.TrainBatchSize,
                    # steps_per_epoch=5,
                    epochs=Config.Epochs,
                    validation_data=val_iter,
                    validation_steps=data_set.ValNum / Config.ValBatchSize,
                    validation_freq=1,
                    callbacks=[Callback_Tensorboard,
                               Callback_Checkpoint,
                               Callback_Scheduler,
                               ],
                    verbose=1,
                    initial_epoch=Config.Epoch_Initial)
