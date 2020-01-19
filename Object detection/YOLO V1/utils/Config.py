# pascal voc path
import numpy as np
from datetime import datetime
import os

# directory of data set
ImagePath = "./.VOCdevkit/VOC2012/JPEGImages"
AnnotationsPath = "./.VOCdevkit/VOC2012/Annotations/"

# image segmentation setting
ImageSize = 448
CellSize = 7
BoxPerCell = 2

# classes info
Classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
ClassesNum = len(Classes)
ClassesDict = dict(zip(Classes, [i for i in range(ClassesNum)]))

# data save path
LoadSavedData = True
TrainSavePath = "./.data/train_data.pkl"
ValSavePath = "./.data/val_data.pkl"

# offset
Offset = np.array([np.arange(CellSize)] * CellSize * BoxPerCell)
Offset = np.reshape(Offset, (BoxPerCell, CellSize, CellSize))
Offset = np.transpose(Offset, (1, 2, 0))

# weight of loss
LossWeight_Coordinate = 5.0
LossWeight_NoObject = 0.5
LossWeight_Object = 1.0
LossWeight_Classes = 2.0

# network model setting
ReLU_Slope = 0.1
Dropout_Image = 0.2
Dropout_Output = 0.5

# train super parameters
TrainNameStamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# train data setting
TrainPercentage = 0.8
TrainBatchSize = 8
ValBatchSize = 8


# learning rate scheduler
def scheduler(epoch):
    if epoch == 0:
        return 0.0005
    elif epoch < 5:
        return 0.0005 + 0.0001 * (epoch + 1)
    elif epoch < 75:
        return 0.001
    elif 30 < epoch < 105:
        return 0.0001
    else:
        return 0.00001


# epoch setting
Epoch_Initial = 0
Epochs = 135

# predict
HasObjThreshold = 0.2

# debug
DebugOutput = False
#   in draw bbox
DebugOutput_ImageShow_Point = True
DebugOutput_ImageShow_Logits = True
#   in train
DebugOutput_Confidence = False
#   in loss function
DebugOutput_IOU = False
DebugOutput_ObjectDelta = False
DebugOutput_NoObjectDelta = False
DebugOutput_PredBox = False
DebugOutput_loss = False

# weight to restore
RestoreWeightPath = ".log/model.h5"
RestoreWeightPath_Test = ".log/model.h5"
# log directory
LogDirectory_Root = os.path.join(".", ".log", TrainNameStamp)
LogDirectory_Checkpoint = os.path.join(LogDirectory_Root, "checkpoint", "model_{epoch}.h5")
LogDirectory_Tensorboard = os.path.join(LogDirectory_Root, "tensorboard")
try:
    os.makedirs(os.path.join(LogDirectory_Root, "checkpoint"))
except FileExistsError:
    pass
