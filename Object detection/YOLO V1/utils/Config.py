# pascal voc path
import numpy as np
from datetime import datetime

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

# label format
LabelBoxInfoIndex = [0, 4]
LabelHasObjIndex = 4
LabelClassIndex = 5

# data save
LoadSavedData = True
TrainSavePath = "./.data/train_data.pkl"
ValSavePath = "./.data/val_data.pkl"

# offset
Offset = np.array([np.arange(CellSize)] * CellSize * BoxPerCell)
Offset = np.reshape(Offset, (BoxPerCell, CellSize, CellSize))
Offset = np.transpose(Offset, (1, 2, 0))

# predict
HasObjThreshold = 0.5
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

# debug
DebugOutput = False
#   in draw bbox
DebugOutput_ImageShow_Point = True
#   in train
DebugOutput_Confidence = True
#   in loss function
DebugOutput_IOU = False
DebugOutput_ObjectDelta = False
DebugOutput_NoObjectDelta = False
DebugOutput_PredBox = False
DebugOutput_loss = False

# weight of loss
LossWeight_Coordinate = 5.0
LossWeight_NoObject = 0.5
LossWeight_Object = 1.0
LossWeight_Classes = 1.0

# train data
TrainPercentage = 0.8
TrainBatchSize = 1
ValBatchSize = 1
Epochs = 5
InitialEpoch = 0

# model setting
ReLU_Slope = 0.1
Dropout_Image = 0.2
Dropout_Output = 0.5

# train super parameters
LearningRate = 0.0001
