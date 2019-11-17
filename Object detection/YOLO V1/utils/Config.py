# pascal voc path
import numpy as np
from datetime import datetime

ImagePath = "./.VOCdevkit/VOC2012/JPEGImages"
AnnotationsPath = "./.VOCdevkit/VOC2012/Annotations/"

# image segmentation setting
ImageSize = 299
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
DebugOutput_ImageShow_Point = True
DebugOutput_Confidence = True
#   in loss function
DebugOutput_IOU = True
DebugOutput_ObjectDelta = True
DebugOutput_NoObjectDelta = True
DebugOutput_PredBox = False
DebugOutput_loss = False
# train data
TrainPercentage = 0.8
ImageDropoutRate = 0.2
TrainBatchSize = 1
ValBatchSize = 1

# train super parameters
LearningRate = 0.001
