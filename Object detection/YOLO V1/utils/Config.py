# pascal voc path
import numpy as np

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
LoadSavedData = False
TrainSavePath = "./.data/train_data.pkl"
ValSavePath = "./.data/val_data.pkl"

# offset
Offset = np.array([np.arange(CellSize)] * CellSize * BoxPerCell)
Offset = np.reshape(Offset, (BoxPerCell, CellSize, CellSize))
Offset = np.transpose(Offset, (1, 2, 0))

# train
TrainPercentage = 0.8
ImageDropoutRate = 0.2
TrainBatchSize = 1
ValBatchSize = 1

# predict
HasObjThreshold = 0.8

# debug
DebugOutput = True
