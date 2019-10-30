# pascal voc path
ImagePath = "./VOCdevkit/VOC2012/JPEGImages"
AnnotationsPath = "./VOCdevkit/VOC2012/Annotations/"

# image segmentation setting
ImageSize = 448
CellSize = 7

# classes info
Classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
ClassesNum = len(Classes)
ClassesDict = dict(zip(Classes, [i for i in range(ClassesNum)]))

TrainPercentage = 0.8

# data save
TrainSavePath = "./.data/train_data.pkl"
ValSavePath = "./.data/val_data.pkl"