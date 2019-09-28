import mxnet as mx
import mxnet.gluon.data as gdata
import os
import cv2.cv2 as cv2

STYLES = ["Clubs", "Diamonds", "Hearts", "Spades"]
INDEX_LIMIT = 200

def all_path(path):
    result = []  # 所有的文件

    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            result.append(apath)
    #print(result)
    return result


def Create_dataloader_color(path_to_dataset,train_batch_size,test_batch_size,shuffle=True):
    image_files = all_path(path_to_dataset)

    images = []
    labels = []
    for i,style in enumerate(STYLES):
        for number in range(13):
            for index in range(INDEX_LIMIT):
                file_name = "./.dataset/"+style+"/"+str(number)+"_"+str(index)+".jpg"
                images.append(mx.image.imread(file_name))
                labels.append(i)

    data_set = gdata.ArrayDataset(images,labels)
    train_dataloader = gdata.DataLoader(data_set, train_batch_size, shuffle=True)
    test_dataloader = gdata.DataLoader(data_set, test_batch_size, shuffle=True)
    return train_dataloader,test_dataloader


if __name__ == "__main__":
    train,test = Create_dataloader_color(".dataset",4,10)
    for image,index in train:
        cv2.imshow("name",image[0].asnumpy())
        cv2.waitKey()
        print(index)

