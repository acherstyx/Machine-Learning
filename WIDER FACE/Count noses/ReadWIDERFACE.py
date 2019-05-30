import numpy as np 
import cv2.cv2 as cv 
import imutils
import os

class WIDER_FACE_Dataset:
    size=None
    Files=[]
    Numbers=[]
    #image=[]
    __batch_read_pointer=0
    __batch_read_sampleamount=0
    __with_preprocess=False
    def __image_reader(self,filepath):
        image=cv.imread(filepath)
        try:
            image=cv.resize(image,(self.size[1],self.size[0]),interpolation=cv.INTER_CUBIC)
        except TypeError:
            pass
        # 预处理
        # 暂无
        return image 
    def __init__(self,label_file_path,image_root_path,size=None,with_preprocess=False):
        with open(label_file_path) as file:
            while True:
                filename=file.readline()[:-1]
                number=file.readline()
                if not filename or not number:
                    break
                else:
                    self.Files.append(filename)
                    self.Numbers.append(float(number))
        self.__batch_read_sampleamount=len(self.Files)
        self.__image_root_path=image_root_path
        self.size=size
        if not isinstance(with_preprocess,bool):
            raise(TypeError)
        else:
            self.__with_preprocess=with_preprocess
        #for file in self.Files:
        #    file=os.path.join(image_root_path,file)
        #    self.image.append(self.__image_reader(file))
    def nextbatch(self,batch_size):
        images=[]
        batch_files=self.Files[
            self.__batch_read_pointer%self.__batch_read_sampleamount:
            min(self.__batch_read_sampleamount,(self.__batch_read_pointer+batch_size)%self.__batch_read_sampleamount)]
        batch_label=self.Numbers[
            self.__batch_read_pointer%self.__batch_read_sampleamount:
            min(self.__batch_read_sampleamount,(self.__batch_read_pointer+batch_size)%self.__batch_read_sampleamount)]
        for file in batch_files:
            file=os.path.join(self.__image_root_path,file)
            images.append(self.__image_reader(file))
        self.__batch_read_pointer+=batch_size
        return images,batch_label
    def testbatch(self,batch_size):
        self.__batch_read_sampleamount=100
        return self.nextbatch(batch_size)


if __name__ == "__main__":
    a=WIDER_FACE_Dataset("./.dataset/wider_face_split/out.txt","./.dataset/WIDER_train/images/",size=[360,480])
    print(a.Files)
    print(a.Numbers)
    print(a.nextbatch(10))
    image,label=a.nextbatch(10)
    cv.imshow(str(label[5]),image[5])
    cv.waitKey()
    pass