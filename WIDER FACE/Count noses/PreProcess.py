'''
将数据文件预处理成只包含文件名和人数的形式，删去其余数据，来加快读取
'''
def pick_number(file_to_read,file_to_write):
    writer=open(file_to_write,"w")
    with open(file_to_read,"r") as file:
        while True:
            filename=file.readline()
            if not filename:
                writer.close()
                break
            number=file.readline()
            writer.writelines(filename)
            writer.writelines(number)
            for i in range(int(number)):
                file.readline()


if __name__ == "__main__":
    w="./.dataset/wider_face_split/out.txt"
    r="./.dataset/wider_face_split/wider_face_train_bbx_gt.txt"
    pick_number(r,w)