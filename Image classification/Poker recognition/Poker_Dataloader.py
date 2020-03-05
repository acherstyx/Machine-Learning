import mxnet as mx
import mxnet.gluon.data as gdata
import os
import cv2.cv2 as cv2

STYLES = ["Clubs", "Diamonds", "Hearts", "Spades"]
INDEX_LIMIT = 300
OFF_SET = 500


def all_path(path):
    result = []  # 所有的文件

    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            result.append(apath)
    # print(result)
    return result


def Create_dataloader_color(path_to_dataset, train_batch_size, test_batch_size,
                            shuffle=True, show_process=False, OFF_SET=OFF_SET):
    image_files = all_path(path_to_dataset)

    images = []
    labels = []
    counter = 0

    for number in range(13):
        for index in range(INDEX_LIMIT):
            for i, style in enumerate(STYLES):
                file_name = "./.dataset/" + style + "/" + str(number) + "_" + str(index + OFF_SET) + ".jpg"
                images.append(cv2.imread(file_name, 1))
                labels.append(i)
        counter += 1
        print("{done:2.0f}% ".format(done=counter / 13 * 100), end="")
    OFF_SET += INDEX_LIMIT + 1
    for number in range(13):
        for index in range(40):
            for i, style in enumerate(STYLES):
                file_name = "./.dataset/" + style + "/" + str(number) + "_" + str(index + OFF_SET) + ".jpg"
                images.append(cv2.imread(file_name, 1))
                labels.append(i)
    print("")

    data_set_train = gdata.ArrayDataset(images[:-2080], labels[:-2080])
    data_set_test = gdata.ArrayDataset(images[-2080:], labels[-2080:])
    train_dataloader = gdata.DataLoader(data_set_train, train_batch_size, shuffle=True)
    test_dataloader = gdata.DataLoader(data_set_test, test_batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def Create_dataloader_number(path_to_dataset, train_batch_size, test_batch_size,
                             shuffle=True, show_process=False, OFF_SET=OFF_SET):
    image_files = all_path(path_to_dataset)

    images = []
    labels = []
    counter = 0

    for number in range(13):
        for index in range(INDEX_LIMIT):
            for i, style in enumerate(STYLES):
                file_name = "./.dataset/" + style + "/" + str(number) + "_" + str(index + OFF_SET) + ".jpg"
                images.append(cv2.imread(file_name, 1))
                labels.append(number)
        counter += 1
        print("{done:2.0f}% ".format(done=counter / 13 * 100), end="")
    OFF_SET += INDEX_LIMIT + 1
    for number in range(13):
        for index in range(40):
            for i, style in enumerate(STYLES):
                file_name = "./.dataset/" + style + "/" + str(number) + "_" + str(index + OFF_SET) + ".jpg"
                images.append(cv2.imread(file_name, 1))
                labels.append(number)
    print("")

    data_set_train = gdata.ArrayDataset(images[:-2080], labels[:-2080])
    data_set_test = gdata.ArrayDataset(images[-2080:], labels[-2080:])
    train_dataloader = gdata.DataLoader(data_set_train, train_batch_size, shuffle=True)
    test_dataloader = gdata.DataLoader(data_set_test, test_batch_size, shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    print("Loading image... ", end="")
    train, test = Create_dataloader_number(".dataset", 4, 10, show_process=False)
    for image, index in train:
        print(index+1)
        cv2.imshow("name", image[0].asnumpy())
        cv2.waitKey()
