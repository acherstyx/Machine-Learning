import cv2.cv2 as cv2
from warnings import warn
import imutils


class ImgReader:
    fps = None
    size = None
    __image_capture = None

    def __init__(self, source=0, resize=False):
        self.__image_capture = cv2.VideoCapture(source)
        if not self.__image_capture.isOpened():
            raise ValueError
        self.size = (int(self.__image_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(self.__image_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.fps = self.__image_capture.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30

    def read(self, with_pre_processing=True):
        successful, frame = self.__image_capture.read()
        if not successful:
            warn("Failed to get fram from video.")
            raise RuntimeError
        if not isinstance(with_pre_processing, bool):
            warn("Value seems to be invalid. Skip pre-processing.")
            with_pre_processing = False
        # 预处理步骤
        if with_pre_processing:
            frame = self.__image_pre_processing(frame)
        return successful, frame

    def __image_pre_processing(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度
        frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
        return frame

    def show_sample(self, with_pre_processing=False):
        successful, image = self.read(with_pre_processing)
        if successful:
            if with_pre_processing:
                cv2.imshow("Sample with pre-processing(press any key to continue)", image)
            else:
                cv2.imshow("Sample without pre-processing(press any key to continue)", image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            warn("Can't read image, Skip imshow.")

    def recorder(self, file_path, time_in_sec, with_pre_processing=False):
        if file_path[-4:] != ".avi":
            raise ValueError
        videoWriter = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, self.size)

        success, frame = self.read(with_pre_processing)
        frame_left = time_in_sec * self.fps
        while frame_left > 0:
            if success:
                videoWriter.write(frame)
            success, frame = self.read(with_pre_processing)
            frame_left -= 1


if __name__ == "__main__":
    a = ImgReader("./Recognition/00122.MTS")
    _, image = a.read()
    a.show_sample(True)
    a.show_sample(False)

    a.recorder("./sample.avi", 5)
