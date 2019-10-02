import cv2.cv2 as cv2

HAARCASCADE_FRONT_FACE = "./.haarcascade/haarcascade_frontalface_default.xml"

if __name__ == "__main__":
    # define a camera
    camera = cv2.VideoCapture(0)
    # define haar cascade
    front_face_cascade = cv2.CascadeClassifier(HAARCASCADE_FRONT_FACE)
    # start detection
    while True:
        _, image_in = camera.read()
        # cvt to gray
        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        # detect and get result
        faces = front_face_cascade.detectMultiScale(gray, 1.15,1)

        # cap front face
        for (x, y, w, h) in faces:
            img = cv2.rectangle(image_in, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # img = cv2.putText(img, "Front face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # show the result
        cv2.imshow("Face detection", image_in)
        cv2.waitKey(1)
