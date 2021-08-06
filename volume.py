import cv2 as cv
from subprocess import call
from hand_module import HandDetector
import math

width, height = 1280, 720
maximum, minimum = 300, 50

webcam = cv.VideoCapture(0)
webcam.set(3, width)
webcam.set(4, height)

detector = HandDetector(confidence=0.8)

while True:
    success, img = webcam.read()
    landmarks = detector.find_position(img)

    if len(landmarks) != 0:
        
        # tip coordinates
        x1, y1 = landmarks[4][1], landmarks[4][2]
        x2, y2 = landmarks[8][1], landmarks[8][2]
        
        
        # # start
        # cv.circle(img, (x1, y1), 5, [255, 0, 0],-1)
        # # end
        # cv.circle(img, (x2, y2), 5, [255, 0, 0],-1)
        # # center
        # cv.circle(img, ((x1 + x2) // 2, (y1 +y2) // 2), 5, [255, 0, 0],-1)
        # # line
        # cv.line(img, (x1, y1), (x2, y2), [255, 0, 0], 2)


        # distance between fingers
        length = math.hypot(x1 - x2, y1 - y2)

        percentage = 50

        if length < minimum:
            percentage = 0
        elif length > maximum:
            percentage = 100
        else:
            percentage = (length * 0.4) - 20

        call(["amixer", "-D", "pulse", "sset", "Master", str(percentage)+"%"])

    cv.imshow("", img)
    cv.waitKey(1)