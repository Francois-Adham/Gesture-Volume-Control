import cv2 as cv
import mediapipe as mp


class HandDetector:
    def __init__(self, mode = False, max_hands = 2, confidence = 0.5, tracking_confidence = 0.5) -> None:
        self.mode = mode
        self.confidence = confidence
        self.tracking_confidence = tracking_confidence
        self.max_hands = max_hands

        self.mp_hand =  mp.solutions.hands
        self.hands = self.mp_hand.Hands(self.mode, self.max_hands, self.confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw = False):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if(self.results.multi_hand_landmarks and draw):
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hand.HAND_CONNECTIONS)
        return img
    
    def find_position(self, img, hand_number=0, draw=False):
        landmarks = []
        self.find_hands(img, draw)
        if(self.results.multi_hand_landmarks):
            width, height, channels = img.shape
            for id, loc in enumerate(self.results.multi_hand_landmarks[hand_number].landmark):
                x, y = int(loc.x * height), int(loc.y * width)
                landmarks.append((id, x, y))
                if(draw):
                    if( id % 4 == 0):
                        cv.circle(img, (x, y), 10, [0, 0, 0],-1)
                    elif( id % 4 == 1):
                        cv.circle(img, (x, y), 10, [255, 0, 0], -1)
                    elif( id % 4 == 2):
                        cv.circle(img, (x, y), 10, [0, 255, 0], -1)
                    elif( id % 4 == 3):
                        cv.circle(img, (x, y), 10, [0, 0, 255], -1)
        return landmarks


def main():
    video = cv.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = video.read()
        
        image = detector.find_hands(img)
        lm = detector.find_position(img)
        cv.imshow("Image", image)
        cv.waitKey(1)


if __name__ == "__main__":
    main()


