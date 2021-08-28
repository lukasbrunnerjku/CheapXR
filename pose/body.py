import cv2
import mediapipe as mp
import time


class BodyDetector():
    # https://google.github.io/mediapipe/solutions/pose.html

    def __init__(self):

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def findPose(self, img, draw=True):

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img):

        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
               
        return self.lmList


def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = BodyDetector()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            continue

        img = detector.findPose(img)

        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[14])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("", img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()


if __name__ == "__main__":
    main()