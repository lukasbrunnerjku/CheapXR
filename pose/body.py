import cv2
import mediapipe as mp
from typing import Tuple
import numpy as np


class BodyDetector:
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

    def __call__(self, img, flip=True) -> Tuple[bool, np.ndarray]:

        if flip:
            img = cv2.flip(img, 1)
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.pose.process(imgRGB)
        success = self.results.pose_landmarks is not None
        return success, img

    def findPose3D(self, draw=True, parse=True):

        if draw:
            self.mpDraw.plot_landmarks(
                self.results.pose_world_landmarks, 
                self.mpPose.POSE_CONNECTIONS
            )

        if parse:
            keypoints = self.results.pose_world_landmarks.landmark
            return [[kp.x, kp.y, kp.z] for kp in keypoints]

    def findPose2D(self, img, draw=True, parse=True):

        if draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS)

        if parse:
            self.lmList = []
            
            for lm in self.results.pose_landmarks.landmark:
                h, w, _ = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                self.lmList.append([x, y])
            return self.lmList

