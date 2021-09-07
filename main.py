import cv2
import numpy as np
import sys
import time
import zmq

sys.path.append('./pose')
from body import BodyDetector


if __name__ == '__main__':

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = BodyDetector()

    while cap.isOpened():

        msg = socket.recv()

        keypoints = None
        while keypoints is None:
            success, img = cap.read()

            if not success:
                continue

            keypoints = detector.findPose3D(img, draw=False)

        keypoints = [[kp.x, kp.y, kp.z] for kp in keypoints]
        socket.send_pyobj(keypoints)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        print(f'fps: {fps:.1f}   ', end='\r')

        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()