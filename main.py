import cv2
import numpy as np
import sys
import time
import zmq

sys.path.append('./pose')
from body import BodyDetector


class EMA:

    def __init__(self, num, fac):
        self.num = num
        self.fac = fac 
        self.vs = []

    def add(self, v):
        if len(self.vs) == self.num:
            self.vs.pop(0)
        self.vs.append(v)

    def __call__(self):
        num = len(self.vs)
        return sum([self.vs[i] * self.fac**i for i in range(num)]) / num



text = """
0 - nose
1 - left eye inner
2 - left eye
3 - left eye outer
4 - right eye inner
5 - right eye
6 - right eye outer
7 - left ear
8 - right ear
9 - mouth left
10 - mouth right
11 - left shoulder
12 - right shoulder
13 - left elbow
14 - right elbow
15 - left wrist
16 - right wrist
17 - left pinky
18 - right pinky
19 - left index
20 - right index
21 - left thumb
22 - right thumb
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
29 - left heel
30 - right heel
31 - left foot index
32 - right foot index
"""

table = {}
for line in text.split('\n'):
    if len(line) > 0:
        num, desc = line.split(' - ')
        table[int(num)] = desc


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default='', help=text)
    args = parser.parse_args()
    if len(args.indices) > 0:
        args.indices = list(map(int, args.indices.split(' ')))

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    pTime = 0
    cTime = 0
    ema_fps = EMA(num=20, fac=0.9)
    color = (255, 0, 255)
    font = cv2.FONT_HERSHEY_PLAIN
    cap = cv2.VideoCapture(0)
    detector = BodyDetector()

    while cap.isOpened():

        msg = socket.recv()

        success = False
        while not success:
            success, img = cap.read()
            if success:
                success, img = detector(img, flip=False)

        socket.send_pyobj(img)

        # keypoints = detector.findPose3D(draw=False, parse=True)
        # socket.send_pyobj(keypoints)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        ema_fps.add(fps)

        if len(args.indices) > 0:
            lmList = detector.findPose2D(img, draw=True, parse=True)

            for idx in args.indices:
                x, y = lmList[idx]
                img = cv2.circle(img, (x, y), 6, color, -1)
                cv2.putText(img, table[idx], (x-8, y-8), font, 3, color, 3)
                cv2.putText(img, str(int(ema_fps())), (10, 70), font, 3, color, 3)
                cv2.imshow("", img)
        else:
            print(f'fps: {int(ema_fps())}   ', end='\r')

        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()