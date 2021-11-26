import cv2
from handdetector import HandDetector
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #With Draw
    #hands, img = detector.findHands(img, flipType=True) #No Draw
    copy = img.copy()
    ROI_number = 0


    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] #List of 21 Landmarks points
        bbox1 = hand1["bbox"] # Bounding Box info x, y, w, h
        x, y, w, h = bbox1
        #centerPoint1 = hand1["center"] # center of hand cx, cy
        handType1 = hand1["type"] # Hand Type Left or Right
        #print(len(lmList1), lmList1)
        ROI = img[y:y+h, x:x + w]
        ROI_number += 1


    cv2.imshow("Image", img)
    cv2.waitKey(1)
