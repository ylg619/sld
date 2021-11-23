import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands = detector.findHands(img) #With Draw
    #hands, img = detector.findHands(img, flipType=True) #No Draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] #List of 21 Landmarks points
        bbox1 = hand1["bbox"] # Bounding Box info x, y, w, h
        centerPoint1 = hand1["center"] # center of hand cx, cy
        handType1 = hand1["type"] # Hand Type Left or Right

        print(len(lmList1), lmList1)
        #print(bbox1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
