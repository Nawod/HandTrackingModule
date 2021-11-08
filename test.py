import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

#define the webcam
cap = cv2.VideoCapture(0)

detector = htm.handDetector() #call hand detector class

pTime = 0 #varibles for calculate FPS
cTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img, True) #call the method
    lmList = detector.findPosition(img)
    
    if len(lmList) !=0: #check weather any hand detected
        print(lmList[4])

    #Calculate the FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN , 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)