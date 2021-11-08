import cv2
import mediapipe as mp
import time

class handDetector():
    #Detection model initialization
    def __init__(self, mode=False, maxHands = 2, modelCom = 1, detectionCon = 0.5, trackCon=0.5 ):
        # static_image_mode = False,
        # max_num_hands = 2,
        # model_complexity = 1,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5

        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #Mediapipe hand detection
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelCom,self.detectionCon,self.trackCon) #create an object for detect hands
        self.mpDraw = mp.solutions.drawing_utils #object for draw hand landmarks

    #hand detector method
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image to RGB
        self.results = self.hands.process(imgRGB) #call hands object

        #retrived and visualized hands info
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                #Check weather draw is True
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS) #visualized the hand landmarks

        return img
                

    #land mark position values extract
    def findPosition(self, img, handNo=0, draw=True):
        lmList = [] #landmark list
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
             
            for id, lm in enumerate(myHand.landmark): #landmarks details/locations
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) #position according to display pixles
                lmList.append([id,cx, cy])

                if draw:
                    cv2.circle(img, (cx,cy), 7, (0,0,255), cv2.FILLED)

        return lmList


def main():
    #define the webcam
    cap = cv2.VideoCapture(0)

    detector = handDetector() #call hand detector class

    pTime = 0 #varibles for calculate FPS
    cTime = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img) #call the method
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
    

if __name__ == "__main__":
    main()