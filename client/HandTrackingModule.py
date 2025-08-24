import cv2 
import mediapipe as mp 

class HandDetector:
    def __init__(self) -> None:
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        

    def findHands(self, frame, draw:bool = True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return frame

    def findPositions(self, frame, hand_no=0, draw = True ):
        lmList = [] 

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(myHand.landmark):
                height, width, _ = frame.shape 
                cx,cy = int(lm.x * width), int(lm.y * height)

                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx,cy), 7, (255,0,0), cv2.FILLED)

        return lmList

    def findFinger(self, lmList, fingerIds = [4, 8, 12, 16, 20]):
        fingers = []
        if len(lmList) == 0:
            return [], 0

        for id in range(0,5):
            if lmList[fingerIds[id]][2] < lmList[fingerIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            if lmList[fingerIds[0]][1] < lmList[fingerIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

        return fingers,total_fingers
