import cv2 
import numpy as np
import time 
from HandTrackingModule import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    cTime = 0 
    pTime = 0 


    while True:
        success, frame = cap.read()

        if not success:
            break 

        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame, (0,0), fx=1.5, fy=1.5)
        frame = detector.findHands(frame)
        lmList = detector.findPositions(frame)    
        fingers, total_fingers = detector.findFinger(lmList)
        if len(fingers) != 0:
            print(f"{fingers} , {total_fingers}")
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 3)

        cv2.imshow("Hands detector", frame)

        if cv2.waitKey(1) == ord("q"):
            break 
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
