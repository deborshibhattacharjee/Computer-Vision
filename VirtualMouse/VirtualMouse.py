import cv2
import numpy as np
import time
import autopy
import pyautogui
import HandTrackingModule as HTM

#################################
wCam, hCam = 640, 480
frameR = 150  # Frame reduction
smoothening = 6
#################################

pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HTM.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    # Find Hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        print(lmList)

        # Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Only Index Finger : Moving Mode
        if fingers[1] and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # Smoothen the values
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            # Move mouse
            autopy.mouse.move(cLocX, cLocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY

            if fingers[1] and fingers[4]:
                pyautogui.scroll(-120)

            if fingers[1] and fingers[0]:
                pyautogui.scroll(120)

        # Both index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, lineinfo = detector.findDistance(8, 12, img, r=10)

            # Click mouse if distance is small
            if length < 15:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 10, (0, 255, 0),
                           cv2.FILLED)
                pyautogui.click(clicks=1, interval=0.20)

    # Check Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 3)
    # Display results
    cv2.imshow("Image", img)
    cv2.waitKey(1)

