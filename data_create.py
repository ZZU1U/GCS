import cv2
import mediapipe as mp
import pyautogui as pygui
import numpy as np
import time

gestures = ["Раскрытая,порозень_пальцы","Раскрыта,пальцы_вместе","Кулак","Указательный_палец_вверх","Указательный_палец_вперед","Сведенные_пальцы","Большой_палец_вверх"]

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
qw = pygui.size()


ids = np.matrix([[]])
zeroid = np.array([0,0])


mp.MAX_NUM_HANDS = 1


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    er = img.shape
    dx = qw[0]/er[0]
    dy = qw[1]/er[1]
    i = 0
    if results.multi_hand_landmarks:
        for gest in gestures:
            print("Show gesture: ",gest)
            time.sleep(1)
            print("In 3")
            time.sleep(1)
            print("2")
            time.sleep(1)
            print("1")
            time.sleep(1)

            ids = np.matrix([[0,0]])

            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    if id == 0:
                        nule = [cx,cy]
                    if id == len(ids)-1:
                        ids = np.append(ids,[[ cx-nule[0], cy-nule[1] ]],axis=0)
            ids = np.delete(ids, 0, 0)
            minl = ((ids.item(18, 0) - ids.item(19, 0)) ** 2 + (ids.item(18, 1) - ids.item(19, 1)) ** 2) ** 0.5
            ids = ids/minl
            np.savetxt(gest+str(i)+'.txt', ids)
            if gest == gestures[-1]:
                i+=1
        cv2.waitKey(1)