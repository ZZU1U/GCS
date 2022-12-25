import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

nameof = input()

col = int(input())
i = 0

fps = int(input())
j = 0

print("Strarting the initializing of", nameof, "u have 5 seconds to get ready.")
print("Just open your camera and show it a gesture!")
for x in range(1, 6):
  print(x)
  time.sleep(1)

print("Less gooo")

os.mkdir('class_'+nameof)

idss = np.empty([fps, 20, 2])

ids = np.empty([20,2])

# Main cycle
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()

    # Error detecting

    if not success:
      print("Ignoring empty camera frame.")
      continue

    # image capture

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:

        # multi you can change to right or left for each hand

      for handLms in results.multi_hand_landmarks:
          ni = 0
          for id, lm in enumerate(handLms.landmark):              # write to ids
              h, w, c = image.shape
              cx, cy = float(lm.x * w), float(lm.y * h)
              if id == 0:
                n = (cx, cy)
              else:
                ids[id-1] = (cx-n[0],cy-n[1])
          ids /= abs(max(ids.min(), ids.max(), key=abs))
          idss[j] = ids
          
          if j == fps - 1:
            i += 1
            j = 0
            print('['+i*10//col*'*'+(col-i)*10//col*' '+']')
            with open('class_'+nameof+'/'+nameof+'_list'+str(i)+'.lst', 'w') as outfile:
              for slice_2d in idss:
                np.savetxt(outfile, slice_2d, fmt="%1.12f")

          else:
            j += 1

          if i == col:
            break

      if i == col:
            break
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()