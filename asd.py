import cv2
import mediapipe as mp
import os
import numpy as np
from tensorflow import keras

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

idss = np.empty([40, 20, 2])

l = ['hand_l', 'hand_r', 'okay_l', 'okay_r', 'two_on_l', 'two_on_r', 'fist_l', 'fist_r']

ids = np.empty([20, 2])

model = keras.models.load_model('model')

fps = 10

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

        idss[:-1] = idss[1:]
        idss[-1] = ids

        if not np.array_equal(idss[0], np.zeros([20, 2])):
          if fps != 0:
            fps -= 1
          else:
            fps = 10
            gest = model.predict(idss.reshape(1,40,20,2), verbose=0)
            print(l[np.argmax(gest)], np.max(gest))
            

    cv2.imshow('GSC', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()