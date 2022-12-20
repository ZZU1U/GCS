import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Main cycle
cap = cv2.VideoCapture(0)
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
          ids = pd.DataFrame(columns=list('XY'))
          for id, lm in enumerate(handLms.landmark):
              # write to ids
              h, w, c = image.shape
              cx, cy = float(lm.x * w), float(lm.y * h)
              idn = pd.DataFrame([[cx,cy]], columns=list('XY'), index=[id])
              ids = pd.concat([ids, idn])

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()