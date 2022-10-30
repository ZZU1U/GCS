import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    ids = np.matrix([[0, 0]])
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for handLms in results.multi_hand_landmarks:
          for id, lm in enumerate(handLms.landmark):
              h, w, c = image.shape
              cx, cy = int(lm.x * w), int(lm.y * h)
              if id == 0:
                  nule = [cx, cy]
              if id == len(ids) - 1:
                  ids = np.append(ids, [[cx - nule[0], cy - nule[1]]], axis=0)
          ids = np.delete(ids, 0, 0)
          minl = ((ids.item(5,0)-ids.item(0,0))**2+(ids.item(5,1)-ids.item(0,1))**2)**0.5
          for i in range(len(ids)):
              cv2.circle(image, (int(ids.item(i,0)*100/minl)+200,int(ids.item(i,1)*100/minl)+400), 3,(255,255,255), 3)
          print(ids/minl)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()