import cv2
import mediapipe as mp
import numpy as np

lastids = np.array()


print("Press esc to escape")

# define gesture type

def gesture_type(s):
    # detect by x

    if s.item(5, 0)-s.item(17, 0) < -30:
        print("reverse", end=" ")

    elif s.item(5, 0)-s.item(17, 0) > 30:
        print("just", end=" ")
    else:
        print("side", end=" ")

    # detect by y

    if abs(s.item(5, 1) - s.item(1, 1)) > 80:
        print("не в камеру")
    else:
        print("в камеру")


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Main cycle
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
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
        ids = np.matrix([[0, 0]])

        # Draw the hand annotations on the image

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:

            # multi you can change to right or left for each hand

            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # write to ids
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 0:
                        ids = np.array([[0, 0], [cx, cy]])
                    else:
                        ids = np.append(ids, [[cx-ids.item(1, 0), cy-ids.item(1,  1)]], axis=0)
                ids = np.delete(ids, 0, 0)
                if lastids.size < 3:
                    lastids = np.append(lastids, [ids], axis=2)
                else:
                    lastids = np.append(lastids, [ids], axis=2)
                    lastids = np.delete(lastids, 0)
                print(lastids)
        cv2.imshow('MediaPipe', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()