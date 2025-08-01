import cv2
import mediapipe as mp
import numpy

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, static_image_mode=True)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (600, 600))
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    custom_connection_lines = mp_drawing.DrawingSpec(color = (250,215,15), thickness=2)


    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame , hand_landmark, mp_hands.HAND_CONNECTIONS, custom_connection_lines)
            print(hand_landmark.landmark[0].x, "     |     ", hand_landmark.landmark[0].x)
    

    cv2.imshow("SIGN LANG", frame)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()