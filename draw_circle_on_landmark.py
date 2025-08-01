import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    custom_connection_spec = mp_drawing.DrawingSpec(color=(250, 215, 15), thickness=2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8] 
            thumb_tip = hand_landmarks.landmark[4] 
            ix ,iy = int(index_tip.x*w), int(index_tip.y*h)
            tx ,ty = int(thumb_tip.x*w), int(thumb_tip.y*h)
            cv2.circle(image, (ix,iy), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, (tx,ty), 10, (0, 255, 0), cv2.FILLED)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, connection_drawing_spec = custom_connection_spec) #
            print(ix,iy, tx, ty)

    cv2.imshow("Sign Language Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
