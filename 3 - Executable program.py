import cv2
import mediapipe as mp
import pickle
import numpy as np

model = pickle.load(open("./trained_model_values_only.p", 'rb'))


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.5)

# labels_dict = {0: "A", 1: "B", 2: "L"}


while True:

    data_aux = []
    x_ = []
    y_ = [] 


    ret, frame = cap.read()
    H, W, _ = frame.shape

    #frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
                )
    
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)


        try:
            prediction = model.predict([np.asarray(data_aux)])
            predictor = prediction[0]

            cv2.rectangle(frame, (x1,y1), (x2, y2), (0,0,255), 3)
            cv2.putText(frame, predictor, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA)

        except Exception as e:
            print(f"Exception: {e}")


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()