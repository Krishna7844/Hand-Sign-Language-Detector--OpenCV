#-------------------Looping-------------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import cv2 
import mediapipe as mp
import numpy as np


#---------Loading the Media Pipe Hands tracking-----------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, static_image_mode=True)
#--------------------------------------------------------------------------------------------

#-----------------Loading paths of images into directories-------------------------------------
path= "images/"
existed_dataframe = "Dataframe_A.csv"
files = os.listdir(path=path)
files = [i for i in files if os.path.isdir(path + "/" + i) or i.endswith(".png")]

#--------------------------------------------------------------------------------------------

#-------------------------------Making of Directory List------------------------------------
labels = []
data = []
directories = []
for label in sorted(files):
    content = os.listdir(os.path.join(path, label))
    for file in content:
        data.append(file)
        labels.append(label)
        directories.append(label)
directories = set(directories)
directories = list(directories)
directories.sort()
print(directories)
#--------------------------------------------------------------------------------------------

#----------------------Starting tracking and saving of image data in csv-----------------------
dict = {}
x_cord = []
y_cord = []
missed_images = []

for letters in directories:
    for i in range(0, 290):
        img = f"images/{letters}/{letters}{i}.png"
        frame = cv2.imread(img)
        frame = cv2.resize(frame, (224,224))
        # frame = cv2.flip(frame, 1)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)
        custom_connection_lines = mp_drawing.DrawingSpec(color = (250,215,15), thickness=2)    #making colorful joining lines in the palm

        x_cord = []
        y_cord = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:   
                try:   
                    mp_drawing.draw_landmarks(frame , hand_landmark, mp_hands.HAND_CONNECTIONS, custom_connection_lines)
                    for i in hand_landmark.landmark:
                        x_cord.append(i.x)
                        y_cord.append(i.y)
                except:
                    missed_images.append(img)

        print(len(x_cord))
        print(len(y_cord))
        
        dict = {}
        
        
        for j, (x_val, y_val) in enumerate(zip(x_cord,y_cord)):
            dict[f"x{j}"] = x_val
            dict[f"y{j}"] = y_val
        dict.update({"label":letters})

        print(dict)
        df = pd.read_csv("./Dataframe_pre_final.csv")
        temp_df = pd.DataFrame([dict])
        result = pd.concat([df, temp_df])
        dict = dict.clear()
        result.to_csv("./Dataframe_pre_final.csv", index=False)


df["file"] = data
df["labels"] = labels
print(missed_images)

cv2.imshow("SIGN LANG", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()