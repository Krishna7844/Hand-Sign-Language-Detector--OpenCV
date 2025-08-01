
import cv2
import numpy as np
cap = cv2.VideoCapture("Videos/R.mov")
c = 0
while True:
    r, frame = cap.read()
    if r == True:
        frame = cv2.resize(frame, (500, 500))
        filename = "./images/R/R" + str(c)+ ".png"
        cv2.imwrite(filename, frame)
        cv2.imshow("ws",frame)
        c += 1
        if cv2.waitKey(25) & 0xff == ord("p"):

            break
    else:
        break
cap.release()
cv2.destroyAllWindows()