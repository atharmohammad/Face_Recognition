import numpy as np
import cv2

cap = cv2.VideoCapture(0) 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) # returns list of tuples having co-ordinates of faces in the frame

    for (x,y,w,h) in faces : 
        cv2.rectangle(gray_frame, (x,y),(x+w,y+h),(255,0,0),2 )

    cv2.imshow("Face_Recognition Activate", gray_frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('f'):
        break

cap.release()
cv2.destroyAllWindows()
