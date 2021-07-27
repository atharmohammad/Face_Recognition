import numpy as np
import cv2

cap = cv2.VideoCapture(0) 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
data_path = "./data/"
filename = input("Enter your name :")
face_data = []
skip = 0

while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) # returns list of tuples having co-ordinates of faces in the frame
    
    faces = sorted(faces,key=lambda f:f[2]*f[3], reverse=True) # sorting on the basis of area and then reversing to sort in descending order

    for (x,y,w,h) in faces : 
        cv2.rectangle(gray_frame, (x,y),(x+w,y+h),(255,0,0),2 )

        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset : x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        skip += 1
        
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Face_Recognition Activate", gray_frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('f'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(f"{data_path}{filename}",face_data)
print(f"data succesfully saved at {data_path}{filename}.npy")

cap.release()
cv2.destroyAllWindows()
