import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)
face_cascade =  cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data_path = "./data/"
face_data = []
labels = []
names = {}


def distance(x1,x2):    # Returning the eucledian distance
    return np.sqrt(((x1-x2)**2).sum())


def knn(train,query,k=5):
    vals = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        vals.append([distance(query,ix),iy])

    vals = sorted(vals,key = lambda f:f[0])
    vals = vals[:k] # taking the first k values from the sorted vals
    new_vals = np.array(vals)
    new_vals = np.unique(new_vals[:,-1],return_counts=True)
    index = new_vals[1].argmax()
    return new_vals[0][index]

id = 0

for f in os.listdir(data_path):
    if f.endswith(".npy"):
        data_item = np.load(data_path+f)
        face_data.append(data_item)
        target = id*np.ones((data_item.shape[0],))
        labels.append(target)
        names[id] = f[:-4]
        id += 1

face_dataset = np.concatenate(face_data,axis=0)
labels_dataset = np.concatenate(labels,axis=0).reshape((-1,1))

train_dataset = np.concatenate((face_dataset,labels_dataset),axis=1)

print(face_dataset.shape)
print(labels_dataset.shape)
print(train_dataset.shape)


while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) 

    for (x,y,w,h) in faces :
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        output = knn(train_dataset,face_section.flatten())
        pred_name = names[int(output)]

        cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x,y),(x+w,y+h),(255,0,0),2 )

    cv2.imshow("Face Recognition",gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('f'):
        break
    
cap.release()
cv2.destroyAllWindows()