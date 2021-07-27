import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0) # initialising the camera object
face_cascade =  cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data_path = "./data/"  # folder path to where face data files will be saved
face_data = [] # face data array
labels = [] # face labels array
names = {} # names dictionary to map each face label to face name


def distance(x1,x2):    # Returning the eucledian distance
    return np.sqrt(((x1-x2)**2).sum())


def knn(train,query,k=5):
    vals = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]  # take i th row but not take the last column as it is labels column
        iy = train[i, -1] # take the last column of the ith row
        vals.append([distance(query,ix),iy])

    vals = sorted(vals,key = lambda f:f[0])
    vals = vals[:k] # taking the first k values from the sorted vals
    new_vals = np.array(vals)
    new_vals = np.unique(new_vals[:,1],return_counts=True) # take the last column and all the rows
    index = new_vals[1].argmax()
    yes = (new_vals[1][index]/k)*100 # caculating the %age of prediction
    return (new_vals[0][index],yes)

id = 0

for f in os.listdir(data_path): # iterating over all the files in the path
    if f.endswith(".npy"):
        data_item = np.load(data_path+f)
        face_data.append(data_item) # data item is matrix of how many face collected * 10000
        target = id*np.ones((data_item.shape[0],)) # so we make a array of labels * number of faces collected
        labels.append(target) 
        names[id] = f[:-4]
        id += 1


face_dataset = np.concatenate(face_data,axis=0) # concatinating all the faces , every row has data of single face 
labels_dataset = np.concatenate(labels,axis=0).reshape((-1,1)) # concatinating all the labels 
                                    #and it gives one row and all column so we reshape it to give one column and all rows

train_dataset = np.concatenate((face_dataset,labels_dataset),axis=1) # now concatinating both dataset to have each row contain 1st column of label and other columns having the face data belonging to that label


#print(face_dataset.shape)
#print(labels_dataset.shape)
#print(train_dataset.shape)


while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # converting frame to gray scale to save memory
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)  # detecting face using haarcascade

    for (x,y,w,h) in faces :
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset] # extracting the face section 
        face_section = cv2.resize(face_section,(100,100))
 
        output = knn(train_dataset,face_section.flatten()) # now using KNN to predict the face name
        pred_name = names[int(output[0])]
        pred_precentage = str(output[1]) + "%"

        cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(gray_frame,pred_precentage,(x+w,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x,y),(x+w,y+h),(255,0,0),2 )

    cv2.imshow("Face Recognition",gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF # by doing AND operation with 0xFF we convert 32 bit into an 8-bit so to compare it to ASCII val
    if key_pressed == ord('f'): # if key 'f' is pressed we break out of the loop
        break 
    
cap.release() # releasing the camera resources
cv2.destroyAllWindows() # destroying all the windows 