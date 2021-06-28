import pickle
import cv2
import numpy as np
import os
import argparse
import face_recognition as fr
from sklearn.svm import SVC


ap = argparse.ArgumentParser()

ap.add_argument("-e", "--example", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

imagepath = args['example']
img = cv2.imread(imagepath)
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#cv2.imshow('original',rgb)
#cv2.waitKey(0)
boxes = fr.face_locations(img =rgb,model= args['detection_method'])
#(y,r,b,x)=boxes[0]
#cv2.rectangle(rgb,(x,y),(r,b),(0,255,0),3)
#cv2.imshow('converted', rgb)
#cv2.waitKey(0)

#get encodings for test image
xhat = fr.face_encodings(rgb,known_face_locations=boxes)

with open('Names.pickle','rb') as f:
    knownNames = pickle.load(f)
with open('Encodings.pickle','rb') as f:
    knownEncodings = pickle.load(f)
x =[]
y = []
for i,k in enumerate(knownNames) :
    if k!= 'UNKNOWN' :
        y.append(k)
        x.append(knownEncodings[i][0])
x = np.array(x)
y = np.array(y)

model = SVC(decision_function_shape='ovo')
model.fit(x,y)
yhat = model.predict(xhat)
print(yhat)