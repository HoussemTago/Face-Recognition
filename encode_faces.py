# import the necessary packages
from imutils import paths
import face_recognition as fr
import argparse
import pickle
import cv2
import os
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []


for (i,im) in enumerate(imagePaths) :
	print('processing image {}/{}'.format(i+1,len(imagePaths)))
	# Loading the images with OpenCV :
	image = cv2.imread(im)
	#cv2.imshow('origin',image)
	rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	#cv2.imshow('converted', rgb)
	#cv2.waitKey(0)
	boxes = fr.face_locations(img =rgb,model= args['detection_method'])
	#We discard images that have 0 or more than 1 face :
	if len(boxes)!= 1 :
		knownNames.append('UNKNOWN')
		knownEncodings.append('NOTNEEDED')
		#(y,r,b,x) = boxes[0]
		print('image has {} faces! => Discarded'.format(len(boxes)))
		#draw the face location rectangle on the image :
		#cv2.rectangle(rgb,(x,y),(r,b),(0,255,0),5)
		#cv2.imshow('converted', rgb)
		#cv2.waitKey(0)
	else :
		#get face encodings :
		encoding = fr.face_encodings(rgb,known_face_locations=boxes)
		#get name from file name :
		name = (im.split(os.path.sep)[1].split(' ')[0])
		knownNames.append(name)
		knownEncodings.append(encoding)
print(len(knownNames))
print(len(knownEncodings))
with open('Encodings.pickle', 'wb') as f:
	pickle.dump(knownEncodings, f)
with open('Names.pickle', 'wb') as f:
	pickle.dump(knownNames, f)