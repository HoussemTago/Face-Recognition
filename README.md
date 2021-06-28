# Face-Recognition
Face Recognition Project using Python. 

# Project steps :
## Step 1 : Creating a dataset of images :
bingImageAPI.py uses the bing_image_downloader library to create a dataset that will later be used to train our face recognition model.
In this example, we download 80 images (20 image for each one of 4 famous football players)

## Step 2 : Get the list of encodings for each one of the image
In the encode_faces.py, we go through the images in our dataset and :
  - Read and process image using OpenCV
  - Locate the faces in each image
  - Get the encodings for each image
  - Extract the names of the people from the file names


The encode_faces.py file is executed using command lines :

  -i DATASET, --dataset DATASET
  path to input directory of faces + images
  
  -e ENCODINGS, --encodings ENCODINGS
  path to serialized db of facial encodings
  
  -d DETECTION_METHOD, --detection-method DETECTION_METHOD
  face detection model to use: either `hog` or `cnn`

Command line example : python encode_faces.py -i dataset -e encodings.pickle -d hog

## Step 3 : Build a face recognition model using SVM and predict the person on a test image
In the recognize_faces.py :
  - Extract the lists of known Encodings and names from the pickle files (The output of step2) : This is our training data
  - Read and process the test image using OpenCV
  - Locate the faces in the image
  - Get the encodings for the image
  - Build an SVM classification model using the one-vs-one method (multi-class classification problem)


The encode_faces.py file is executed using command lines :
  -i DATASET, --dataset DATASET
  path to input directory of faces + images
  
  -e ENCODINGS, --encodings ENCODINGS
  path to serialized db of facial encodings
  
  -d DETECTION_METHOD, --detection-method DETECTION_METHOD
  face detection model to use: either `hog` or `cnn`
  
Command line example : python recognize_faces.py -e "examples\cristiano ronaldo3.jpg" -d hog

This file outputs the accuracy score of the model built using a train/test split (test size = 20%) and the prediction of the person on the image using the SVM model.

