# Face_detection_with_flask
The face training  is done using face images in the training folder with the help of training.ipynb script.
The encoded faces are serialised to the disk as face_pkl2.pkl
The facedetectapp.py reads this serialised file and detects face from a live feed using laptop camera.
The identified faces are marked as preset with their time of detection and date.
