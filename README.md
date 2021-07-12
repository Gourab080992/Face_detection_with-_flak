# Face_detection_with_flask
The face training  is done using face images in the training folder with the help of training.ipynb script.
The encoded faces are serialised to the disk as face_pkl2.pkl
The facedetectapp.py reads this serialised file and detects face from a live feed using laptop camera.
The identified faces are marked as preset with their time of detection and date.
Flask is used for web connection and a simple HTML script gives us the view.
DLIB library is used in the facial encodings of the traning images.
For the identification of faces the trained image to whcih the face matches the most is taken as the prediction.
