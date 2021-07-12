from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import pickle
import csv
from datetime import datetime
from datetime import date
app=Flask(__name__)
camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.


file_to_read = open("face_pkl2.pkl", "rb")
loaded_dictionary = pickle.load(file_to_read)
#print(loaded_dictionary)

known_face_names=list(loaded_dictionary.get('names'))
known_face_encodings=list(loaded_dictionary.get('encodings'))


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def MarkAttendance(name):
    already_in_file = set()
    dt= set()
    #date=date.today()
    with open('ATTLOG.csv', "r+") as g: 
        # just read
        for line in g:
            already_in_file.add(line.split(",")[0])
            #dt.add(line.split(",")[2])
# process the current entry:
   # if name in already_in_file and  date.today() != dt:
       # with open('ATTNDNCELOGER1.csv', "a") as g:
            # append
           ## now = datetime.now()
           # dtString = now.strftime('%H:%M:%S')
           # dtString1=date.today()
           # g.writelines(f'\n{name},{dtString},{dtString1}')


    if name not in already_in_file:
        with open('ATTLOG.csv', "a") as g:
            # append
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dtString1=date.today()
            g.writelines(f'\n{name},{dtString},{dtString1}')

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                MarkAttendance(name)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)