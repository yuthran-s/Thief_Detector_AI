import face_recognition
import cv2
import numpy as np
import os
from os.path import basename
from datetime import datetime

# CONSTANTS
VIDEO = cv2.VideoCapture(0) # Capturing the Video

TOLERANCE = 0.6 # The Tolerance 0.6 you can play around with it

FRAME_THICKNESS = 3 # The thickness of the frame

MODEL = "hog" # CNN if you have a graphics card(I do)

known_faces_dir = "data/known_faces"


match = None


known_names = []


global face_encoding
global name

for filename in os.listdir(known_faces_dir):
    file = f"{known_faces_dir}/{filename}"
    image = face_recognition.load_image_file(file, mode="RGB")
    face_encoding = face_recognition.api.face_encodings(image, known_face_locations=None, model='hog')
    name = basename(filename).split(" ")[0]

    known_names.append(name)




def detector():
    while True:

        ret, image = VIDEO.read()

        locations = face_recognition.face_locations(image, model=MODEL, number_of_times_to_upsample=2)
        encodings = face_recognition.face_encodings(image, locations)

        for face_encodings in encodings:
            results = face_recognition.compare_faces(np.array(face_encoding), face_encodings, tolerance=TOLERANCE)

            if True in results:
                match = known_names[results.index(True)]

                location = "bangalore"
                camera = "camera1"

                # CONSTANTS in here
                now = datetime.now()
                year = now.strftime("%Y")
                date = now.strftime("%d")
                month = now.strftime("%B")
                time = now.strftime("%H%M%S")

                filename = f"{location}_{camera}_{date}{month}{year}_{time}.jpg"

                cv2.imwrite(f"data/camera_feed/{filename}", image)

                (place, camera, date, time) = basename(filename).split(".")[0].split("_")

                date_time = datetime.strptime(f'{date} {time}', '%d%B%Y %H%M%S')

                print(f'Found {match} in {place} on camera location: {camera} at time: {date_time} ')
                print(f' Reference photo file: {filename}')




        cv2.imshow('Detector.AI',image)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break


detector()