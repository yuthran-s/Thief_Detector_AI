import face_recognition
import cv2
import numpy as np

# CONSTANTS
VIDEO = cv2.VideoCapture(0) # Capturing the Video

TOLERANCE = 0.5# The Tolerance 0.6 you can play around with it

FRAME_THICKNESS = 3 # The thickness of the frame

MODEL = "hog" # CNN if you have a graphics card(I do)




def detector():
    while True:
        ret, image = VIDEO.read()

        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0,255,0]

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        cv2.imshow('Detector.AI',image)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

detector()