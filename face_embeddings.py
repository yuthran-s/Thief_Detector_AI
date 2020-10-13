import face_recognition
import cv2
import numpy as np

# CONSTANTS
VIDEO = cv2.VideoCapture(0) # Capturing the Video

TOLERANCE = 0.5# The Tolerance 0.6 you can play around with it

FRAME_THICKNESS = 3 # The thickness of the frame

MODEL = "hog" # CNN if you have a graphics card(I do)





def embeddings():

    thief_name = input("Enter thief's name: ")

    # sample Number for each thief
    sampleNum = 1
    while True:
        ret, image = VIDEO.read()

        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)


        key = cv2.waitKey(1)


        if key == ord('k'):
            cv2.imwrite(f"data/known_faces/{str(thief_name)} {str(sampleNum)}.jpg", image)
            print("clicked")
            sampleNum += 1




        cv2.imshow(f'Detector.AI thief {thief_name}',image)


        if key == ord('q'):
            break

embeddings()