import cv2
import numpy as np
import face_recognition

white_list = []

def add_to_white_list(image_path):
    global white_list
    img = face_recognition.load_image_file(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(img)

    if face_encodings:
        white_list.append(face_encodings[0])
        print(f"Added {image_path} to white list.")
    else:
        print(f"No face found in {image_path}.")