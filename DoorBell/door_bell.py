import cv2
import numpy as np
import face_recognition
import os

white_list_path = 'White_List' # path to the white list file
white_list = []
white_list_images = os.listdir(white_list_path) # get all images in the white list directory


# whitelist : [elon must.jpg, elon_musk2.jpg]
# splittext -> elon_must

for person in white_list_images:
    img = cv2.imread(f'{white_list_path}/{person}')
    white_list.append((img, os.path.splitext(person)[0]))

def encode_faces(white_list_images):
    encode_faces = []
    
    for img in white_list_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)

        if face_encodings:
            encode_faces.append(face_encodings[0])
        else:
            encode_faces.append(None)  # Append None if no face is found

def add_to_white_list(person):
    global white_list_images
    img = face_recognition.load_image_file(person)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(img)

    if face_encodings:
        white_list_images.append(face_encodings[0])
        print(f"Added {person} to white list.")
    else:
        print(f"No face found in {person}.")

def run_camera():
    captured_faces = cv2 .VideoCapture(0)

    while True:
        succ, img = captured_faces.read()
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        face_in_frame = face_recognition.face_locations(img_small)
        encode_frame = face_recognition.face_encodings(img_small, face_in_frame)

        