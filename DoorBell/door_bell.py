import cv2
import numpy as np
import face_recognition
import os

white_list_path = 'White_List' # path to the white list file
white_list_images = [] # list to store white list images
white_list_names = [] # list to store white list names
white_list = os.listdir(white_list_path) # get all images in the white list directory

for person in white_list:
    img = cv2.imread(f'{white_list_path}/{person}')
    white_list_images.append(img)
    white_list_names.append(os.path.splitext(person)[0])  # Store names without file extension


def add_to_white_list(person):
    global white_list
    img = face_recognition.load_image_file(person)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(img)

    if face_encodings:
        white_list.append(face_encodings[0])
        print(f"Added {person} to white list.")
    else:
        print(f"No face found in {person}.")