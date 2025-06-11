import cv2
import numpy as np
import face_recognition
import os
from white_list import WhiteList

white_list = None
encoded_faces = None
white_list_images = None

def setup_camera():
    white_list = WhiteList('DoorBell/Samples')
    encoded_faces = white_list.get_encoded_faces()
    white_list_images = white_list.get_white_lists()

def run_door_bell():
    captured_faces = cv2.VideoCapture(0)

    while True:
        succ, img = captured_faces.read()
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        face_in_frame = face_recognition.face_locations(img_small)
        encode_frame = face_recognition.face_encodings(img_small, face_in_frame)

        for encode_face, face_location in zip(encode_frame, face_in_frame):
            matches = face_recognition.compare_faces(encoded_faces, encode_face)
            face_distance = face_recognition.face_distance(encoded_faces, encode_face)

            match_index = np.argmin(face_distance)

            if matches[match_index]:
                name = white_list_images[match_index][1]
                y1, x2, y2, x1 = face_location
                y1 *= 4
                x2 *= 4
                y2 *= 4
                x1 *= 4

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_PLAIN,
                            1.5, (255, 255, 255), 2)

        # if the face is not in the white list, draw a rectangle around the face and label it as "Unknown"
        for face_location in face_in_frame:
            y1, x2, y2, x1 = face_location
            y1 *= 4
            x2 *= 4
            y2 *= 4
            x1 *= 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, 'Unknown', (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (255, 255, 255), 2)
            

        cv2.imshow('Door Bell', img)

        # q is temp key to exit the camera (for testing purposes)
        # this will be replaced with a end call button in the future
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captured_faces.release()
    cv2.destroyAllWindows()

