import cv2
import numpy as np
import face_recognition
import os
from white_list import WhiteList

class DoorBell:
    def __init__(self):
        white_list = WhiteList('DoorBell/Samples')
        encoded_faces = white_list.get_encoded_faces()
        white_list_images = white_list.get_white_lists()

    def view_white_list(self):
        print("White List:")

    def run_door_bell(self):
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            succ, frame = cam.read()

            if not succ:
                print("Error: Could not read frame.")
                break

            # resizing the image for faster processing
            frame_small = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            face_loc = face_recognition.face_locations(frame_small)
            encode_frame = face_recognition.face_encodings(frame_small, face_loc)

            for encode_face, face_location in zip(encode_frame, face_loc):
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

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_PLAIN,
                                1.5, (255, 255, 255), 2)

            # if the face is not in the white list, draw a rectangle around the face and label it as "Unknown"
            for face_location in face_loc:
                y1, x2, y2, x1 = face_location
                y1 *= 4
                x2 *= 4
                y2 *= 4
                x1 *= 4

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_PLAIN,
                            1.5, (255, 255, 255), 2)
                

            cv2.imshow('Door Bell', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Door Bell', cv2.WND_PROP_VISIBLE) < 1:
                break  

        cam.release()
        cv2.destroyAllWindows()

