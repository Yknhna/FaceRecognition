import os
import cv2
import face_recognition
import numpy as np


class WhiteList:
    def __init__(self, images_path: str) -> None:
        self.white_list_images = os.listdir(images_path)
        print(self.white_list_images)

        self.white_list = []
        for person in self.white_list_images:
            img = cv2.imread(f'{images_path}/{person}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.white_list.append((img, os.path.splitext(person)[0]))

        print(self.white_list)

    def encode_faces(self) -> None:
        pass

    def add_white_list(self, person) -> None:
        pass


if __name__ == '__main__':
    path = 'whitelist_images'
    test = WhiteList(path)