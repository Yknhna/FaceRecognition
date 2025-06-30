import os
import cv2
import face_recognition
import numpy as np


class WhiteList:

    path: str

    white_list_images: list[str]
    white_list: list[tuple]

    _encoded_faces: list

    def __init__(self, images_path: str) -> None:
        self.path = images_path
        self.white_list_images = os.listdir(images_path)

        self.white_list = []
        for person in self.white_list_images:
            img = cv2.imread(f'{self.path}/{person}')
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.white_list.append((img, os.path.splitext(person)[0]))

        self._encoded_faces = []
        print(self._encode_faces())
        # Encode every face for future use.

    def _encode_faces(self) -> bool:
        """Encodes each white listed person's face and stores them in a list, encoded_faces.
        Returns True iff encoded_face is non-empty i.e. it has been successfully encoded.
        """
        for person in self.white_list:
            image = person[0]
            encoded_face = face_recognition.face_encodings(image)
            if encoded_face:
                self._encoded_faces.append(encoded_face[0])
            else:
                return False
        return True

    def get_white_lists(self) -> list[tuple]:
        """ Returns list of white listed person's faces."""
        return self.white_list

    def get_encoded_faces(self) -> list:
        """ Returns list of encoded faces."""
        return self._encoded_faces

    def add_white_list(self, img, person) -> bool:
        """ Adds <person> to the whitelist, which the person's image has been already
        added to the directory, <self.path>.
        Returns True iff encoded_face is not None i.e. it has been successfully encoded.
        """
        self.white_list.append((img, os.path.splitext(person)[0]))
        encoded_face = face_recognition.face_encodings(img)

        if encoded_face:
            self._encoded_faces.append(encoded_face[0])
            return True
        return False


if __name__ == '__main__':
    path = 'Samples'
    test = WhiteList(path)