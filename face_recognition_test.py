import cv2
import numpy as np
import face_recognition

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


img_elon = face_recognition.load_image_file('images/elon_musk1.jpg')
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)

img_test_elon = face_recognition.load_image_file('images/elon_musk2.jpg')
img_test_elon = cv2.cvtColor(img_test_elon, cv2.COLOR_BGR2RGB)

cv2.imshow('Elon Musk', img_elon)
cv2.imshow('Elon Musk Test', img_test_elon)
cv2.waitKey(0)