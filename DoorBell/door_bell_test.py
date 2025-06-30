# temporary test file for doorbell
import cv2
import numpy as np
import face_recognition
from door_bell import DoorBell

def test_Doorbell():
    doorbell = DoorBell()
    
    doorbell.add_sample()
    doorbell.run_door_bell()
    
test_Doorbell()