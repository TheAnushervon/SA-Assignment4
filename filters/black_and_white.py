import cv2

def apply_black_and_white(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
