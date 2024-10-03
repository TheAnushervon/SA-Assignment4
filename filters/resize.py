import cv2

def apply_resize(frame, scale_factor=0.5):
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    return cv2.resize(frame, (width, height))
