import cv2

def apply_blur(frame, kernel_size=5):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
