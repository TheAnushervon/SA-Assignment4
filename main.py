import cv2
from filters.black_and_white import apply_black_and_white
from filters.mirror import apply_mirror
from filters.resize import apply_resize
from filters.blur import apply_blur

def main():
    cap = cv2.VideoCapture(0)  # 0 for webcam, or provide a video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply filters
        resized_frame = apply_resize(frame)
        mirrored_frame = apply_mirror(resized_frame)
        black_and_white_frame = apply_black_and_white(mirrored_frame)
        blurred_frame = apply_blur(black_and_white_frame)

        # Display the resulting frame
        cv2.imshow('Processed Video', blurred_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
