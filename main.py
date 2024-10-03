from queue import Queue

import cv2

from filters import (
    BnWFilter,
    BlurFilter,
    MirrorFilter,
    ResizeFilter,
    ShowFilter,
    EdgeDetectionFilter,
)

sink_pipe = Queue()


def video_processing_loop(source_pipe):
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            source_pipe(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def top_down_approach():
    show_filter = ShowFilter(outputs=[sink_pipe.put])
    bnw_filter = BnWFilter(outputs=[show_filter.input])
    blur_filter = BlurFilter(kernel_size=15, outputs=[bnw_filter.input])
    mirror_filter = MirrorFilter(outputs=[blur_filter.input])
    resize_filter = ResizeFilter(scale_factor=0.5, outputs=[mirror_filter.input])

    input_show_filter = ShowFilter(
        window_name="Input video", outputs=[resize_filter.input]
    )

    source_pipe = input_show_filter.input
    video_processing_loop(source_pipe)


def pipe_approach():
    pipe = (
        ShowFilter("Input Video")
        .pipe(EdgeDetectionFilter())
        .pipe(ResizeFilter(0.5))
        .pipe(ShowFilter(outputs=[sink_pipe.put]))
    )

    video_processing_loop(pipe.first.input)


if __name__ == "__main__":
    top_down_approach()
