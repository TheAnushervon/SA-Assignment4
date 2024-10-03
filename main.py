from queue import Queue
import cv2
from filters import BnWFilter, BlurFilter, MirrorFilter, ResizeFilter, ShowFilter, EdgeDetectionFilter

sink_pipe = Queue()

show_filter = ShowFilter(outputs=[])
blur_filter = BlurFilter(kernel_size=15, outputs=[show_filter.input, sink_pipe])
# bnw_filter = BnWFilter(outputs=[show_filter.input])
mirror_filter = MirrorFilter(outputs=[blur_filter.input])
resize_filter = ResizeFilter(scale_factor=0.5,outputs=[mirror_filter.input])
edge_filter = EdgeDetectionFilter(outputs=[resize_filter.input])

input_show_filter = ShowFilter(
    window_name='Input video',
    outputs=[edge_filter.input]
)



source_pipe = input_show_filter.input

cap = cv2.VideoCapture(0) 

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        source_pipe(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print('KeyboardInterrupt')

cap.release()
cv2.destroyAllWindows()


