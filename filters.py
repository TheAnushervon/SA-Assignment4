from __future__ import annotations
from dataclasses import dataclass, field
from queue import Queue
import cv2
from abc import abstractmethod, ABC

@dataclass
class Filter(ABC):
    outputs: list[Queue | callable] = field(default_factory=list, kw_only=True)

    def pipe(self, filter: Filter, /):
        self.outputs.append(filter.input)

        return filter

    @abstractmethod
    def apply(self, frame):
        pass

    def input(self, frame):
        out_frame = self.apply(frame)
        for output in self.outputs:
            if callable(output):
                output(out_frame)
            elif isinstance(output, Queue):
                output.put(out_frame)


class BnWFilter(Filter):
    def apply(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

@dataclass
class BlurFilter(Filter):
    kernel_size: int = 5

    def apply(self, frame):
        return cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)


class MirrorFilter(Filter):
    def apply(self, frame):
        return cv2.flip(frame, 1)
    
@dataclass
class ResizeFilter(Filter):
    scale_factor: float = 0.5

    def apply(self, frame):
        width = int(frame.shape[1] * self.scale_factor)
        height = int(frame.shape[0] * self.scale_factor)
        return cv2.resize(frame, (width, height))


@dataclass
class ShowFilter(Filter):
    window_name: str = 'Processed Video'
    pinned: bool = True

    def __post_init__(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def apply(self, frame):
        cv2.imshow(self.window_name, frame)
        if self.pinned:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        return frame

    

class EdgeDetectionFilter(Filter):
    def apply(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, threshold1=100, threshold2=200)
        return edges
