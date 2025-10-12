import time
from collections import deque


class Fps_Counter:
    start_time: float | None
    time_queue: deque
    cum_time: float

    def __init__(self):
        self.start_time = None
        self.time_queue = deque()
        self.cum_time = 0

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        assert self.start_time is not None
        now = time.perf_counter()
        elapsed_time = now - self.start_time
        if 100 < len(self.time_queue):
            popped_time = self.time_queue.popleft()
            self.cum_time -= popped_time
        self.time_queue.append(elapsed_time)
        self.cum_time += elapsed_time
        self.start_time = now

    def get_fps(self):
        assert len(self.time_queue) != 0
        fps = 1 / (self.cum_time / len(self.time_queue))
        return fps
