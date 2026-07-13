"""Rolling position history for the puller's motion trail -- same
seconds-based windowing as RollingHistory (see plotting.py), just storing
(x, y) sim-space positions instead of scalar series."""
from collections import deque


class Trail:
    def __init__(self, window_seconds):
        self.window_seconds = window_seconds
        self.points = deque()  # (t, x, y), oldest first

    def reset(self):
        self.points.clear()

    def add(self, t, x, y):
        self.points.append((t, x, y))
        cutoff = t - self.window_seconds
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()
