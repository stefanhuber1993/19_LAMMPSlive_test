"""Rolling per-atom position history behind every atom's fading motion
trail (not just the puller's) -- same seconds-based windowing as
RollingHistory (see plotting.py), just storing a whole-system snapshot per
frame instead of a scalar series.

Snapshots are keyed by atom id (not array index): LAMMPS is free to
reorder atoms in its local arrays between steps (e.g. periodic spatial
sorting for cache locality), so matching "the same atom" across frames has
to go through id, not position-in-array.
"""
from collections import deque


class AtomTrails:
    def __init__(self, window_seconds):
        self.window_seconds = window_seconds
        self.frames = deque()  # (t, {id: (x, y, is_puller)}), oldest first

    def reset(self):
        self.frames.clear()

    def add(self, t, ids, positions, is_puller):
        snapshot = {
            int(atom_id): (float(x), float(y), bool(puller))
            for atom_id, (x, y), puller in zip(ids, positions, is_puller)
        }
        self.frames.append((t, snapshot))
        cutoff = t - self.window_seconds
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()
