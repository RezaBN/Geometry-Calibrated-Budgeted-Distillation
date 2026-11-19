"""Low-overhead teacher forward timing with CUDA-event accounting."""
from __future__ import annotations

import time
import torch


class TeacherForwardTimer:
    """Time teacher forwards without synchronizing after every minibatch call."""
    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_seconds = 0.0

    def run(self, forward, *args, **kwargs):
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = forward(*args, **kwargs)
            end.record()
            self._events.append((start, end))
            return output
        started = time.perf_counter()
        output = forward(*args, **kwargs)
        self._cpu_seconds += time.perf_counter() - started
        return output

    def elapsed_seconds(self) -> float:
        if self.device.type == "cuda" and self._events:
            torch.cuda.synchronize(self.device)
            return self._cpu_seconds + sum(start.elapsed_time(end) / 1000.0 for start, end in self._events)
        return self._cpu_seconds
