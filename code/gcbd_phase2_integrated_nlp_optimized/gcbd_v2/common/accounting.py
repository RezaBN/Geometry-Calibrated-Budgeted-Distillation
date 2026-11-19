"""Auditable teacher execution and wall-clock ledger."""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class TeacherLedger:
    geometry_examples: int = 0
    geometry_forward_invocations: int = 0
    prediction_examples: int = 0
    prediction_forward_invocations: int = 0
    cache_bytes: int = 0
    cache_write_seconds: float = 0.0
    cache_read_seconds: float = 0.0
    teacher_seconds: float = 0.0
    student_seconds: float = 0.0
    dataloader_wait_seconds: float = 0.0
    total_wall_seconds: float = 0.0
    peak_gpu_memory_bytes: int = 0
    peak_host_memory_bytes: int = 0

    def record_geometry(self, examples: int, invocations: int, seconds: float = 0.0) -> None:
        if examples < 0 or invocations < 0 or seconds < 0:
            raise ValueError("ledger increments must be nonnegative")
        self.geometry_examples += int(examples)
        self.geometry_forward_invocations += int(invocations)
        self.teacher_seconds += float(seconds)

    def record_prediction(self, examples: int, invocations: int, seconds: float = 0.0) -> None:
        if examples < 0 or invocations < 0 or seconds < 0:
            raise ValueError("ledger increments must be nonnegative")
        self.prediction_examples += int(examples)
        self.prediction_forward_invocations += int(invocations)
        self.teacher_seconds += float(seconds)

    @property
    def teacher_examples_total(self) -> int:
        return self.geometry_examples + self.prediction_examples

    @property
    def teacher_forward_invocations_total(self) -> int:
        return self.geometry_forward_invocations + self.prediction_forward_invocations

    def as_dict(self) -> dict:
        result = asdict(self)
        result["teacher_examples_total"] = self.teacher_examples_total
        result["teacher_forward_invocations_total"] = self.teacher_forward_invocations_total
        return result
