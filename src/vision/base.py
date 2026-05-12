"""Vision backend interface.

A vision backend takes a single camera frame and returns the 5 servo values
plus the wall-clock nanosecond timestamps needed for staged latency
measurement. All backends (MediaPipe, YOLO, future ones) implement this same
contract so the rest of the pipeline does not care which one is producing
servo commands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisionResult:
    finger_values: list[int]
    capture_time_ns: int
    vision_done_time_ns: int
    confidence: float = 0.0
    detected: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    keypoints_xy: list[tuple[float, float]] | None = None  # 21 points in processed-frame pixel coords


class VisionBackend(ABC):
    """Single-frame vision processor.

    The runner is responsible for capturing the frame and stamping
    `capture_time_ns` *before* calling `process`. The backend stamps
    `vision_done_time_ns` itself once it has produced a result; this
    keeps the boundary between "frame captured" and "vision finished"
    honest even when the backend re-uses the input timestamp.
    """

    name: str = "base"
    model_version: str = "unknown"

    @abstractmethod
    def process(self, frame, capture_time_ns: int) -> VisionResult:
        ...

    def close(self) -> None:
        return None

    def __enter__(self) -> "VisionBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
