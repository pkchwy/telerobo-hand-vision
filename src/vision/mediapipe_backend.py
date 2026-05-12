"""MediaPipe HandLandmarker backend (Tasks API).

Uses `mediapipe.tasks.vision.HandLandmarker` (the only API supported in
mediapipe >= ~0.10.20). The 21 hand landmarks are converted to 5 servo
PWM values via the shared `landmarks_to_servo` module so MediaPipe and
YOLO produce commands the same way — making the latency comparison
honest.

The model file (`weights/hand_landmarker.task`) can be fetched from:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import VisionBackend, VisionResult
from .landmarks_to_servo import NEUTRAL_SERVO, landmarks_to_servo


DEFAULT_MODEL_PATH = "weights/hand_landmarker.task"


class MediaPipeBackend(VisionBackend):
    name = "mediapipe"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mirror: bool = True,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe hand_landmarker.task not found at {model_path!r}. "
                "Download it from "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.model_path = model_path
        self.mirror = mirror
        self.model_version = f"mediapipe-tasks:{Path(model_path).name}"
        self._t0_ns: int | None = None

    def process(self, frame, capture_time_ns: int) -> VisionResult:
        import cv2

        if self.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires monotonically increasing ms timestamps.
        if self._t0_ns is None:
            self._t0_ns = capture_time_ns
        timestamp_ms = max(0, (capture_time_ns - self._t0_ns) // 1_000_000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        finger_values = list(NEUTRAL_SERVO)
        confidence = 0.0
        detected = False
        keypoints_xy: list[tuple[float, float]] | None = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            lm = np.array([[pt.x, pt.y, pt.z] for pt in hand_landmarks])
            finger_values = landmarks_to_servo(lm)
            detected = True
            h, w = frame.shape[:2]
            keypoints_xy = [(float(pt.x) * w, float(pt.y) * h) for pt in hand_landmarks]
            if result.handedness and result.handedness[0]:
                confidence = float(result.handedness[0][0].score)

        return VisionResult(
            finger_values=finger_values,
            capture_time_ns=capture_time_ns,
            vision_done_time_ns=time.time_ns(),
            confidence=confidence,
            detected=detected,
            metadata={"mirror": self.mirror, "model": self.model_path},
            keypoints_xy=keypoints_xy,
        )

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
