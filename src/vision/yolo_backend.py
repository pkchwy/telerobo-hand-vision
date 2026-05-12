"""YOLO-Pose hand keypoints backend.

Uses an Ultralytics YOLO-pose model trained on the hand-keypoints dataset
(21 keypoints, MediaPipe-compatible index order). The 21 points are then
converted to 5 servo PWM values via the shared `landmarks_to_servo`
module — exactly the same code path MediaPipe uses, so the only thing
that differs between backends is the model itself.

The default model path (`weights/yolo_hand_pose.pt`) is where the trained
checkpoint from `tools/train_yolo_hand.ipynb` should be placed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import VisionBackend, VisionResult
from .landmarks_to_servo import NEUTRAL_SERVO, landmarks_to_servo


DEFAULT_MODEL_PATH = "weights/yolo_hand_pose.pt"


class YoloBackend(VisionBackend):
    name = "yolo"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
        mirror: bool = True,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {model_path!r}. Train one with "
                f"tools/train_yolo_hand.ipynb (Colab) and copy best.pt here."
            )

        from ultralytics import YOLO  # lazy import; ultralytics needs Python 3.12

        self._model = YOLO(model_path)
        self.model_path = model_path
        self.model_version = f"ultralytics:{Path(model_path).name}"
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.mirror = mirror

    def process(self, frame, capture_time_ns: int) -> VisionResult:
        import cv2

        if self.mirror:
            frame = cv2.flip(frame, 1)

        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        finger_values = list(NEUTRAL_SERVO)
        confidence = 0.0
        detected = False
        keypoints_xy: list[tuple[float, float]] | None = None

        if results:
            r = results[0]
            kpts = getattr(r, "keypoints", None)
            boxes = getattr(r, "boxes", None)
            if kpts is not None and kpts.xy is not None and len(kpts.xy) > 0:
                # Pick the highest-confidence detection if multiple hands appear.
                best = 0
                if boxes is not None and len(boxes) > 1:
                    best = int(boxes.conf.argmax().item())
                xy = kpts.xy[best].cpu().numpy()  # (21, 2) in pixel coords
                if xy.shape[0] == 21:
                    finger_values = landmarks_to_servo(xy.astype(np.float64))
                    detected = True
                    keypoints_xy = [(float(x), float(y)) for x, y in xy]
                    if boxes is not None and len(boxes) > best:
                        confidence = float(boxes.conf[best].cpu().item())

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
        self._model = None
