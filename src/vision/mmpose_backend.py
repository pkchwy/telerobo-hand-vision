"""MMPose Hand Estimation backend.

Uses `mmpose.apis.MMPoseInferencer` to detect 21 hand landmarks.
The landmarks are converted to 5 servo PWM values via the shared
`landmarks_to_servo` module.

Requires:
    pip install openmim
    mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0"
    pip install mmpose
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import VisionBackend, VisionResult
from .landmarks_to_servo import NEUTRAL_SERVO, landmarks_to_servo


class MMPoseBackend(VisionBackend):
    name = "mmpose"

    def __init__(
        self,
        model_path: str = "hand",
        device: str = "cuda:0",
        mirror: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MMPose backend.

        Args:
            model_path: Alias or path to the config file/checkpoint.
                Defaults to 'hand' which loads a standard 21-point model.
            device: Computing device, e.g., 'cuda:0' or 'cpu'.
            mirror: Whether to horizontally flip the input frame.
            **kwargs: Additional arguments passed to MMPoseInferencer.
        """
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError:
            raise ImportError(
                "mmpose not found. Please install it with:\n"
                "pip install openmim\n"
                "mim install mmengine \"mmcv>=2.0.0\" \"mmdet>=3.0.0\"\n"
                "pip install mmpose"
            )

        # MMPoseInferencer automatically downloads weights if an alias is used.
        self._inferencer = MMPoseInferencer(pose2d=model_path, device=device, **kwargs)
        self.model_path = model_path
        self.device = device
        self.mirror = mirror
        self.model_version = f"mmpose:{model_path}"

    def process(self, frame: np.ndarray, capture_time_ns: int) -> VisionResult:
        import cv2

        if self.mirror:
            frame = cv2.flip(frame, 1)

        # MMPoseInferencer accepts numpy array (BGR).
        # It returns a generator yielding results per frame.
        result_generator = self._inferencer(frame, show=False)
        result = next(result_generator)

        finger_values = list(NEUTRAL_SERVO)
        confidence = 0.0
        detected = False
        keypoints_xy: list[tuple[float, float]] | None = None

        # The structure of result is {'predictions': [[instance1, instance2, ...]], ...}
        # where each instance is a dict with 'keypoints' and 'keypoint_scores'.
        if result.get("predictions"):
            predictions = result["predictions"]
            if predictions:
                # MMPose results can be nested: [[instance1, ...]] for video/frames
                # or [instance1, ...] for single images in some versions.
                first_item = predictions[0]
                if isinstance(first_item, list) and len(first_item) > 0:
                    best_pred = first_item[0]
                else:
                    best_pred = first_item

                if isinstance(best_pred, dict) and "keypoints" in best_pred:
                    kp = np.array(best_pred["keypoints"])  # (21, 2)
                    scores = np.array(best_pred.get("keypoint_scores", [0.0] * len(kp)))
                    
                    if len(kp) >= 21:
                        # Ensure we only use the first 21 points (standard topology).
                        # Order: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
                        kp_21 = kp[:21]
                        finger_values = landmarks_to_servo(kp_21)
                        detected = True
                        confidence = float(np.mean(scores[:21]))
                        keypoints_xy = [(float(p[0]), float(p[1])) for p in kp_21]

        return VisionResult(
            finger_values=finger_values,
            capture_time_ns=capture_time_ns,
            vision_done_time_ns=time.time_ns(),
            confidence=confidence,
            detected=detected,
            metadata={
                "mirror": self.mirror,
                "model": self.model_path,
                "device": self.device
            },
            keypoints_xy=keypoints_xy,
        )

    def close(self) -> None:
        pass
