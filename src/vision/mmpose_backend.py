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


# Aliases MMPoseInferencer resolves via mmpose's model-index.yml — these are
# either short aliases ("hand") or model `Name:` entries from a metafile.yml.
# All auto-download their checkpoints on first use. Anything not in this set
# is treated as a filesystem path and existence-checked.
#
# These four cover distinct hand-keypoint architectures, which is what makes
# them useful as a comparison set:
#   rtmpose-m_...-hand5     RTMPose (SimCC, transformer-distilled, fastest)
#   td-hm_hrnetv2-...       HRNetv2-w18 (high-resolution heatmap CNN)
#   td-hm_hourglass52-...   Stacked Hourglass (OpenPose-family)
#   td-hm_res50-...         ResNet-50 SimpleBaseline (classic topdown)
MMPOSE_PRESETS: dict[str, str] = {
    "rtmpose_hand": "hand",  # short alias resolves to rtmpose-m_8xb256-210e_hand5-256x256
    "hrnet_hand": "td-hm_hrnetv2-w18_8xb32-210e_coco-wholebody-hand-256x256",
    "hourglass_hand": "td-hm_hourglass52_8xb32-210e_coco-wholebody-hand-256x256",
    "resnet_hand": "td-hm_res50_8xb32-210e_coco-wholebody-hand-256x256",
}

_BUILTIN_ALIASES = {"hand", *MMPOSE_PRESETS.values()}


class MMPoseBackend(VisionBackend):
    name = "mmpose"

    def __init__(
        self,
        model_path: str = "hand",
        device: str | None = None,
        mirror: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MMPose backend.

        Args:
            model_path: Alias (e.g. 'hand') or path to the config/checkpoint.
                Aliases trigger auto-download; filesystem paths are validated.
            device: Computing device, e.g., 'cuda:0' or 'cpu'. None lets
                MMPose auto-detect (falls back to CPU if no GPU is available).
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

        if model_path not in _BUILTIN_ALIASES and not Path(model_path).exists():
            raise FileNotFoundError(
                f"MMPose model not found at {model_path!r}. Use a built-in "
                f"alias ({sorted(_BUILTIN_ALIASES)}) or provide a valid path."
            )

        inferencer_kwargs: dict[str, Any] = {"pose2d": model_path, **kwargs}
        if device is not None:
            inferencer_kwargs["device"] = device
        self._inferencer = MMPoseInferencer(**inferencer_kwargs)
        self.model_path = model_path
        self.device = device
        self.mirror = mirror
        self.model_version = f"mmpose:{model_path}"

    def process(self, frame: np.ndarray, capture_time_ns: int) -> VisionResult:
        import cv2

        if self.mirror:
            frame = cv2.flip(frame, 1)

        result_generator = self._inferencer(frame, show=False)
        result = next(result_generator)

        finger_values = list(NEUTRAL_SERVO)
        confidence = 0.0
        detected = False
        keypoints_xy: list[tuple[float, float]] | None = None

        # MMPose result shape: {'predictions': [[instance1, ...]], ...}
        # Older versions sometimes return [instance1, ...] for single images.
        if result.get("predictions"):
            predictions = result["predictions"]
            if predictions:
                first_item = predictions[0]
                if isinstance(first_item, list) and len(first_item) > 0:
                    best_pred = first_item[0]
                else:
                    best_pred = first_item

                if isinstance(best_pred, dict) and "keypoints" in best_pred:
                    kp = np.asarray(best_pred["keypoints"], dtype=np.float64)
                    scores = np.asarray(
                        best_pred.get("keypoint_scores", [0.0] * len(kp)),
                        dtype=np.float64,
                    )

                    if len(kp) >= 21:
                        # Order matches MediaPipe/YOLO topology:
                        # 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle,
                        # 13-16=ring, 17-20=pinky
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
                "device": self.device,
            },
            keypoints_xy=keypoints_xy,
        )

    def close(self) -> None:
        self._inferencer = None
