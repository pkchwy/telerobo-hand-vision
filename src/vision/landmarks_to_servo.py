"""Shared 21-keypoint hand landmarks → finger angles → servo PWM.

MediaPipe Hands and the Ultralytics hand-keypoints dataset use the same
21-point topology in the same index order (0=wrist, 1-4=thumb, 5-8=index,
9-12=middle, 13-16=ring, 17-20=pinky), so both backends can share this
math. Keeping the conversion identical is what makes per-backend latency
numbers honestly comparable — only the model changes, not the post-vision
pipeline.

Inputs are accepted as (21, 2) or (21, 3) numpy arrays; the angle
computation is dimension-agnostic.
"""

from __future__ import annotations

import numpy as np

OPEN_ANGLES = np.array([170, 165, 165, 165, 160], dtype=np.float64)
CLOSED_ANGLES = np.array([40, 75, 55, 60, 50], dtype=np.float64)

# (p1, vertex, p3) joints used to score how "bent" each finger is.
_FINGER_JOINTS: tuple[tuple[int, int, int], ...] = (
    (2, 3, 4),     # thumb:  MCP, IP, TIP
    (5, 6, 7),     # index:  MCP, PIP, DIP
    (9, 10, 11),   # middle: MCP, PIP, DIP
    (13, 14, 15),  # ring:   MCP, PIP, DIP
    (17, 18, 19),  # pinky:  MCP, PIP, DIP
)


def _angle_at(p1: np.ndarray, vertex: np.ndarray, p3: np.ndarray) -> float:
    v1, v2 = p1 - vertex, p3 - vertex
    l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if l1 < 1e-6 or l2 < 1e-6:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def finger_angles(landmarks: np.ndarray) -> np.ndarray:
    """Return a (5,) array of joint angles in degrees: thumb, index, middle, ring, pinky."""
    return np.array([_angle_at(landmarks[a], landmarks[b], landmarks[c])
                     for a, b, c in _FINGER_JOINTS])


def angles_to_servo(angles: np.ndarray) -> list[int]:
    """Map 5 joint angles to 5 servo PWM values in [500, 1500]."""
    norm = np.clip((OPEN_ANGLES - angles) / (OPEN_ANGLES - CLOSED_ANGLES + 1e-3), 0.0, 1.0)
    pwm = np.clip(1500 - (norm * 1000), 500, 1500).astype(int)
    return pwm.tolist()


def landmarks_to_servo(landmarks: np.ndarray) -> list[int]:
    return angles_to_servo(finger_angles(landmarks))


NEUTRAL_SERVO: list[int] = [1500, 1500, 1500, 1500, 1500]
