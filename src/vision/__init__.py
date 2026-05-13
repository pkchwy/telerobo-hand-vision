"""Vision backends.

Use `get_backend(name, **kwargs)` to construct a backend by name. New
backends only need to subclass `VisionBackend` and be registered in
`_REGISTRY`.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import VisionBackend, VisionResult


def _make_mediapipe(**kwargs: Any) -> VisionBackend:
    from .mediapipe_backend import MediaPipeBackend

    return MediaPipeBackend(**kwargs)


def _make_yolo(**kwargs: Any) -> VisionBackend:
    from .yolo_backend import YoloBackend

    return YoloBackend(**kwargs)


def _make_mmpose(**kwargs: Any) -> VisionBackend:
    from .mmpose_backend import MMPoseBackend

    return MMPoseBackend(**kwargs)


_REGISTRY: dict[str, Callable[..., VisionBackend]] = {
    "mediapipe": _make_mediapipe,
    "yolo": _make_yolo,
    "mmpose": _make_mmpose,
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_backend(name: str, **kwargs: Any) -> VisionBackend:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown vision backend '{name}'. Available: {available_backends()}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = ["VisionBackend", "VisionResult", "get_backend", "available_backends"]
