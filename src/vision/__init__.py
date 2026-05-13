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


def _make_mmpose_preset(preset_name: str) -> Callable[..., VisionBackend]:
    """Build an MMPose constructor pinned to a specific preset model.

    User-supplied `model_path` still wins, so each preset is just a default.
    """
    def _factory(**kwargs: Any) -> VisionBackend:
        from .mmpose_backend import MMPOSE_PRESETS, MMPoseBackend

        kwargs.setdefault("model_path", MMPOSE_PRESETS[preset_name])
        return MMPoseBackend(**kwargs)

    _factory.__name__ = f"_make_{preset_name}"
    return _factory


_REGISTRY: dict[str, Callable[..., VisionBackend]] = {
    "mediapipe": _make_mediapipe,
    "yolo": _make_yolo,
    # MMPose architecture presets — comparison set for hand-keypoint research.
    # Each auto-downloads weights on first run.
    "rtmpose": _make_mmpose_preset("rtmpose_hand"),
    "hrnet": _make_mmpose_preset("hrnet_hand"),
    "hourglass": _make_mmpose_preset("hourglass_hand"),
    "resnet": _make_mmpose_preset("resnet_hand"),
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
