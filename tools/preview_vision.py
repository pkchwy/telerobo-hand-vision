"""
tools/preview_vision.py — standalone vision backend preview.

Runs a vision backend against a live camera and draws keypoints on screen.
No gRPC, no server, no Arduino — single terminal. Use this when you only
want to see what the backend detects (e.g. to sanity-check a new .pt file).

Usage:
    python -u -m tools.preview_vision --backend yolo
    python -u -m tools.preview_vision --backend yolo --model-path weights/best.pt
    python -u -m tools.preview_vision --backend mediapipe --show-indices
"""
from __future__ import annotations

import argparse
import sys

import cv2

from src.vision import available_backends, get_backend
from tools.run_backend import CameraSource, _draw_preview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=available_backends(), required=True)
    parser.add_argument("--model-path", default=None,
                        help="Override the backend's default model file. "
                             "yolo: .pt path (default weights/yolo_hand_pose.pt). "
                             "mediapipe: .task path (default weights/hand_landmarker.task).")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show-indices", action="store_true",
                        help="Label each keypoint with its index (0-20).")
    mirror_grp = parser.add_mutually_exclusive_group()
    mirror_grp.add_argument("--mirror", dest="mirror", action="store_true", default=None)
    mirror_grp.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = parser.parse_args()

    backend_kwargs: dict = {}
    if args.model_path:
        backend_kwargs["model_path"] = args.model_path
    if args.mirror is not None:
        backend_kwargs["mirror"] = args.mirror

    backend = get_backend(args.backend, **backend_kwargs)
    print(f"[preview] backend={backend.name} model_version={backend.model_version}")
    print("[preview] press q to quit")

    source = CameraSource(args.camera_index, args.width, args.height, args.fps)
    try:
        for frame, capture_time_ns in source:
            result = backend.process(frame, capture_time_ns)
            key = _draw_preview(frame, result, backend.name,
                                show_indices=args.show_indices)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        backend.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
