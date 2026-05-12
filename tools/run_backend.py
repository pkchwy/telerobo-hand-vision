"""Live camera runner for a single vision backend.

Captures frames from a camera, runs them through the chosen vision
backend, streams `HandData` to the gRPC server, correlates incoming
`Ack` messages by `frame_id`, and writes a per-frame CSV plus a
`run_metadata.json`. At the end, prints p50/p90/p95/p99/jitter/throughput
to the terminal.

Usage:
    python -m tools.run_backend --backend mediapipe --frames 1000
    python -m tools.run_backend --backend yolo --duration 30 --warmup 3
    python -m tools.run_backend --backend mediapipe --source fake
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional

from src.metrics import (
    CameraInfo,
    FrameRecord,
    LatencyLogger,
    RunMetadata,
    get_offset_ns,
)
from src.network.client import (
    DEFAULT_SERVER_ADDRESS,
    DEFAULT_SERVER_PORT,
    NetworkClient,
)
from src.network import hand_control_pb2
from src.vision import VisionBackend, available_backends, get_backend


def _open_camera(index: int, width: int, height: int, fps: int):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {index}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class FrameSource:
    """Yields (frame, capture_time_ns) tuples until exhausted or stopped."""

    def stop(self) -> None: ...

    def __iter__(self) -> Iterator: ...

    def info(self) -> CameraInfo:
        return CameraInfo()


class CameraSource(FrameSource):
    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.cap = _open_camera(index, width, height, fps)
        import cv2  # noqa: F401

        self._width = int(self.cap.get(3))
        self._height = int(self.cap.get(4))
        self._fps = float(self.cap.get(5)) or float(fps or 0)
        self._stop = False

    def __iter__(self):
        while not self._stop:
            t = time.time_ns()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            yield frame, t

    def stop(self) -> None:
        self._stop = True
        try:
            self.cap.release()
        except Exception:
            pass

    def info(self) -> CameraInfo:
        return CameraInfo(width=self._width, height=self._height, fps=self._fps)


class FakeSource(FrameSource):
    """Black-frame source for end-to-end pipeline testing without a camera."""

    def __init__(self, fps: int = 30, width: int = 64, height: int = 64) -> None:
        import numpy as np

        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._period_s = 1.0 / fps if fps > 0 else 0.0
        self._fps = fps
        self._width = width
        self._height = height
        self._stop = False

    def __iter__(self):
        while not self._stop:
            t = time.time_ns()
            yield self._frame, t
            if self._period_s:
                time.sleep(self._period_s)

    def stop(self) -> None:
        self._stop = True

    def info(self) -> CameraInfo:
        return CameraInfo(width=self._width, height=self._height, fps=float(self._fps))


def _make_run_id(backend_name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{backend_name}"


# MediaPipe-style 21-keypoint hand topology. If the YOLO model was trained
# with a different point order, edges drawn with these indices will look
# wrong even when the points themselves are placed correctly — that's why
# we also colour each finger group and (optionally) label index numbers.
_HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# (B, G, R) — one colour per finger group + wrist palette.
_FINGER_COLOR: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),                                            # wrist (white)
    **{i: (0,   0,   255) for i in (1, 2, 3, 4)},                  # thumb  (red)
    **{i: (0,   255, 0)   for i in (5, 6, 7, 8)},                  # index  (green)
    **{i: (255, 0,   0)   for i in (9, 10, 11, 12)},               # middle (blue)
    **{i: (0,   255, 255) for i in (13, 14, 15, 16)},              # ring   (yellow)
    **{i: (255, 0,   255) for i in (17, 18, 19, 20)},              # pinky  (magenta)
}


def _draw_preview(frame, result, backend_name: str, *, show_indices: bool = False):
    import cv2

    # Backends mirror internally; mirror our copy so keypoint coords line up.
    preview = cv2.flip(frame, 1) if result.metadata.get("mirror") else frame.copy()

    if result.detected and result.keypoints_xy:
        pts = [(int(x), int(y)) for x, y in result.keypoints_xy]
        # Edges coloured by destination point's finger group — easy to see
        # if the model's index order differs from the MediaPipe convention.
        for a, b in _HAND_EDGES:
            cv2.line(preview, pts[a], pts[b], _FINGER_COLOR.get(b, (200, 200, 200)), 2)
        for i, p in enumerate(pts):
            cv2.circle(preview, p, 4, _FINGER_COLOR.get(i, (0, 255, 0)), -1)
            if show_indices:
                cv2.putText(preview, str(i), (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    fv = result.finger_values
    overlay = (
        f"{backend_name}  T:{fv[0]} I:{fv[1]} M:{fv[2]} R:{fv[3]} P:{fv[4]}"
        f"  conf={result.confidence:.2f}  det={int(result.detected)}"
    )
    cv2.putText(preview, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2, cv2.LINE_AA)
    legend = "T=red I=green M=blue R=yellow P=magenta"
    cv2.putText(preview, legend, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imshow("RoboTime preview", preview)
    return cv2.waitKey(1) & 0xFF


def _stream_thread(
    client: NetworkClient,
    outbound: "queue.Queue",
    pending: dict[int, FrameRecord],
    pending_lock: threading.Lock,
    logger: LatencyLogger,
    print_every: int,
    done_event: threading.Event,
) -> None:
    """Drain the gRPC reply stream and merge Ack timestamps into records."""

    def gen() -> Iterator[hand_control_pb2.HandData]:
        while True:
            item = outbound.get()
            if item is None:
                return
            yield item

    n = 0
    try:
        for ack in client.stream(gen()):
            with pending_lock:
                rec = pending.pop(ack.frame_id, None)
            if rec is None:
                continue
            rec.server_receive_time_ns = ack.server_receive_time_ns
            rec.serial_write_done_time_ns = ack.serial_write_done_time_ns
            rec.robot_ack_time_ns = ack.robot_ack_time_ns
            logger.log(rec)
            n += 1
            if print_every and n % print_every == 0:
                e2e = rec.end_to_end_latency_ns() / 1e6
                print(
                    f"[runner] #{n:05d} frame={rec.frame_id} "
                    f"e2e={e2e:7.2f} ms warmup={rec.warmup}"
                )
    finally:
        done_event.set()


def run(
    backend_name: str,
    *,
    source: str = "camera",
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    frames: Optional[int] = None,
    duration_s: Optional[float] = None,
    warmup_frames: int = 30,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    server_port: int = DEFAULT_SERVER_PORT,
    runs_dir: Path = Path("runs"),
    print_every: int = 50,
    show: bool = False,
    show_indices: bool = False,
    model_path: Optional[str] = None,
    mirror: Optional[bool] = None,
) -> int:
    run_id = _make_run_id(backend_name)
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    metadata_path = out_dir / "run_metadata.json"

    backend_kwargs: dict = {}
    if model_path:
        backend_kwargs["model_path"] = model_path
    if mirror is not None:
        backend_kwargs["mirror"] = mirror
    backend: VisionBackend = get_backend(backend_name, **backend_kwargs)
    print(f"[runner] backend={backend.name} model_version={backend.model_version}")

    src: FrameSource = (
        FakeSource(fps=fps) if source == "fake"
        else CameraSource(camera_index, width, height, fps)
    )

    client = NetworkClient(server_address, server_port)
    logger = LatencyLogger(csv_path=csv_path)

    pending: dict[int, FrameRecord] = {}
    pending_lock = threading.Lock()
    outbound: "queue.Queue[Optional[hand_control_pb2.HandData]]" = queue.Queue(maxsize=1024)
    done_event = threading.Event()

    reader = threading.Thread(
        target=_stream_thread,
        args=(client, outbound, pending, pending_lock, logger, print_every, done_event),
        daemon=True,
    )
    reader.start()

    start_time_ns = time.time_ns()
    deadline_ns = start_time_ns + int(duration_s * 1e9) if duration_s else None
    frame_id = 0
    sent = 0
    try:
        for frame, capture_time_ns in src:
            if deadline_ns and capture_time_ns >= deadline_ns:
                break

            result = backend.process(frame, capture_time_ns)
            client_send_time_ns = time.time_ns()
            warmup = frame_id < warmup_frames

            rec = FrameRecord(
                run_id=run_id,
                backend=backend.name,
                frame_id=frame_id,
                warmup=warmup,
                detected=result.detected,
                confidence=result.confidence,
                capture_time_ns=result.capture_time_ns,
                vision_done_time_ns=result.vision_done_time_ns,
                client_send_time_ns=client_send_time_ns,
                clock_offset_ns=get_offset_ns("client"),
            )
            with pending_lock:
                pending[frame_id] = rec

            outbound.put(
                hand_control_pb2.HandData(
                    frame_id=frame_id,
                    backend_name=backend.name,
                    finger_values=result.finger_values,
                    confidence=result.confidence,
                    capture_time_ns=result.capture_time_ns,
                    vision_done_time_ns=result.vision_done_time_ns,
                    client_send_time_ns=client_send_time_ns,
                    warmup=warmup,
                )
            )

            if show:
                key = _draw_preview(frame, result, backend.name,
                                    show_indices=show_indices)
                if key == ord('q'):
                    break

            frame_id += 1
            sent += 1
            if frames is not None and sent >= frames:
                break
    except KeyboardInterrupt:
        print("[runner] interrupted; finishing in-flight frames")
    finally:
        outbound.put(None)
        done_event.wait(timeout=5.0)
        end_time_ns = time.time_ns()
        src.stop()
        backend.close()
        client.close()
        logger.close()
        if show:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    metadata = RunMetadata(
        run_id=run_id,
        backend=backend.name,
        model_version=backend.model_version,
        camera=src.info(),
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        frame_count=sent,
        warmup_frames=min(warmup_frames, sent),
        target_frames=frames,
        target_seconds=duration_s,
        server_address=f"{server_address}:{server_port}",
    )
    metadata.write(metadata_path)

    summary = logger.summarise()
    print(summary.format())
    print(f"[runner] csv      -> {csv_path}")
    print(f"[runner] metadata -> {metadata_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=available_backends(),
        help="Vision backend to run.",
    )
    parser.add_argument("--source", choices=["camera", "fake"], default="camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=None,
                        help="Stop after N frames (after warmup).")
    parser.add_argument("--duration", type=float, default=None, dest="duration_s",
                        help="Stop after N seconds.")
    parser.add_argument("--warmup", type=int, default=30, dest="warmup_frames",
                        help="Number of leading frames marked warmup=true.")
    parser.add_argument("--server-address", default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--show", action="store_true",
                        help="Open a preview window with keypoints + servo overlay (press q to quit).")
    parser.add_argument("--show-indices", action="store_true",
                        help="With --show, label each keypoint with its index (0-20). "
                             "Useful for debugging whether a YOLO model's point order "
                             "matches the MediaPipe convention.")
    parser.add_argument("--model-path", default=None,
                        help="Override the backend's default model file. "
                             "For yolo: path to a .pt file (default weights/yolo_hand_pose.pt). "
                             "For mediapipe: path to a .task file (default weights/hand_landmarker.task).")
    mirror_grp = parser.add_mutually_exclusive_group()
    mirror_grp.add_argument("--mirror", dest="mirror", action="store_true", default=None,
                            help="Force horizontal flip on input frames (selfie view). "
                                 "Default depends on the backend (typically on).")
    mirror_grp.add_argument("--no-mirror", dest="mirror", action="store_false",
                            help="Disable horizontal flip — feed the camera frame to the model as-is.")
    args = parser.parse_args()

    return run(
        args.backend,
        source=args.source,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        duration_s=args.duration_s,
        warmup_frames=args.warmup_frames,
        server_address=args.server_address,
        server_port=args.server_port,
        runs_dir=args.runs_dir,
        print_every=args.print_every,
        show=args.show,
        show_indices=args.show_indices,
        model_path=args.model_path,
        mirror=args.mirror,
    )


if __name__ == "__main__":
    sys.exit(main())
