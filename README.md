# telerobo-hand-vision

A modular, low-latency gRPC pipeline for driving a robotic hand from any
**pluggable vision backend** (MediaPipe, YOLO, MMPose, ...). Designed as a research
platform for fairly comparing different hand-tracking approaches over the
**same** network + actuation path, with per-stage latency measurement.

## Architecture

```
Camera ─▶ VisionBackend ─▶ gRPC Client ─▶ gRPC Server ─▶ Serial ─▶ Arduino
                                                           ▲           │
                                                           │           ▼
                                                          ACK ◀───── "Servos updated: ..."
```

Stage timestamps (ns) flow with every frame, allowing offline analysis of
where time is actually spent (`vision`, `network`, `serial`, `command_accept`).

## Project layout

```
proto/                hand_control.proto (staged-timestamp message schema)
src/
  vision/             VisionBackend interface + MediaPipe + YOLO + MMPose (RTMPose/HRNet/Hourglass/ResNet presets)
  network/            gRPC client / server + generated stubs
  robot/              Serial bridge to Arduino with ACK parsing
  metrics/            CSV logger, run_metadata.json, time-sync hook
  sim/                fake_robot.py (Arduino simulator over a PTY)
arduino/              lehand.ino
tools/
  run_backend.py      Live camera runner: backend -> CSV -> summary
  regen_proto.sh      Regenerate gRPC stubs after editing the .proto
  wsparser.py         Offline analysis of .pcapng captures
docs/
  time_sync_integration.md
runs/<run_id>/        metrics.csv + run_metadata.json per run
```

## Setup

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
./tools/regen_proto.sh
```

## Vision-only preview (no robot, single terminal)

If you only want to see what a backend detects on camera — e.g. to
sanity-check a new `.pt` file — skip the gRPC/Arduino setup entirely:

```bash
python -u -m tools.preview_vision --backend yolo
python -u -m tools.preview_vision --backend yolo --model-path weights/best.pt
python -u -m tools.preview_vision --backend mediapipe --show-indices
python -u -m tools.preview_vision --backend rtmpose      # MMPose preset
python -u -m tools.preview_vision --backend hrnet        # MMPose preset
python -u -m tools.preview_vision --backend hourglass    # MMPose preset
python -u -m tools.preview_vision --backend resnet       # MMPose preset
```

This bypasses the network + serial path; no latency CSV is written.

### Available backends

| Backend name | Architecture | Notes |
|---|---|---|
| `mediapipe` | MediaPipe Hands (Tasks API) | Regression CNN, fast baseline |
| `yolo` | Ultralytics YOLO-Pose | Anchor-based one-stage, trainable in-repo |
| `rtmpose` | RTMPose-m (Hand5) | SimCC, transformer-distilled — fastest MMPose option |
| `hrnet` | HRNetv2-w18 (COCO-Wholebody-Hand) | High-resolution heatmap CNN — accurate baseline |
| `hourglass` | Stacked Hourglass-52 | OpenPose-family multi-scale heatmaps |
| `resnet` | ResNet-50 SimpleBaseline | Classic topdown CNN baseline |

All six backends produce the same 21-keypoint MediaPipe topology and run
through the same `landmarks_to_servo` post-processing, so latency and
servo-angle comparisons are apples-to-apples — only the model changes.
MMPose presets auto-download weights from openmmlab on first invocation.

To load a custom MMPose config not covered by the presets, edit
`MMPOSE_PRESETS` in `src/vision/mmpose_backend.py` and add a new entry, then
register it in `src/vision/__init__.py`.

## Running it (three terminals)

**Order matters.** Terminal 1 must be started *first* — it creates the
`/tmp/robot_write` symlink that Terminal 2 opens. If you skip it (or start
Terminal 2 first), the server will crash with
`SerialException: could not open port /tmp/robot_write`.

Activate the venv in **every** terminal: `source venv/bin/activate`.

### Terminal 1 — simulated Arduino (or real hardware)

```bash
# Simulated:
python -u src/sim/fake_robot.py
# Skip this terminal entirely if you have a real Arduino on /dev/ttyACM0;
# then pass --robot-port /dev/ttyACM0 in Terminal 2.
```

### Terminal 2 — gRPC server

```bash
python -u -m src.network.server --robot-port /tmp/robot_write
# real hardware: --robot-port /dev/ttyACM0
```

### Terminal 3 — runner (pick one)

```bash
# (a) No camera, fake frames — quick pipeline sanity check
python -u -m tools.run_backend --backend yolo --source fake \
    --frames 100 --warmup 10 --fps 60

# (b) Live camera with on-screen preview (MediaPipe). Press q to quit.
python -u -m tools.run_backend --backend mediapipe \
    --camera-index 0 --width 640 --height 480 --fps 30 \
    --warmup 30 --show

# (c) Same but bounded to 1000 frames (writes a complete run)
python -u -m tools.run_backend --backend mediapipe \
    --camera-index 0 --width 640 --height 480 --fps 30 \
    --frames 1000 --warmup 30 --show
```

Useful runner flags: `--show` opens a preview window with keypoints and the
five servo values overlaid, `--show-indices` labels each keypoint 0–20
(handy for checking a YOLO model's point order), `--duration 30` stops
after 30 s instead of a fixed frame count, `--mirror` / `--no-mirror`
overrides the backend's default flip.

The runner prints a p50/p90/p95/p99 summary and writes
`runs/<run_id>/metrics.csv` + `run_metadata.json`. Compare backends by
re-running with `--backend yolo` and inspecting the two `runs/<run_id>/`
folders side by side.

## Per-frame CSV columns

`run_id, backend, frame_id, warmup, detected, confidence,`
`capture_time_ns, vision_done_time_ns, client_send_time_ns,`
`server_receive_time_ns, serial_write_done_time_ns, robot_ack_time_ns,`
`vision_latency_ns, network_latency_ns, serial_latency_ns,`
`command_accept_latency_ns, end_to_end_latency_ns, clock_offset_ns`

`end_to_end_latency_ns = robot_ack_time_ns - capture_time_ns`. This is
the time from "frame captured" to "robot acknowledged the command", which
is the most honest single-number latency we can produce without extra
sensors. It does **not** include physical servo travel time.

## Adding a new vision backend

1. Subclass `VisionBackend` in a new module under `src/vision/`.
2. Implement `process(frame, capture_time_ns) -> VisionResult`.
3. Register the constructor in `src/vision/__init__.py::_REGISTRY`.

The runner will pick it up via `--backend <name>` automatically.

## Time synchronisation

Two-machine clock synchronisation is intentionally out of scope for this
codebase; another effort handles it. The integration point is
`src/metrics/time_sync_hook.py` — see [docs/time_sync_integration.md](docs/time_sync_integration.md).
Until it is filled in, the `clock_offset_ns` column is `0` and
`run_metadata.time_sync_method` is `"none"`.

## Out of scope (for now)

- Two-machine clock synchronisation algorithm (separate effort).
- Measuring the physical start of servo motion (encoder / IMU / high-speed camera).
- Full YOLO -> finger mapping (skeleton only).
- Plot generation, dashboards, automated cross-backend reports — open the CSVs in your tool of choice.
