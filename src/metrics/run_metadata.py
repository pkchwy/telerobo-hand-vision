"""Per-run metadata file (`run_metadata.json`).

Captures the constant context for a run so that several months from now
the matching CSV can still be interpreted without guessing: which
backend, which model version, which camera settings, which machine, and
how the time-sync hook was configured.
"""

from __future__ import annotations

import json
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .time_sync_hook import time_sync_method


@dataclass
class CameraInfo:
    width: int = 0
    height: int = 0
    fps: float = 0.0


@dataclass
class MachineInfo:
    hostname: str = field(default_factory=socket.gethostname)
    os: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    python: str = field(default_factory=lambda: sys.version.split()[0])


@dataclass
class RunMetadata:
    run_id: str
    backend: str
    model_version: str = "unknown"
    camera: CameraInfo = field(default_factory=CameraInfo)
    machine: MachineInfo = field(default_factory=MachineInfo)
    start_time_ns: int = 0
    end_time_ns: int = 0
    frame_count: int = 0
    warmup_frames: int = 0
    target_frames: Optional[int] = None
    target_seconds: Optional[float] = None
    server_address: str = ""
    time_sync_method: str = field(default_factory=time_sync_method)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False) + "\n")
