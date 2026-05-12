"""Per-frame latency CSV logger + run-end summary."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CSV_COLUMNS = [
    "run_id",
    "backend",
    "frame_id",
    "warmup",
    "detected",
    "confidence",
    "capture_time_ns",
    "vision_done_time_ns",
    "client_send_time_ns",
    "server_receive_time_ns",
    "serial_write_done_time_ns",
    "robot_ack_time_ns",
    "vision_latency_ns",
    "network_latency_ns",
    "serial_latency_ns",
    "command_accept_latency_ns",
    "end_to_end_latency_ns",
    "clock_offset_ns",
]


@dataclass
class FrameRecord:
    run_id: str
    backend: str
    frame_id: int
    warmup: bool = False
    detected: bool = True
    confidence: float = 0.0
    capture_time_ns: int = 0
    vision_done_time_ns: int = 0
    client_send_time_ns: int = 0
    server_receive_time_ns: int = 0
    serial_write_done_time_ns: int = 0
    robot_ack_time_ns: int = 0
    clock_offset_ns: int = 0

    def vision_latency_ns(self) -> int:
        if self.vision_done_time_ns and self.capture_time_ns:
            return self.vision_done_time_ns - self.capture_time_ns
        return 0

    def network_latency_ns(self) -> int:
        if self.server_receive_time_ns and self.client_send_time_ns:
            return self.server_receive_time_ns - self.client_send_time_ns
        return 0

    def serial_latency_ns(self) -> int:
        if self.serial_write_done_time_ns and self.server_receive_time_ns:
            return self.serial_write_done_time_ns - self.server_receive_time_ns
        return 0

    def command_accept_latency_ns(self) -> int:
        if self.robot_ack_time_ns and self.serial_write_done_time_ns:
            return self.robot_ack_time_ns - self.serial_write_done_time_ns
        return 0

    def end_to_end_latency_ns(self) -> int:
        if self.robot_ack_time_ns and self.capture_time_ns:
            return self.robot_ack_time_ns - self.capture_time_ns
        return 0

    def as_row(self) -> list:
        return [
            self.run_id,
            self.backend,
            self.frame_id,
            int(self.warmup),
            int(self.detected),
            f"{self.confidence:.4f}",
            self.capture_time_ns,
            self.vision_done_time_ns,
            self.client_send_time_ns,
            self.server_receive_time_ns,
            self.serial_write_done_time_ns,
            self.robot_ack_time_ns,
            self.vision_latency_ns(),
            self.network_latency_ns(),
            self.serial_latency_ns(),
            self.command_accept_latency_ns(),
            self.end_to_end_latency_ns(),
            self.clock_offset_ns,
        ]


@dataclass
class LatencyLogger:
    csv_path: Path
    records: list[FrameRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        self._fp = self.csv_path.open("a", newline="")
        self._writer = csv.writer(self._fp)

    def log(self, record: FrameRecord) -> None:
        self.records.append(record)
        self._writer.writerow(record.as_row())
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def summarise(self) -> "RunSummary":
        return RunSummary.from_records(self.records)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class RunSummary:
    n_total: int
    n_measured: int
    n_dropped: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    jitter_ms: float
    throughput_msg_s: float

    @classmethod
    def from_records(cls, records: Iterable[FrameRecord]) -> "RunSummary":
        records = [r for r in records if not r.warmup]
        n_total = len(records)
        e2e_ms = [r.end_to_end_latency_ns() / 1e6 for r in records if r.end_to_end_latency_ns() > 0]
        n_measured = len(e2e_ms)
        n_dropped = n_total - n_measured

        if not e2e_ms:
            return cls(n_total, n_measured, n_dropped, 0, 0, 0, 0, 0, 0)

        first_ts = min(r.capture_time_ns for r in records if r.capture_time_ns)
        last_ts = max(r.robot_ack_time_ns for r in records if r.robot_ack_time_ns)
        duration_s = (last_ts - first_ts) / 1e9 if last_ts > first_ts else 0.0
        throughput = n_measured / duration_s if duration_s > 0 else 0.0

        return cls(
            n_total=n_total,
            n_measured=n_measured,
            n_dropped=n_dropped,
            p50_ms=_percentile(e2e_ms, 0.50),
            p90_ms=_percentile(e2e_ms, 0.90),
            p95_ms=_percentile(e2e_ms, 0.95),
            p99_ms=_percentile(e2e_ms, 0.99),
            jitter_ms=statistics.pstdev(e2e_ms) if len(e2e_ms) > 1 else 0.0,
            throughput_msg_s=throughput,
        )

    def format(self) -> str:
        lines = [
            "=" * 60,
            "Run summary (warmup excluded)",
            "=" * 60,
            f"  total frames    : {self.n_total}",
            f"  measured frames : {self.n_measured}",
            f"  dropped         : {self.n_dropped}",
            f"  throughput      : {self.throughput_msg_s:7.2f} msg/s",
            f"  end-to-end p50  : {self.p50_ms:7.2f} ms",
            f"  end-to-end p90  : {self.p90_ms:7.2f} ms",
            f"  end-to-end p95  : {self.p95_ms:7.2f} ms",
            f"  end-to-end p99  : {self.p99_ms:7.2f} ms",
            f"  jitter (stdev)  : {self.jitter_ms:7.2f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)
