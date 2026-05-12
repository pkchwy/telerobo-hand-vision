"""Serial bridge to the robot (real Arduino or `fake_robot.py`).

Writes a comma-separated servo command and reads a single ACK line back.
The Arduino firmware (`arduino/lehand.ino`) and the simulator
(`src/sim/fake_robot.py`) both reply with `Servos updated: ...\\n`, which
is what we treat as the per-frame ACK.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import serial


@dataclass
class WriteResult:
    serial_write_done_time_ns: int
    robot_ack_time_ns: Optional[int]
    ack_line: Optional[str]


class SerialRobot:
    def __init__(
        self,
        port: str,
        baud: int = 9600,
        ack_timeout_s: float = 0.2,
    ) -> None:
        self.port_name = port
        self.baud = baud
        self.ack_timeout_s = ack_timeout_s
        self._ser = serial.Serial(port, baud, timeout=ack_timeout_s)

    @property
    def is_open(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    def write_command(self, finger_values: list[int]) -> WriteResult:
        command = ",".join(str(int(v)) for v in finger_values) + "\n"
        self._ser.write(command.encode("ascii"))
        self._ser.flush()
        write_done = time.time_ns()

        line = self._ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            return WriteResult(
                serial_write_done_time_ns=write_done,
                robot_ack_time_ns=time.time_ns(),
                ack_line=line,
            )
        return WriteResult(
            serial_write_done_time_ns=write_done,
            robot_ack_time_ns=None,
            ack_line=None,
        )

    def close(self) -> None:
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass

    def __enter__(self) -> "SerialRobot":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
