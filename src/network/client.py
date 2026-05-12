"""gRPC client wrapper.

The client is a thin library used by `tools/run_backend.py`: caller hands
an iterable of `HandData`, the client returns an iterable of `Ack`. The
runner is responsible for correlating Acks back to the per-frame state by
`frame_id`.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import grpc

from src.network import hand_control_pb2, hand_control_pb2_grpc


DEFAULT_SERVER_ADDRESS = "localhost"
DEFAULT_SERVER_PORT = 50051


class NetworkClient:
    def __init__(
        self,
        address: str = DEFAULT_SERVER_ADDRESS,
        port: int = DEFAULT_SERVER_PORT,
        ready_timeout_s: float = 10.0,
    ) -> None:
        options = [
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
        ]
        self.channel = grpc.insecure_channel(f"{address}:{port}", options=options)
        grpc.channel_ready_future(self.channel).result(timeout=ready_timeout_s)
        self.stub = hand_control_pb2_grpc.HandControllerStub(self.channel)

    def stream(
        self,
        hand_data_iter: Iterable[hand_control_pb2.HandData],
    ) -> Iterator[hand_control_pb2.Ack]:
        return self.stub.StreamHandData(hand_data_iter)

    def close(self) -> None:
        try:
            self.channel.close()
        except Exception:
            pass

    def __enter__(self) -> "NetworkClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["NetworkClient", "hand_control_pb2", "DEFAULT_SERVER_ADDRESS", "DEFAULT_SERVER_PORT"]
