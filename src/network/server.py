"""gRPC server: bridges HandData stream to the robot and replies with Ack.

Each incoming HandData triggers a serial write to the robot. The server
captures `server_receive_time_ns`, `serial_write_done_time_ns`, and (when
the robot replies) `robot_ack_time_ns`, then sends an Ack message back so
the client can write the full per-frame latency row.
"""

from __future__ import annotations

import argparse
import time
from concurrent import futures

import grpc

from src.network import hand_control_pb2, hand_control_pb2_grpc
from src.robot.serial_robot import SerialRobot


DEFAULT_ROBOT_PORT = "/tmp/robot_write"
DEFAULT_ROBOT_BAUD = 9600
DEFAULT_GRPC_PORT = 50051


class HandControllerServicer(hand_control_pb2_grpc.HandControllerServicer):
    def __init__(self, robot: SerialRobot) -> None:
        self.robot = robot
        self.message_count = 0
        self.start_time_ns: int | None = None

    def StreamHandData(self, request_iterator, context):
        print("[server] client connected")
        self.start_time_ns = time.time_ns()

        for hand_data in request_iterator:
            server_receive_time_ns = time.time_ns()
            self.message_count += 1

            if self.robot.is_open:
                result = self.robot.write_command(list(hand_data.finger_values))
                serial_write_done_time_ns = result.serial_write_done_time_ns
                robot_ack_time_ns = result.robot_ack_time_ns or 0
            else:
                serial_write_done_time_ns = time.time_ns()
                robot_ack_time_ns = 0

            if self.message_count % 50 == 0:
                elapsed_s = (time.time_ns() - (self.start_time_ns or server_receive_time_ns)) / 1e9
                rate = self.message_count / elapsed_s if elapsed_s > 0 else 0.0
                e2e_ms = (
                    (robot_ack_time_ns - hand_data.capture_time_ns) / 1e6
                    if robot_ack_time_ns and hand_data.capture_time_ns
                    else 0.0
                )
                print(
                    f"[server] #{self.message_count:05d} | {rate:6.1f} msg/s "
                    f"| backend={hand_data.backend_name} "
                    f"| e2e={e2e_ms:7.2f} ms"
                )

            yield hand_control_pb2.Ack(
                frame_id=hand_data.frame_id,
                success=True,
                server_receive_time_ns=server_receive_time_ns,
                serial_write_done_time_ns=serial_write_done_time_ns,
                robot_ack_time_ns=robot_ack_time_ns,
            )

        print(f"[server] client disconnected after {self.message_count} messages")


def serve(robot_port: str, robot_baud: int, grpc_port: int) -> None:
    robot = SerialRobot(robot_port, robot_baud)
    print(f"[server] robot serial bound to {robot_port} @ {robot_baud}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = HandControllerServicer(robot)
    hand_control_pb2_grpc.add_HandControllerServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{grpc_port}")
    server.start()
    print(f"[server] gRPC listening on :{grpc_port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("[server] shutting down")
    finally:
        server.stop(0)
        robot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-baud", type=int, default=DEFAULT_ROBOT_BAUD)
    parser.add_argument("--grpc-port", type=int, default=DEFAULT_GRPC_PORT)
    args = parser.parse_args()
    serve(args.robot_port, args.robot_baud, args.grpc_port)


if __name__ == "__main__":
    main()
