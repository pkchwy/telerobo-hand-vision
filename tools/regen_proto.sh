#!/usr/bin/env bash
# Regenerate gRPC Python stubs from proto/hand_control.proto into src/network/.
# Run from repo root after activating the project venv.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m grpc_tools.protoc \
  -Iproto \
  --python_out=src/network \
  --grpc_python_out=src/network \
  proto/hand_control.proto

# Generated grpc stub uses bare `import hand_control_pb2`; rewrite to relative
# import so the file works when consumed as src.network.hand_control_pb2_grpc.
sed -i 's/^import hand_control_pb2 as/from . import hand_control_pb2 as/' \
  src/network/hand_control_pb2_grpc.py

echo "Regenerated src/network/hand_control_pb2*.py"
