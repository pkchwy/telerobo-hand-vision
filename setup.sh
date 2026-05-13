#!/bin/bash
# One-shot environment setup for the telerobo-hand-vision project.
# Installs the gRPC pipeline + all three vision backends (MediaPipe, YOLO,
# MMPose) into a single venv.
#
# Tested on Debian/Ubuntu. On Arch-based systems (e.g. CachyOS), provide
# python3.11 via pacman/AUR or pyenv and skip the apt steps.
set -e

echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo "Installing Python 3.11 and pip..."
sudo apt install -y python3.11 python3.11-venv python3-pip

echo "Installing system build dependencies..."
sudo apt install -y build-essential python3.11-dev libgl1

echo "Creating virtual environment (Python 3.11)..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel "setuptools>=70"

echo "Installing PyTorch 2.4 + CUDA 12.1..."
# Pinned because mmcv only publishes prebuilt wheels for specific
# torch+cuda combos; this version works for YOLO (ultralytics) too.
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121

echo "Installing core pipeline + vision dependencies..."
# gRPC + serial + camera + mediapipe + yolo, all in one shot.
# numpy<2 and opencv<4.11 are required by xtcocotools' numpy 1.x ABI.
pip install pyserial grpcio grpcio-tools protobuf mediapipe \
  "opencv-python<4.11" "opencv-contrib-python<4.11" "numpy<2" \
  pyshark ultralytics

echo "Installing MMPose stack..."
# Direct install from OpenMMLab's wheel index (skipping openmim, which
# downgrades setuptools to 60.2.0 and breaks mmcv's source build).
pip install mmengine
pip install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
# chumpy's old setup.py can't find pip inside the isolated build env.
pip install chumpy --no-build-isolation
pip install "mmdet==3.3.0" mmpose

echo "Patching mmdet's mmcv upper-bound check (2.2.0 -> 2.3.0)..."
# mmdet 3.3.0 hard-asserts mmcv<2.2.0; we use exactly 2.2.0. APIs are
# compatible, so we just relax the version assert.
MMDET_INIT="$(python -c 'import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), "__init__.py"))')"
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET_INIT"

echo "Downgrading setuptools to <70..."
# setuptools 80+ removed pkg_resources; mmengine.utils.package_utils
# imports it at runtime, so the venv needs an older setuptools.
pip install "setuptools<70"

echo "Regenerating gRPC stubs..."
./tools/regen_proto.sh

echo "Configuring USB port permissions..."
sudo usermod -a -G dialout $USER
sudo usermod -a -G tty $USER

echo
echo "Setup complete. To verify any backend:"
echo "  source venv/bin/activate"
echo "  python -u -m tools.preview_vision --backend mediapipe"
echo "  python -u -m tools.preview_vision --backend yolo"
echo "  python -u -m tools.preview_vision --backend mmpose"
