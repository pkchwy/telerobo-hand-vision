#!/bin/bash
echo "Sistem güncelleniyor..."
sudo apt update
sudo apt upgrade -y

echo "Python 3.12 ve pip kontrol ediliyor..."
sudo apt install -y python3.12 python3.12-venv python3-pip

echo "Sistem kütüphaneleri kuruluyor..."
sudo apt install -y build-essential python3.12-dev libgl1

echo "Virtual environment oluşturuluyor (Python 3.12)..."
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "Python kütüphaneleri kuruluyor..."
pip install pyserial grpcio grpcio-tools protobuf mediapipe opencv-python numpy pyshark ultralytics

echo "gRPC stub'ları üretiliyor..."
./tools/regen_proto.sh

echo "USB port izinleri ayarlanıyor..."
sudo usermod -a -G dialout $USER
sudo usermod -a -G tty $USER

