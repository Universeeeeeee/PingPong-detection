#!/usr/bin/env bash
set -u

echo '[OS]'
uname -a
test -r /etc/os-release && sed -n '1,12p' /etc/os-release

echo '[NVIDIA]'
command -v nvidia-smi || true
nvidia-smi 2>/dev/null || true
command -v nvcc || true
nvcc --version 2>/dev/null || true

echo '[ZED SDK]'
test -d /usr/local/zed && ls -ld /usr/local/zed || true
command -v ZED_Diagnostic || true
command -v ZED_Explorer || true
python3 -c 'import pyzed.sl as sl; print("pyzed OK; SDK", sl.Camera.get_sdk_version())' 2>&1 || true

echo '[USB devices]'
command -v lsusb || true
lsusb 2>/dev/null || true

echo '[No packages or system settings were changed]'
