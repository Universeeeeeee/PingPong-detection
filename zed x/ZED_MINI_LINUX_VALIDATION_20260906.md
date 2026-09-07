# ZED Mini Linux 接入基线（2026-09-06）

## 结论

ZED Mini 已在 Linux 上完成从 USB/UVC 到 ZED SDK/Python API 的端到端验证。USB 3 链路稳定完成 300 帧 `1344×376 @ 100 FPS` UVC 采集；ZED SDK 5.4.1 又完成 `HD720 @ 60 FPS` 的同步左右图和深度采集。稳定采集区间内，SDK 丢帧增量和时间戳估算丢帧数均为 0。

## 主机

```text
OS: Ubuntu 22.04.5 LTS
Kernel: 6.8.0-124-generic
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
NVIDIA driver: 580.159.03
nvidia-smi maximum supported CUDA: 13.0
nvcc toolkit: CUDA 11.5 (V11.5.119)
Docker: 29.1.3
NVIDIA Container Toolkit: not detected
System Python: /usr/bin/python3
Conda base Python: 3.13.9
```

`nvidia-smi` 中显示的 CUDA 13.0 是驱动支持上限，不证明 CUDA 13 toolkit 已安装。实际 `nvcc` 是 11.5。安装 ZED SDK 前必须选择明确匹配的 SDK/CUDA 方案，不能以升级整台主机来试错。

## USB 和权限

```text
USB video: 2b03:f682 STEREOLABS ZED-M camera
USB HID:   2b03:f681 STEREOLABS ZED-M HID Interface
Video bus: USB 3, 5000M
Driver: uvcvideo
Video nodes: /dev/video0, /dev/video1
User pc groups: includes video and plugdev
```

相机的 UVC video interface 位于 5 Gbit/s 链路；HID 接口位于独立的 USB 2/HID 路径。这是正常的复合设备枚举结果。

## UVC 公布模式

```text
2560×720: 60/30/15 FPS
1344×376: 100/60/30/15 FPS
3840×1080: 30/15 FPS
4416×1242: 15 FPS
Pixel format: YUYV 4:2:2
```

上述宽度是左右目 side-by-side 总宽度；`1344×376` 对应每目 `672×376`。

## 100 FPS 实测

测试命令把数据写到 `/dev/null`，未保存图像：

```bash
v4l2-ctl --device=/dev/video0 \
  --set-fmt-video=width=1344,height=376,pixelformat=YUYV \
  --set-parm=100 \
  --stream-mmap=4 \
  --stream-count=300 \
  --stream-to=/dev/null \
  --verbose
```

结果：

```text
Frames: 300
Sequence: continuous 0..299
Frame interval: approximately 10.00 ms
Final reported rate: approximately 99.83 FPS
Persistent output: none
```

V4L2 输出将内核时间戳标记为 `ts-monotonic, ts-src-soe`。这只能描述当前 UVC/V4L2 路径，不能替代或重定义 ZED SDK `TIME_REFERENCE.IMAGE` 的语义；进入正式算法后仍以 SDK 文档定义和外部延迟测量为准。

## ZED SDK 与 Python 环境

```text
ZED SDK: 5.4.1, /usr/local/zed
SDK installer: Ubuntu 22, CUDA 12.8 build
pyzed: 5.4
Project Python: /home/pc/zed_mini/zed x/.venv/bin/python (Python 3.10.12)
Project NumPy: 2.2.6
Project OpenCV: opencv-python-headless 4.12.0.88
System Python/NumPy: unchanged
Conda base: unchanged
```

安装使用 `--runtime_only --skip_cuda --skip_od_module --skip_python`，随后把 `pyzed` 单独安装到项目虚拟环境。没有运行 `apt update`，没有替换 NVIDIA 驱动、CUDA toolkit、系统 Python 或 Conda 环境。`libsl_zed.so` 的动态依赖检查无缺失。

相机无关代码在 Linux 项目虚拟环境（pyzed 5.4、NumPy 2.2.6）完成 12 项单元测试，全部通过。

## ZED SDK 实机结果

设备：

```text
Model: ZED-M
Serial number: 13376675
Firmware: 1523
SDK state: AVAILABLE
Rectified per-eye resolution: 1280×720
Active stereo baseline: 0.063274279 m
```

无深度采集，240 帧 `HD720 @ 60 FPS`：

```text
Timestamps strictly increasing: yes
Timestamp interval min/median/max: 16.429/16.720/16.845 ms
Timestamp-estimated dropped frames: 0
SDK dropped count start/end/delta: 44/44/0
SDK current FPS: 60.24096
```

SDK 的 dropped count 是从首次 `grab()` 起累计的“没有进入 grab 的旧帧”。因此报告同时保留起点、终点和本次增量；不能把初始化后读到的起始累计值误当成本次稳定采集的丢帧数。

无深度高速采集，300 帧 `VGA @ 100 FPS`：

```text
Rectified per-eye resolution: 672×376
Timestamp interval min/median/max: 9.832/10.001/10.165 ms
Timestamp-estimated dropped frames: 0
SDK dropped count start/end/delta: 55/55/0
SDK current FPS: 100.0
```

`PERFORMANCE` 深度采集，30 帧 `HD720 @ 60 FPS`：

```text
Depth dtype/unit: float32 metres
Depth shape: 720×1280
Valid fraction in the measured scene: 0.9375933
Valid min/median/max: 0.9048/3.0334/9.9956 m
Timestamp-estimated dropped frames: 0
SDK dropped count delta: 0
```

SDK 5.4.1 将 `PERFORMANCE` 标记为 deprecated 并建议 `NEURAL`。当前没有启用 AI/TensorRT 安装分支；深度在迁移架构中仍是可选辅助信息，首阶段不以切换到 `NEURAL` 为前置条件。

复现命令：

```bash
cd "$HOME/zed_mini/zed x"
./.venv/bin/python zed_probe.py --list-devices
./.venv/bin/python zed_probe.py --serial 13376675 --resolution HD720 --fps 60 --frames 240
./.venv/bin/python zed_probe.py --serial 13376675 --resolution HD720 --fps 60 --depth --depth-mode PERFORMANCE --frames 30
./.venv/bin/python -m unittest discover -s tests -v
```

## 下一阶段

SDK 接入门禁已经通过。下一阶段是把真实 ZED 帧送入 ZED 专属检测/双目前端，并统一输出 `BallMeasurement3D`，再接公共确认、滤波和轨迹预测。仍不应为了迁移执行：

- `apt update` 或系统升级；
- NVIDIA 驱动替换；
- CUDA toolkit 替换；
- NVIDIA Container Toolkit 安装。

后续还需用真实球场景验证球检测、左右视差符号、三角化重投影误差、坐标系外参和端到端延迟。当前结果证明采集与数据契约可用，不等同于乒乓球算法精度已经验收。

接口烟雾测试已使用真实 ZED Mini 帧和 active rectified calibration，将人工指定的左右像素对转换为带协方差的 `BallMeasurement3D`。这证明 SDK 到公共测量契约的代码链路可运行；人工像素对不构成球检测或精度验证。

## ZED Mini 球检测首版

运行时代码严格要求 SDK 型号为 `ZED-M` 且 live 输入来自 USB。算法按 Mini 实际数据拓扑实现：左目全图橙球身份检测，右目只在 rectified 极线与 0.25--6 m 深度范围形成的 ROI 内匹配，随后检查正深度、重投影、物理球径和左右 patch NCC，最后输出 `source=zed_mini_rectified_stereo` 的 `BallMeasurement3D`。

无球场景 300 帧 `VGA @ 100 FPS` 稳态结果：

```text
Processing FPS: 99.674
SDK timestamp FPS: 99.829
Timestamp interval min/median/max: 9.829/10.002/10.200 ms
Timestamp-estimated dropped frames: 0
SDK dropped count delta: 0
3D measurements: 0
Median source.read: 5.318 ms
Median ball pipeline: 4.707 ms
```

Linux 中 17 项测试全部实际通过，其中包括同步橙球对生成 Mini 三维测量、右目无身份时禁止输出 3D、非极线候选拒绝和非 Mini 型号拒绝。无球测试只验收稳定性与假阳性，不代表真实球检出率已通过。
