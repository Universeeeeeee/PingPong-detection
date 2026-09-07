# ZED Mini 接入 Linux 快速清单

本清单的目标是验证采集边界，不修改 D455 部署，不升级系统软件。ZED Mini 用于提前验证 USB ZED 的 SDK、图像、标定和算法接口；它不能替代 ZED X 的 GMSL 链路、传感器和最终运动模糊验收。

2026-09-06 的 Linux 接入结果已记录在 `ZED_MINI_LINUX_VALIDATION_20260906.md`：USB 3/UVC 完成 300 帧 `1344×376 @ 100 FPS` 连续采集；ZED SDK 5.4.1 又完成 300 帧 `VGA @ 100 FPS` 和 `HD720 @ 60 FPS` 图像/深度验证。稳定采集区间内未测到丢帧。

所有 Python 命令使用项目环境：

```bash
cd "$HOME/zed_mini/zed x"
./.venv/bin/python --version
```

该环境内的已验证组合是 `pyzed 5.4 + NumPy 2.2.6 + opencv-python-headless 4.12.0.88`。不要改用 Ubuntu 自带的 `cv2`：它按 NumPy 1.x 构建，不能与当前 `pyzed` 要求的 NumPy 2.x 在同一进程运行。

## 1. 接线前：只读预检

在 Linux 上进入本目录后运行：

```bash
bash linux_readonly_preflight.sh
```

脚本只读取 OS、NVIDIA、CUDA、ZED SDK、Python API 和 USB 信息。它不使用 `sudo`，不运行 `apt`，不写系统配置。

如果 `pyzed.sl` 导入失败，先保存完整输出，不要立即安装。需要先根据 Ubuntu、GPU、CUDA、Python 和已有 ZED SDK 版本选择兼容方案。

## 2. 接入 ZED Mini 后：枚举设备

```bash
./.venv/bin/python zed_probe.py --list-devices
```

预期至少出现一个设备，记录：

- `serial_number`
- `camera_model`（ZED Mini 在 SDK 中通常显示为 `ZED_M`）
- `camera_state`

如果未枚举，先运行系统已经安装的 ZED Diagnostic；不要通过升级整台主机尝试碰运气。

## 3. 第一条实机路径：只取 rectified 图像，不算 SDK depth

ZED Mini 的首个高速候选配置：

```bash
./.venv/bin/python zed_probe.py \
  --input live \
  --resolution VGA \
  --fps 100 \
  --frames 300
```

这条命令验证：

- `VIEW.LEFT` 是 `uint8 BGRA`，进入算法前显式转换为 BGR；
- `VIEW.LEFT_GRAY/RIGHT_GRAY` 是同一次成功 `grab()` 的 rectified stereo pair；
- active rectified calibration 能转换为公共标定结构；
- SDK 图像时间戳严格递增；
- 实际 FPS 和 SDK dropped frame count 被记录。

重点检查 JSON：

```text
camera_model
width / height
configured_fps
sdk_current_fps
frames_read
timestamps_strictly_increasing
timestamp_step_ms
sdk_dropped_frame_count_start / sdk_dropped_frame_count / sdk_dropped_frame_count_delta
timestamp_estimated_dropped_frames
baseline_m
calibration_id
```

不要把 `sdk_image_timestamp_semantics=entire_image_available_in_pc_memory` 写成曝光时间。

## 4. ZED Mini 同步彩色双目球检测

默认打开实时预览窗口并持续运行；在窗口中按 `q` 或 `Esc` 退出。不要再传
`--frames 300`，因为固定帧数模式只适合基准测试：

```bash
./.venv/bin/python zed_mini_tracker.py \
  --serial 13376675 \
  --resolution VGA \
  --fps 100 \
  --warmup-frames 20
```

预览窗口会并排显示 Mini 左右彩色图、左目检测框、当前拒绝原因和测得的
`XYZ`。如果通过 SSH 或无桌面环境运行，显式关闭窗口并用 `Ctrl-C` 停止：

```bash
./.venv/bin/python zed_mini_tracker.py \
  --serial 13376675 \
  --resolution VGA \
  --fps 100 \
  --headless
```

需要可重复的短基准时才加 `--frames 300`；这种模式结束后会输出 JSON 报告。

这条路径严格校验 `ZED-M` 型号和 USB 输入。左目运行橙球身份检测；右目只在标定、极线和深度范围约束出的 ROI 中搜索对应候选，然后检查重投影、物理尺寸和左右灰度 patch NCC，最终才输出 `BallMeasurement3D(source=zed_mini_rectified_stereo)`。

无球基线实测为处理约 99.67 FPS、SDK 时间戳约 99.83 FPS、稳定阶段丢帧 0、错误三维测量 0。真实球验收仍需在现场持球和运动球条件下执行，不能用无球结果代替检出率。

## 5. 第二条实机路径：开启 SDK depth 作为辅助证据

图像路径稳定后再运行：

```bash
./.venv/bin/python zed_probe.py \
  --input live \
  --resolution VGA \
  --fps 100 \
  --depth \
  --depth-mode PERFORMANCE \
  --frames 300
```

这一步只判断当前 Linux 主机能否在指定模式下维持采集。若实际 FPS 明显下降，不把 depth 强留在高频主路径；高速左右图三角化仍是主路径，SDK depth 可降频或独立评估。

ZED confidence 的数值方向是“越高越不可信”，不能直接当作 `[0,1]` 的检测置信度。

## 6. 规则球桌识别（ZED Mini）

`--table-pose` 会使用左目 rectified 彩色图提出蓝色台面区域，再使用同一帧的 ZED 米制深度拟合平面，
并只接受尺寸符合已知 `2.74 × 1.525 m` 的外接矩形。初次确认要求完整、清晰的外轮廓；它不会用凸包或
历史位姿补全缺失边界。确认成功后，终端 JSON 的 `table.T_camera_table` 和每个球测量的
`position_table_m` 才有效。

先以低频桌面估计验证深度开销：

```bash
./.venv/bin/python zed_mini_tracker.py \
  --serial 13376675 \
  --resolution HD720 \
  --fps 60 \
  --table-pose \
  --table-hz 8
```

若颜色阈值不适配现场桌面，先只修改下面两个值，不要放宽尺寸验证：

```bash
  --table-hsv-low 85,80,40 \
  --table-hsv-high 135,255,255
```

无桌面、深度不足、桌面不完整或尺寸不符时，`table.valid=false`，球的 `position_table_m=null`；
这属于预期的安全行为。SDK depth 可能使高速图像链路降频，因此应先在 `HD720@60` 验证，再决定是否尝试
`VGA@100`。桌面识别不替代机器人 base 系标定。

## 7. SVO2 回放

```bash
./.venv/bin/python zed_probe.py \
  --input svo \
  --svo /absolute/path/to/capture.svo2 \
  --frames 300
```

默认非实时回放，便于确定性测试。若 SVO 不包含有效图像时间戳，当前 source 会明确报错，不会伪造时间。

## 8. 录制 ZED Mini

录制时优先保存 ZED SDK 原生 SVO/SVO2。它保留左右目同步、SDK 图像时间戳，之后可以直接用本目录的
`--input svo` 回放；MP4 只适合快速查看，不足以重建双目实验。

连续录制（按 `Ctrl-C` 停止）并同时生成可查看的左右目并排 MP4：

```bash
./.venv/bin/python zed_mini_tracker.py \
  --serial 13376675 \
  --resolution VGA \
  --fps 100 \
  --headless \
  --record-svo recordings/zed_mini_$(date +%Y%m%d_%H%M%S).svo2 \
  --record-mp4 recordings/zed_mini_preview.mp4
```

也可以只录原生文件：

```bash
./.venv/bin/python zed_mini_tracker.py --record-svo recordings/zed_mini.svo2
```

默认压缩为 H.264；原生文件的扩展名按实际 SDK 支持使用 `.svo` 或 `.svo2`。结束时终端 JSON
中的 `recording` 字段会报告路径和实际写入的 MP4 帧数。MP4 是左/右彩色图的并排预览，不包含可用于
深度重建的完整 SDK 数据；后续算法回放请使用 SVO/SVO2。原生录制从相机打开即开始，因此会包含
默认的 20 个 warmup 帧；若要严格从第一帧开始，可加 `--warmup-frames 0`。

## 9. ZED SDK 本地网络流接收

只有发送端已经启用 ZED SDK Streaming 时才运行：

```bash
./.venv/bin/python zed_probe.py \
  --input stream \
  --stream-host 192.168.2.123 \
  --stream-port 30000 \
  --frames 300
```

网络地址和端口必须替换为实际发送端配置。ZED Mini 直连 Linux 时优先验证 USB live 模式，网络流不是第一步。

## 10. 离线回归

不连接相机也能运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest discover -s tests -v
```

当前测试覆盖公共数据契约、BGRA/BGR、rectified/raw 标定配对、米制深度、变换方向、双目三角化、测量相关性、capture session 去重，以及 mock ZED SDK 的 open/read/close 链路。

## 11. 首次接入的停止条件

出现以下情况时停止继续部署，保存错误和预检输出：

- 需要 `sudo apt update`、系统级升级或替换 NVIDIA 驱动；
- ZED SDK 与 CUDA/Python 版本不兼容；
- `LOW_USB_BANDWIDTH`、`CAMERA_EXCEEDS_BANDWIDTH` 或持续掉帧；
- active resolution 与返回标定尺寸不一致；
- 时间戳为零、倒序或重复；
- baseline、左右图方向或视差符号与静态近物测试不一致；
- 开启 depth 后高速图像链路明显降频。

这些问题应先定位版本、链路或语义原因，不能通过放宽几何检查绕过。
