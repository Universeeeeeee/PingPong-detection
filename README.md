# PingPong Detection

面向乒乓球场景的相机感知代码：从同步图像中检测乒乓球、估计三维位置，并在可用时识别球桌位姿、输出球桌坐标系下的球位置。

## 展示视频

[在 Bilibili 观看项目展示视频](https://www.bilibili.com/video/BV1owbN67EAS/)

## 当前包含的两条相机路径

| 路径 | 相机 | 当前用途 | 入口 / 文档 |
| --- | --- | --- | --- |
| D455 | Intel RealSense D455 | RGB、左右 IR、深度融合；球桌位姿、球二维/三维检测与短时预测 | [D455 交接与运行说明](D455_HANDOFF_20260904.md) |
| ZED Mini | Stereolabs ZED Mini | ZED SDK 同步彩色双目球检测；可选 SDK depth 球桌识别；SVO/SVO2 和 MP4 录制 | [ZED Mini 常用命令](zed%20x/README.md) |

两条路径使用不同相机 SDK、标定和数据格式，结果不能混用。ZED Mini 用于 SDK、采集和算法链路验证，不替代 ZED X 的最终硬件验收。

## 能力概览

- 球检测：橙球的颜色、形状、运动与双目几何约束；无可靠证据时拒绝输出三维测量。
- 三维位置：D455 通过左右 IR / 对齐深度，ZED Mini 通过同步左右彩色图三角化。
- 球桌识别：蓝色台面候选、外边界、深度平面和已知桌面尺寸共同验证；球桌无效时不发布球桌系坐标。
- 录制与回放：D455 原生多流录制；ZED 原生 SVO/SVO2 录制及可直接查看的 MP4 预览。

## 快速开始

### D455：完整球桌 + 球检测

在已配置 RealSense 依赖的 Linux PC 上：

```bash
cd /path/to/PingPong-detection
./run_d455_tracker.sh --duration 30 --display \
  --ir-exposure-us 2000 --ir-gain 64
```

无图形 SSH 环境请去掉 `--display`。详细环境、输出字段、录制和实时回放命令见 [D455 交接与运行说明](D455_HANDOFF_20260904.md)。

### ZED Mini：完整球桌 + 球检测

进入 ZED 目录后，先枚举相机，使用输出中的实际序列号：

```bash
cd /path/to/PingPong-detection/"zed x"
./.venv/bin/python zed_probe.py --list-devices

ZED_SERIAL=13376675  # 替换为实际 serial_number
./.venv/bin/python zed_mini_tracker.py \
  --serial "$ZED_SERIAL" \
  --resolution HD720 --fps 60 \
  --table-pose --table-hz 8 \
  --print-measurements
```

默认会显示球检测与球桌位姿窗口；按 `q` 或 `Esc` 退出。首次识别需要让完整球桌外轮廓清晰可见。更完整的 PC 命令、录制、SSH 和故障处理见 [ZED Mini 常用命令](zed%20x/README.md)。

## 坐标与安全语义

- 相机系的三维位置单位是米。
- `position_table_m` 只在 `table.valid=true` 时可用；值为 `null` 时不得用历史球桌姿态补全。
- 相机移动后应重新初始化球桌。
- 球桌坐标系尚未标定到机器人 base 坐标系，不能直接当作机器人世界坐标。

## 状态与限制

项目包含单元测试、合成几何测试和开发录像回放，但不同功能的实机精度与实时性能仍需分别验证。尤其是 D455 的三维绝对精度、独立场景泛化，以及 ZED Mini 的现场颜色阈值、深度性能和球桌位姿，均不应仅凭演示视频或离线结果视为最终验收。

## 目录说明

```text
.
├── d455_table_tennis_tracker.py   # D455 完整在线检测程序
├── table_pose_geometry.py         # D455 规则球桌位姿
├── record_d455_motion.py          # D455 原生多流录制
├── D455_HANDOFF_20260904.md       # D455 使用与限制说明
└── zed x/
    ├── README.md                   # ZED Mini PC 常用命令
    ├── zed_mini_tracker.py         # ZED Mini 球 / 球桌在线程序
    ├── zed_table_pose.py           # ZED Mini 规则球桌位姿
    └── zed_recording.py            # ZED MP4 预览录制
```
