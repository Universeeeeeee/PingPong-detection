# ZED X 乒乓球视觉迁移技术方案

版本日期：2026-09-05  
文档状态：最终软件架构已复核，硬件到货前设计稿  
源系统：`d455_handoff`（Intel RealSense D455）  
目标相机：Stereolabs ZED X（具体镜头、采集主机与软件版本待定）

## 1. 文档目的

本文定义如何将现有 D455 球桌与乒乓球检测系统迁移到 ZED X，并在不破坏现有 D455 部署的前提下完成开发、验证和最终接入。

本文回答以下问题：

1. 当前 D455 代码中哪些模块可以复用，哪些必须替换。
2. ZED X 所需的硬件与软件边界是什么。
3. 在实体相机尚未到货时可以提前完成哪些工作。
4. 到货后应如何录制、回放、标定、测试和验收。
5. 如何维持现有 ZMQ/坐标接口，使下游机器人与 policy 状态机尽量不依赖具体相机。

本文不包含：

- 未确认版本的 ZED SDK、CUDA、JetPack 或驱动安装操作。
- 对机器人关节、运动模式或 policy 的实机控制。
- “ZED X 一定达到 95% 三维检出率”之类未经独立数据验证的结论。
- 对当前 `/home/pc/rsy/predict_node` 的覆盖部署方案。

## 2. 结论摘要

迁移是可行的，但不是直接替换 Python import。

最终软件架构采用“两层统一”：第一层只把各相机 SDK 数据转换为少量、按算法语义定义的 typed frame；D455 与 ZED X 各自保留最适合自身成像方式的 frontend。两个 frontend 在输出 `BallMeasurement3D` 后才进入共同的门控、滤波、预测、坐标转换与 ZMQ 接口。禁止把 ZED 形状的大一统 `CameraFrameBundle` 强加给 D455，也不把 D455 的异步 RGB/IR 逻辑复制到 ZED X。

ZED X 的高速球主路径暂定为“同步全局快门彩色双目 + 高频二维检测 + rectified stereo matching/三角化”；ZED Neural Depth 是较低频的辅助证据、球桌/背景几何工具和基准对照，不预设它能随 SVGA 120 Hz 图像逐帧输出。最终速率必须在目标 Jetson/ZED Box 上实测。

可基本保留的部分：

- 橙球 RGB 二维检测思想与大部分实现。
- 球桌外轮廓、深度平面、已知尺寸和状态机。
- 三维候选的尺寸、重投影、身份和运动一致性验证。
- 时间戳状态滤波、速度估计和有限短时预测。
- `ball`、`table`、`table_frame_id` 等下游数据语义。
- 分阶段部署、录像回放、指标统计和安全门禁。

必须替换或重新验证的部分：

- `pyrealsense2` 采集与 RealSense 投影函数。
- D455 的 RGB、左右红外和对齐深度数据模型。
- D455 录制格式和回放工具。
- 所有 D455 内外参、曝光参数、HSV 阈值、尺寸阈值和延迟指标。
- 以左红外光学系命名的坐标元数据。
- 依赖红外亮目标的候选提取逻辑。

硬件上的决定性前提：ZED X 是 GMSL2 相机，不能通过 USB 直接连接当前 x86_64 PC，也不能直接连接 A3 的 Rockchip HDU/MDU。物理采集端需要兼容的 NVIDIA Jetson/ZED Box 与 ZED Link。当前 RTX Linux PC 可以作为开发/处理端，但只有在采集端具备 ZED SDK Local Streaming 所需的硬件编码能力、收发端 SDK 兼容且网络验收通过后，才能采用 ZED SDK 网络流；不能把“属于 Jetson”直接等同于“支持该网络流路径”。

## 3. 当前系统基线

### 3.1 当前权威入口

现有 D455 交接包的主要入口如下：

| 用途 | 文件 |
|---|---|
| 完整球桌与球检测 | `run_d455_tracker.sh` |
| 主处理程序 | `d455_table_tennis_tracker.py` |
| 仅 RGB 二维球检测 | `run_d455_2d.sh` / `d455_rgb_2d.py` |
| 原始实验录制 | `record_d455_motion.sh` / `record_d455_motion.py` |
| 球桌几何 | `table_pose_geometry.py` |
| RGB 球检测 | `ball_image.py` |
| 三维准入与身份验证 | `ball_validation.py` |
| 时间戳滤波与预测 | `ball_motion.py` |
| 快速 RGB-D 实验 | `fast_rgbd.py` |
| 双目几何和 NCC | `ball_epipolar.py` |
| 实时录像回放 | `replay_fast_rgbd.py` / `realtime_source.py` |

### 3.2 当前 D455 数据模型

现有完整程序基于以下不同速率的数据源：

```text
RGB 彩色图（当前典型 30 Hz）
左右红外图（当前目标 90 Hz）
对齐到彩色图的深度
IMU（录制链路可选）
```

由于 RGB 和 IR 不同速率且可能不同步，现有实现包含：

- RGB 身份与位置缓存。
- 原始 IR 缓存和时间邻近配对。
- RGB 视线对 IR 搜索区域的约束。
- 左右 IR 候选匹配和三角化。
- 重复时间戳与同一测量重复消费拒绝。
- 晚到测量的有限历史融合。

这些机制不能全部照搬到 ZED X。ZED X 的核心优势是同步左右彩色图；迁移后应删除不再成立的复杂度，而不是模拟 D455 的异步流结构。

### 3.3 当前接口语义

现有主程序通过 ZMQ（默认 `tcp://*:5555`）发布 Python 对象，核心字段包括：

- `ball_2d.uv`
- `ball_2d.timestamp_s`
- `ball_2d.new_observation`
- `ball.position_camera_m`
- `ball.velocity_camera_mps`
- `ball.camera_valid`
- `ball.measured_this_frame`
- `ball.measurement_source`
- `ball.measurement_age_s`
- `ball.world_valid`
- `ball.position_table_m`
- `ball.velocity_table_mps`
- `table.T_camera_table`
- `table.T_table_camera`
- `table.table_frame_id`

迁移目标是保留这些字段的业务语义，同时补充相机型号、坐标系、SDK 版本和时间源信息。

## 4. 已知条件、未知条件与决策门

### 4.1 已知条件

- 相机系列确定为 Stereolabs ZED X。
- 当前算法开发 PC 为 x86_64 Ubuntu 22.04，具有 NVIDIA RTX GPU。
- 当前系统已有 Python/OpenCV/NumPy/ZMQ 检测与回放框架。
- 当前 D455 生产路径不是 ROS node，主要外部接口是 ZMQ。
- 最终系统需要服务于低延迟乒乓球感知和机器人策略切换。

### 4.2 硬件到货前未知条件

| 未知项 | 为什么必须确认 |
|---|---|
| ZED X 镜头版本（例如宽视场或窄视场） | 决定视场、像素球径、工作距离和标定结果 |
| Jetson 或 ZED Box 型号 | 决定算力、内存、NVENC/NVDEC 能力与可支持帧率；例如 Orin Nano 没有 NVENC |
| ZED Link 采集卡型号 | 决定物理兼容性、端口数量和驱动包 |
| JetPack/L4T 版本 | ZED Link 驱动和 ZED SDK 必须与其匹配 |
| ZED SDK 版本 | 决定 Python API、深度模式、ROS topic 和时间戳能力 |
| ZED Link 驱动版本 | 必须与 JetPack/L4T 和硬件匹配 |
| 相机安装位置与姿态 | 决定球桌可见范围和 robot-base 外参 |
| Jetson 到 PC 的网络 | 决定能否稳定传送高帧率视频及额外延迟 |
| 目标工作距离与光照 | 决定镜头、曝光、增益和深度模式 |

### 4.3 决策门

在以下信息完成前，不执行驱动或 SDK 安装：

```text
Jetson/ZED Box 型号
JetPack/L4T 精确版本
ZED Link 卡型号
ZED X 序列号与镜头配置
厂家兼容矩阵中对应的驱动版本
与驱动匹配的 ZED SDK 版本
```

安装器可能检查或调整 CUDA、Python wrapper 和系统组件，因此不能在未知平台上先试装再说。

## 5. 推荐系统架构

本节的“Jetson 本地”与“RTX PC 网络接收”是部署拓扑候选，不改变第 7 节的软件边界。无论运行在哪台机器上，都使用 `ZedXSource → ZedXBallFrontend → BallMeasurement3D → shared core`。网络模式主要用于开发调试和算力对照；最终低延迟候选优先在采集端完成高频 stereo frontend，只发送低带宽测量。

### 5.1 开发阶段候选：具备硬件编码能力的 Jetson/ZED Box 采集，RTX PC 处理

```text
ZED X
  │ GMSL2
  ▼
Jetson / ZED Box
  ├─ ZED Link host driver
  ├─ ZED SDK
  ├─ 同步左右图采集
  ├─ SVO2 本地录制
  └─ ZED SDK Local Streaming（H.264/H.265）
             │ 千兆有线网络
             ▼
x86_64 RTX Linux PC
  ├─ ZED SDK Receiver
  ├─ ZED 相机适配层
  ├─ 球桌与乒乓球算法
  ├─ 可选在接收端重新计算 ZED depth
  ├─ 回放与指标统计
  └─ ZMQ / ROS 2 适配输出
```

优点：

- 最大化复用当前 PC 的 Python 环境和调试工具。
- 大模型或额外实验可以使用 RTX GPU。
- Jetson 可只负责可靠采集、SVO2 本地记录和硬件编码，减少与感知算法的资源竞争。
- 可在 PC 上将网络流作为普通 ZED SDK 输入。
- Local Streaming 发送左右 side-by-side 视频；需要 depth 时由接收端 SDK 根据解码后的 stereo pair 重新计算，而不是把 Jetson 上已算出的 depth map 当作流内容透传。

成立条件：

- 采集端型号具有 ZED SDK H.264/H.265 Local Streaming 所需的 NVENC 硬件编码能力。
- 接收端具有兼容的 NVIDIA GPU 和硬件解码能力。
- Jetson/ZED Box、ZED Link、JetPack/L4T、ZED SDK 版本组合在官方支持范围内。
- `enableStreaming()`、目标分辨率/帧率、持续码率和丢包恢复通过实体机测试。
- Jetson Orin Nano 没有 NVENC，不能默认复用这条硬件编码路径；如选用它，需要另行验证自定义软件编码/ROS 2 传输，或改为 Jetson 本地感知。

风险：

- 编码、网络、解码和排队会增加延迟。
- 视频压缩可能影响小球纹理与左右匹配。
- Wi-Fi 抖动不适合作为最终低延迟链路。
- Jetson 和 PC 必须安装兼容的 ZED SDK。
- Local Streaming 传输的是经 H.264/H.265 编码的左右 side-by-side 视频，不是无损 raw stereo；网络结果不能替代本地 lossless SVO2 的 NCC/亚像素基准。
- 高频 IMU、VIO 等能力还取决于 Streaming 版本与收发端 SDK；不能把“接收端可运行 SDK 模块”扩大解释为采集端所有原始传感器数据都无条件、全频率透传。

### 5.2 最终部署候选：Jetson 侧完成感知

```text
ZED X → Jetson/ZED Box → 球/球桌检测 → 低带宽坐标消息 → 机器人
```

优点：

- 避免传输完整视频导致的延迟和带宽压力。
- 避免跨主机网络接收和排队；但 ZED `TIME_REFERENCE.IMAGE` 仍只是 SDK 图像时间戳，不能称为物理曝光时刻。
- 只需向下游发送球位置、速度、球桌姿态和健康状态。

风险：

- Jetson 算力可能限制检测分辨率、帧率和深度模式。
- Python/OpenCV 实现需在 ARM64/L4T 环境重新验证。
- 如果同时启用高质量深度、录像和感知，可能出现 GPU/内存竞争。

最终选型不能只看平均 FPS，应比较两个架构的端到端 P50、P95、P99 延迟、丢帧率和正确三维覆盖率。

### 5.3 ROS 2 架构候选

如果最终机器人控制栈以 ROS 2 为主，可以让 Jetson 运行官方 ZED ROS 2 Wrapper，提供：

- 左右校正彩色图与灰度图。
- 注册到左目的深度图。
- 深度置信图和视差图。
- 彩色点云。
- IMU 与相机到 IMU 的变换。
- 相机健康与频率诊断。

但不建议在第一阶段同时重写相机后端和把整个程序改造成 ROS node。第一阶段应先用 ZED SDK Python API 保持现有程序形态，确认感知效果；ROS 2 作为独立适配层在结果稳定后接入。

## 6. D455 到 ZED X 的模块映射

| D455 实现 | ZED X 目标实现 | 迁移级别 |
|---|---|---|
| `pyrealsense2` | `pyzed.sl` | 替换 |
| `rs.pipeline` | `sl.Camera` | 替换 |
| `wait_for_frames()` | `Camera.grab()` | 替换 |
| RGB frame | `retrieve_image(VIEW.LEFT)` 得到 BGRA，再显式转 BGR | 替换接口，保留图像算法 |
| IR left/right | 同一图像事件的 `VIEW.LEFT_GRAY/RIGHT_GRAY` | 数据语义改变 |
| `rs.align(color)` | ZED 注册到左图的 depth/XYZ | 删除或替换 |
| RealSense intrinsics/extrinsics | ZED calibration parameters | 替换 |
| `rs2_project_point_to_pixel` | 通用针孔投影或 ZED 标定 API | 替换 |
| `rs2_deproject_pixel_to_point` | ZED `DEPTH/XYZ` 或通用反投影 | 替换 |
| D455 自定义 `capture.db3` | 指定压缩模式的 ZED SVO2 + sidecar metadata | 替换 |
| D455 重放适配器 | SVO2 reader / typed-frame reader | 替换 |
| RGB `ImageBallDetector` | 左目 RGB `ImageBallDetector` | 大部分保留并重标 |
| IR NCC/三角化 | 同步左右彩色图转灰度后 NCC/三角化 | 思路保留，参数重标 |
| 三维候选 | `BallMeasurement3D` | 新增正式的相机无关边界 |
| `BallTrackGate` | 同一门控逻辑 | 逻辑保留；将 `rgb_match_seconds` 等 D455 命名改为通用身份时效语义 |
| `TimestampedBallFilter` | 同一时间戳滤波器 | 数学逻辑保留；将按 `rgb`/`ir` 字符串判断相关性的代码改为显式 `correlation_group` |
| ZMQ payload | 保持字段语义并扩展 metadata | 保留 |

## 7. 建议的软件边界

不要在主循环中到处直接调用 `pyzed.sl`。相机 SDK 调用收敛在 source/backend 中，但“隐藏 SDK”不是唯一边界：D455 与 ZED X 的成像拓扑不同，球检测 frontend 也应分开。真正稳定的公共边界是已经完成身份、几何与来源描述的三维球测量。

### 7.1 最终分层与统一边界

```text
                         CAMERA SOURCE
─────────────────────────────────────────────────────────
 D455Source                                  ZedXSource
 RGB≈30 Hz + aligned depth                   LEFT color
 LEFT/RIGHT IR≈90 Hz                         LEFT/RIGHT gray
 异步 typed events                           可选低频 depth/confidence
        │                                           │
        ▼                                           ▼
 DetectionFrame / StereoFrame / DepthFrame / RigCalibration
        │                                           │
        ▼                                           ▼
 D455BallFrontend                           ZedXBallFrontend
 RGB 身份 + RGB→IR + IR 检测                LEFT 颜色身份
 异步关联 + IR stereo                       rectified stereo matching
 aligned depth 辅助                          SDK depth 辅助
        │                                           │
        └──────────────────┬────────────────────────┘
                           ▼
                   BallMeasurement3D
───────────────────────────┼───────────────────────────────
                从这里开始使用共同核心
                           ▼
                     BallTrackGate
                           ▼
                TimestampedBallFilter
                           ▼
                  prediction / table frame
                           ▼
                     ZMQ observation
```

设计含义：

- `ZedXSource` 负责如何取得和标准化 ZED 数据；`ZedXBallFrontend` 负责如何从这些数据得到可信三维球测量。
- `D455BallFrontend` 保留 IR 帧差、局部对比度、RGB→IR 投影和异步关联，不为了表面统一而改造成 ZED frontend。
- `ImageBallDetector` 可以由两个 frontend 共用，因为它的真实接口已经是普通 BGR 图和时间戳。
- `CameraFrameBundle` 如有需要只能作为 `ZedXSource` 的内部、单次采集快照，不作为 D455/ZED X 的公共算法接口。
- 当前 `BallTrackGate` 与 `TimestampedBallFilter` 的数学逻辑可复用，但代码仍有 `rgb_match_seconds`、`source.startswith('rgb'/'ir')` 等 D455 语义。实现公共边界时只做定向解耦，并用 D455 回放做前后回归，不能直接宣称现有代码已经完全相机无关。

### 7.2 按算法语义拆分 typed frame

建议的最小公共数据类型如下；它们是独立事件，不要求每个时刻组成一个所有字段都存在的大 bundle：

```python
DetectionFrame:
    image_bgr: np.ndarray
    sdk_image_timestamp_ns: int
    frame_number: int
    capture_session_id: str
    intrinsics: CameraIntrinsics
    camera_frame_id: str

StereoFrame:
    left_gray: np.ndarray
    right_gray: np.ndarray
    sdk_image_timestamp_ns: int
    frame_number: int
    capture_session_id: str
    calibration: StereoCalibration

DepthFrame:
    depth_m: np.ndarray
    valid_mask: np.ndarray
    confidence: Optional[np.ndarray]
    sdk_image_timestamp_ns: int
    frame_number: int
    capture_session_id: str
    intrinsics: CameraIntrinsics
    aligned_to_frame_id: str
```

ZED X 第一版映射：

```text
VIEW.LEFT (BGRA) → 显式 BGRA→BGR → DetectionFrame
VIEW.LEFT_GRAY + VIEW.RIGHT_GRAY → StereoFrame
MEASURE.DEPTH + MEASURE.CONFIDENCE → DepthFrame（可选、允许低于图像频率）
```

约束：

- ZED 左右图必须来自同一次 `read()` 或 `grab()` 对应的同步 stereo pair；不得把不同 frame number 拼成 `StereoFrame`。
- `frame_number` 是 source 每次成功采集后递增的进程内序号，必须与随机生成的 `capture_session_id` 联合使用；重启、重新打开设备或回放跳转后不得只靠帧号去重。
- `VIEW.LEFT_GRAY/RIGHT_GRAY` 可直接用于 stereo matching，无需先取 BGRA 再经过 BGR 转灰度。
- 标准 `VIEW.LEFT/RIGHT` 是 8-bit BGRA；只有完成显式转换后的数组才能命名为 `image_bgr`。
- 深度单位在 source 中统一为米，无效深度使用显式 mask，不用零值冒充有效距离。
- `TIME_REFERENCE.IMAGE` 是 SDK 图像时间参考而非物理曝光时刻；所有事件保留同一原始整数纳秒时间戳，进入现有滤波器时再确定性转换为秒。
- 另外记录处理主机 monotonic 到达时间；跨主机时钟未经 PTP/Chrony 与来源验证不得直接相减。

### 7.3 标定与 canonical camera frame

建议使用 SDK 无关、变换方向可读的结构：

```python
CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray
    distortion_model: str

StereoCalibration:
    left: CameraIntrinsics
    right: CameraIntrinsics
    T_right_from_left: np.ndarray
    image_geometry: "rectified" | "unrectified"
    calibration_variant: "calibration_parameters" | "calibration_parameters_raw"
    calibration_id: str

CameraRigCalibration:
    reference_frame_id: str
    detection_intrinsics: CameraIntrinsics
    stereo: StereoCalibration
    T_reference_from_detection: np.ndarray
```

采用 `T_dst_from_src` 命名，固定公式为 `p_dst = T_dst_from_src @ p_src`，避免 `T_left_right` 的方向歧义。Canonical frame 定义为 frontend 输出 `position_camera_m` 所在的参考光学系：D455 当前是左 IR optical frame；ZED X 是 rectified left optical frame。ZED X 中 detection 与 reference 都是左目，因此 `T_reference_from_detection` 为单位变换。

必须强制以下配对：

```text
VIEW.LEFT / VIEW.RIGHT / VIEW.LEFT_GRAY / VIEW.RIGHT_GRAY
    ↔ camera_configuration.calibration_parameters

VIEW.LEFT_UNRECTIFIED / VIEW.RIGHT_UNRECTIFIED
    ↔ camera_configuration.calibration_parameters_raw
```

前者描述已校正图像，通常是零畸变的 PINHOLE 模型；后者描述真实镜头并保留原始畸变。第一版只允许 rectified 路径。禁止对 rectified 图像再套 raw distortion，也禁止使用 rectified 参数解释 unrectified 图像。source 初始化时断言视图、标定变体、分辨率、frame ID 和 calibration ID 一致。

### 7.4 真正的公共接口：`BallMeasurement3D`

```python
BallMeasurement3D:
    timestamp_s: float
    position_camera_m: np.ndarray       # shape (3,)
    covariance_m2: np.ndarray           # shape (3,3)
    confidence: float
    source: str
    observation_id: tuple
    correlation_group: tuple
    identity_timestamp_s: Optional[float]
    camera_frame_id: str
    calibration_id: str
```

- `observation_id` 用于拒绝重复消费同一测量。
- `correlation_group` 显式表示来自同一 stereo pair 的 NCC、SDK depth 等相关证据，滤波器不得按来源字符串猜测独立性或重复计数。
- `identity_timestamp_s` 取代共享层的 RGB 专用命名，表示最近一次独立颜色/外观身份确认时间。
- `covariance_m2` 必须由各 frontend 根据自身几何和实验误差给出；不能把 D455 的焦距、基线和噪声地板用于 ZED X。
- `camera_frame_id` 与 `calibration_id` 是下游接纳测量的硬门禁。

`BallTrackGate`、`TimestampedBallFilter`、预测和坐标转换只接收该类型或它的字段，不接触 `pyrealsense2`、`pyzed.sl`、BGRA、IR 图像或 raw depth scale。

### 7.5 建议目录结构

在保留现有 D455 快照的前提下，新实现可采用：

```text
zed_x/
├── ZED_X_MIGRATION_TECHNICAL_PLAN.md
├── QUICKSTART_ZED_MINI_LINUX.md
├── linux_readonly_preflight.sh
├── zed_capture.py
├── zed_sdk_adapter.py
├── zed_source.py
├── zed_probe.py
├── zed_record.py
├── zed_replay.py
├── camera_types.py
├── zed_frontend.py
├── projection.py
├── zed_table_tennis_tracker.py
├── run_zed_x_tracker.sh
└── tests/
```

原则：

- 不重命名或覆盖当前 D455 文件。
- 初期直接复用上级目录中已经 SDK 无关的算法模块；需要修改共享代码时另开变更并用 D455 回放做回归，不在 ZED 后端提交里顺手改动。
- 当接口稳定后，再决定是否提取正式的 `camera_common` 包。
- 不为尚未确认的多相机需求预先设计复杂插件系统。

## 8. 乒乓球检测迁移设计

### 8.1 左目 RGB 二维检测

第一阶段继续使用 `ball_image.py::ImageBallDetector`：

1. 使用 `VIEW.LEFT` 读取 ZED 左目校正 BGRA 图，检查 `uint8` 四通道布局后显式转换为 BGR。
2. 使用现有橙球颜色、形状、运动和历史关联逻辑。
3. 基于 ZED X 分辨率重新测量球的像素直径、拖影宽度和最小面积。
4. 重新采集空场、静止球、慢速球、发球、接球、遮挡和干扰物录像。
5. HSV 和面积阈值不能直接复制 D455 默认值。

ZED X 是同步全局快门相机，理论上更适合高速运动目标，但实际运动模糊仍受曝光时间、增益、照明和球速影响。到货后需要以短曝光、高照度为起点做曝光扫描，而不是仅依靠全局快门名称判断图像足够清晰。

### 8.2 主路径：同步左右图 stereo matching/三角化

保留 `ball_epipolar.py` 的 NCC、亚像素定位和三角化思想，但不能直接复用其中的 RealSense 类型、RGB→IR 投影与 D455 标定约定：

1. 左目 BGR 检测给出球中心、轮廓和搜索 ROI。
2. 使用同一 frame number 的 `VIEW.LEFT_GRAY/RIGHT_GRAY`；配套使用 rectified `calibration_parameters`。
3. 利用 rectified 极线约束在右图同一行附近搜索 NCC 峰值。
4. 执行亚像素视差修正，并报告峰值强度、次峰比和左右一致性。
5. 使用运行时 ZED 标定的焦距和 baseline 三角化，不能使用宣传页上的近似参数。
6. 检查正视差、工作距离、重投影误差、物理球径、左右外观和运动创新。
7. 输出 `BallMeasurement3D`，并将 observation ID 与 stereo frame number 绑定。

ZED 左右图硬件同步后，不再需要 D455 的 RGB-to-IR 时间邻近拼接。SVGA 最高 120 FPS 是高速候选模式，但它只是相机采集上限；实际 `read()`、检测、匹配和发布频率必须在目标平台上验收。

### 8.3 辅助路径：ZED SDK depth/XYZ

处理流程：

```text
左目二维球候选
  → 球区域有效深度/XYZ采样
  → 前景连通性与离散度检查
  → confidence 门限
  → 物理球径检查
  → 时序创新门限
  → 三维候选
```

注意：

- 优先读取浮点 `DEPTH` 或 `XYZ`，不能使用仅供显示的 8-bit depth visualization。
- 球区域深度不能只取中心单像素，应使用前景 mask、邻域统计和异常值拒绝。
- ZED SDK 深度置信图应作为证据之一，但不能作为球身份的唯一证据。
- 深度模式的普通场景精度不代表小型高速球体精度。
- 深度稳定化等时域处理可能增加延迟或污染高速前景，需要单独 A/B 测试。
- SDK depth 与自定义 stereo 都来自同一左右图，属于相关证据；不得把两者当作两个独立观测送入滤波器。可以用于一致性检查，或从同一 `correlation_group` 中选择较可信的一项。
- 该路径优先服务于球桌/背景几何、低频球深度辅助和算法对照，不作为 120 Hz 球三维输出的默认承诺。

### 8.4 三维融合原则

建议输出明确的来源枚举：

```text
zed_depth
zed_xyz
zed_stereo_ncc
zed_depth_ncc_consensus
predicted_only
```

`measured_this_frame=true` 只在本周期接纳了新 `sdk_image_timestamp_ns` 对应的三维观测时成立。预测、历史保持或重复读取同一帧不能将其设为 true。

### 8.5 高频图像与低频 Neural Depth 的调度

不能把 `camera_fps=120` 理解成 `NEURAL_* depth=120 Hz`。例如官方 ZED SDK v5.0.1 RC、ZED X Driver v1.3.0 的 Orin AGX 单相机参考表中，NEURAL/NEURAL_LIGHT 为 30 FPS，NEURAL_PLUS 为 29 FPS 且 GPU 占用明显更高；这不是本项目目标硬件的保证值，实际数字还会随平台、分辨率、功耗模式、SDK/驱动和其他负载变化。

首选调度是官方 Split Process 模式：

```text
每个新图像：read() → LEFT color + LEFT/RIGHT gray → 2D + stereo
每 N 帧：   grab() → DEPTH/CONFIDENCE → 辅助验证或球桌更新
```

`read()` 只取新图像而不重新计算深度，`grab()` 在选定帧执行深度计算。当前官方示例为 C++；若目标 ZED SDK 的 Python binding 没有等价、经过验证的 `read()` 路径，应将采集 backend 写成小型 C++ 进程/扩展，或降低整体采集频率，不能用两个进程同时抢占同一相机来规避。

验收至少记录：image FPS、stereo 3D FPS、depth FPS、每类测量 age、GPU/CPU、丢帧和 P95/P99 延迟。若 split-process 在目标版本不可用或不稳定，退化方案是 `DEPTH_MODE.NONE` 的高频主进程，并把 Neural Depth 仅用于独立回放/基准实验。

### 8.6 深度模式实验

硬件到货后的初始候选：

| 模式 | 目的 | 主要风险 |
|---|---|---|
| `NEURAL_LIGHT` | 最低计算负载和延迟基线 | 可能遗漏小物体细节 |
| `NEURAL` | 精度与速度折中 | GPU 负载和延迟高于 LIGHT |
| `NEURAL_PLUS` | 最高质量对照 | 计算开销大，不一定满足实时性 |

每种模式必须在相同 SVO2、相同检测结果和相同统计脚本下比较。不能把不同录像、不同分辨率或不同可见性标注的结果直接并列。

## 9. 球桌检测迁移设计

球桌检测的大部分高级逻辑可以保留：

- 蓝色台面外轮廓。
- 真实边界与背景颜色变化验证。
- 深度平面拟合。
- 已知尺寸矩形恢复。
- 缺角恢复。
- 连续观测确认。
- 初始化后固定、独立验证和失效重检测。

需要替换：

- RealSense 投影/反投影 API。
- 深度有效值定义和置信度规则。
- D455 分辨率对应的面积与边缘像素阈值。
- D455 深度噪声对应的平面残差阈值。
- D455 左 IR 坐标系对应的命名。

建议首版使用 ZED 左目校正 RGB + 注册深度，不启用 ZED positional tracking 参与球桌静态姿态。先保留现有“固定相机、固定球桌”的可解释几何链路，避免将 VIO 漂移、重定位状态和球桌视觉状态混在一起。

如果后续相机安装在会运动的机器人头部，再增加：

```text
T_base_camera(t)
T_camera_table(t)
T_base_table(t) = T_base_camera(t) × T_camera_table(t)
```

其中 `T_base_camera(t)` 应来自机器人关节状态和标定，或经过验收的视觉惯性定位；不能仅靠球桌矩形消除所有 180° 方向歧义。

## 10. 坐标系与下游兼容

ZED SDK 默认图像坐标系与当前 D455 光学系约定相近：

```text
x：图像右方
y：图像下方
z：相机前方
原点：左目
```

建议初始化时显式指定：

```text
coordinate_units = METER
coordinate_system = IMAGE
```

不要依赖 SDK 默认值而不记录。运行记录中至少加入：

```json
{
  "camera_model": "ZED X",
  "camera_serial": "...",
  "camera_frame": "zed_left_camera_optical",
  "coordinate_system": "IMAGE",
  "coordinate_units": "meter",
  "zed_sdk_version": "...",
  "zed_link_driver_version": "...",
  "jetpack_version": "...",
  "l4t_version": "..."
}
```

下游不得把 D455 标定和 ZED X 标定混用。相机更换必须产生新的 calibration ID；球桌重检测仍递增 `table_frame_id`。

## 11. 时间戳、同步与延迟

### 11.1 时间戳要求

- 左右图使用同一 `read()`/`grab()` 图像事件对应的 frame number 和图像时间戳。
- IMU 如需融合，先验证 live、SVO2 与 Local Streaming 三种输入下的 stream/SVO 版本和传感器频率；每帧同步值可使用 `TIME_REFERENCE.IMAGE`，但不能默认网络接收端天然拥有采集端完整高频 IMU 序列。
- 所有滤波更新使用 `TIME_REFERENCE.IMAGE` 对应的 SDK 图像时间戳，不使用 Python `time.time()` 或算法完成时间替代帧时间。
- ZED 官方将其描述为 SDK 图像时间参考，并说明进入主机的数据包在主机接收时打时间戳；它不是经过定义的曝光开始/中点/结束时间。
- 额外使用 monotonic clock 记录当前主机的接收、处理开始、处理完成和发布时间。
- 网络接收模式必须区分发送端 SDK 图像时间、PC 接收时间和处理时间；未验证时钟来源与传递语义前，不跨主机直接做时间戳减法。
- 如需“物理曝光到控制输出”延迟，使用 LED/光电传感器、GPIO、示波器或高速摄像等外部测量，不从 `TIME_REFERENCE.IMAGE` 名称推断。

### 11.2 延迟字段

建议新增诊断：

```text
sdk_image_age_at_grab_ms
receiver_arrival_to_process_ms
host_queue_ms
detection_ms
depth_ms
stereo_match_ms
fusion_ms
publish_ms
end_to_end_ms
stream_decode_ms       # 网络模式
external_exposure_to_publish_ms  # 仅在有外部测量时填写
dropped_before_process
sequence_gap
```

### 11.3 实时原则

- 输入队列容量优先保持 1；过期图像应丢弃，不能排队形成越来越旧的轨迹。
- 处理结果必须带原始帧序号和图像时间戳。
- P95/P99 比平均值更重要。
- “算法运行 60 FPS”不等于“SDK 图像时间戳到下游消息低于 16.7 ms”，更不能自动证明物理曝光到下游的延迟。
- 网络流模式和 Jetson 本地模式必须使用同一延迟口径比较。

## 12. 录制与回放方案

### 12.1 基准记录格式与压缩分级

ZED X 的主要实验记录建议使用 SVO2，因为它能保存同步左右视频、时间戳和高频传感器数据，并允许回放时重新选择深度配置。但“SVO2”是容器/记录格式，不等于无损或 raw；实际图像质量由 `SVO_COMPRESSION_MODE` 决定。

分两类记录：

| 用途 | 压缩要求 | 说明 |
|---|---|---|
| golden benchmark | `LOSSLESS (PNG/ZSTD)`，或经逐像素验证的 H.264/H.265 lossless | 用于 HSV、球边缘、NCC、亚像素视差和 depth 对照；不得使用默认 lossy 文件冒充无损基准 |
| 长时工程记录 | H.265/H.264 lossy 或已验证可承受的 lossless 模式 | 用于稳定性、流程和故障复现；单独报告压缩伪影影响 |

官方文档给出的 ROS 2/SVO 默认配置可能是 H.265 lossy，因此录制程序必须显式设置模式并在日志中确认最终生效。开始正式数据采集前，分别验证目标分辨率/帧率下的写盘吞吐、掉帧、CPU/GPU、文件增长速度和异常关闭后的可回放性。

每次录制同时生成 sidecar `run.json`，记录：

- 相机序列号。
- ZED SDK、ZED Link、JetPack/L4T 版本。
- 分辨率、帧率、曝光、增益和白平衡。
- SVO2 compression mode、是否 lossless、实测平均码率和丢帧计数。
- 深度模式和范围。
- 是否启用深度稳定化、positional tracking 和 IMU。
- SVO2 文件 SHA256。
- 主机时间、时区和代码版本。
- 场景标签和动作有效区间。

### 12.2 ROS bag 使用边界

- 只验证相机算法时，以明确压缩模式的 SVO2 为主；几何/匹配 golden benchmark 必须优先使用 lossless。
- 需要同步机器人 joint state、policy 状态、控制命令或动捕时，使用 ROS bag/MCAP 记录完整系统。
- 不要只保存标注 MP4；压缩视频不足以恢复原始左右图、标定、深度和传感器时间信息。

### 12.3 统一离线数据接口

为复用当前测试工具，可以编写 SVO2 导出器，将每个 grab 导出为统一索引：

```text
recordings/<session>/
├── source.svo2
├── frames.jsonl
├── calibration.json
├── run.json
└── optional_cache/
    ├── left/
    ├── right/
    ├── depth/
    └── confidence/
```

缓存是可再生数据，不作为唯一权威记录。评分程序应能直接消费 SVO2 或确定性导出的缓存，并把压缩模式纳入数据集 ID。

## 13. 硬件到货前可完成的工作

### 阶段 A：冻结当前 D455 基线

1. 保存本地 `d455_handoff/manifest.json`。
2. 记录 Linux 当前 `/home/pc/rsy/predict_node` 的差异文件。
3. 不覆盖 D455 生产目录。
4. 明确当前 D455 指标的录像、分母、时限和代码哈希。

### 阶段 B：提取 SDK 无关算法

1. 先定义 `DetectionFrame`、`StereoFrame`、`DepthFrame`、`CameraRigCalibration` 和 `BallMeasurement3D`；不定义跨相机的大一统 `CameraFrameBundle`。
2. 将投影、反投影和刚体变换从 `pyrealsense2` 对象改为 NumPy 数据，并采用 `T_dst_from_src` 命名。
3. `ball_image.py` 保持 BGR 输入接口；不要把 ZED BGRA 转换塞入检测器。
4. 将 `ball_validation.depth_evidence(depth, scale, ...)` 的传感器 scale 处理前移到 source，公共验证只消费米制 `depth_m`。
5. 让 `BallTrackGate` 使用通用 identity freshness 语义，让 `TimestampedBallFilter` 使用显式 `correlation_group`，不按 `rgb`/`ir` 来源字符串猜测相关性。
6. 为通用投影、标定配对、变换方向、重复 measurement ID 和 correlation group 增加单元测试。
7. 保持 D455 现有入口继续工作；以同一 D455 回放的前后 JSONL 比较检测数、XYZ、时间戳、速度和 table pose，确认只是解耦而非改算法。

### 阶段 C：建立 ZED 后端骨架

在没有硬件时可以实现但不能宣称通过实机验证：

- 参数解析。
- `pyzed.sl` 延迟导入和清晰错误信息。
- typed frame 与 rectified 标定结构转换。
- BGRA→BGR、直接 gray views、depth meter、confidence 的接口形状。
- `ZedXBallFrontend` 到 `BallMeasurement3D` 的 mock 链路。
- SVO2 路径输入。
- 网络流地址输入。
- 运行元数据和 ZMQ payload。
- mock frame 单元测试。

### 阶段 D：准备验收工具

- 可见性/球心标注格式。
- 二维检测评分。
- 左右球心与视差评分。
- 三维真值导入接口。
- 延迟分位数统计。
- 误检、重复帧、倒序时间戳和预测占比统计。
- D455 与 ZED X 结果分别报告，不共用同一成绩标签。

### 13.1 当前离线实现状态（2026-09-06）

已在本目录落地第一批不依赖相机、OpenCV 或 ZED SDK 的实现：

- `camera_types.py`：公共 typed frame、标定结构和 `BallMeasurement3D`，包含单位、形状、frame ID、calibration ID 与 covariance 约束。
- `projection.py`：无畸变 pinhole 投影/反投影、`T_dst_from_src` 刚体变换、DLT 双目三角化和像素误差一阶传播。
- `zed_capture.py`：`pyzed.sl` 延迟导入、BGRA→BGR、同一帧号/SDK 图像时间戳的数据归一化、米制深度有效 mask，以及 observation/correlation key。
- `zed_frontend.py`：以已匹配左右球心为输入的 mock stereo frontend，输出正式 `BallMeasurement3D`。
- `tests/test_camera_foundation.py`：覆盖 rectified/raw 标定误配、BGRA 通道约束、深度有效 mask、投影闭环、变换方向、三角化以及同帧证据相关性。

第二批接入准备也已落地：

- `zed_sdk_adapter.py`：把 `Camera.open()` 后的 active rectified calibration 转换为公共标定，并生成可追踪的 calibration ID。
- `zed_source.py`：统一 live、SVO2 和 ZED SDK stream 输入，默认不计算 depth；每次打开生成独立 capture session ID。
- `zed_probe.py`：设备枚举与有限帧 JSON 诊断，记录配置、时间戳步长、实际 FPS 和 SDK dropped frame count。
- `linux_readonly_preflight.sh`：不使用 sudo/apt 的 Linux、NVIDIA、CUDA、ZED SDK、Python 和 USB 只读盘点。
- `QUICKSTART_ZED_MINI_LINUX.md`：ZED Mini 接线后的命令顺序、预期结果和停止条件。

当前测试命令：

```bash
cd "/path/to/d455_handoff/zed x"
python3 -m unittest discover -s tests -v
```

这只证明 SDK 无关的数据契约、合成几何闭环和 mock SDK 接入成立，不代表 ZED Mini/ZED X 的实际标定、帧率、时间戳、图像质量、USB/网络链路或乒乓球识别效果已经验收。连接 ZED Mini 后应严格按快速清单从只读诊断开始实机验证。

## 14. 实体到货后的实施顺序

### 阶段 0：只读平台盘点

先收集，不安装：

```text
Jetson 型号
JetPack 版本
L4T 版本
GPU/内存信息
ZED Link 设备信息
已有 ZED Link 驱动
已有 ZED SDK
网络接口与链路速率
磁盘空间
```

完成兼容矩阵后才形成安装变更单。

### 阶段 1：官方工具最小连通验证

1. 验证系统能识别 ZED X 序列号。
2. 验证左右图同步、无坏帧、无明显丢帧。
3. 读取工厂标定、温度和 IMU 信息。
4. 记录支持的分辨率、帧率和曝光范围。
5. 不启动机器人，不发布控制命令。

### 阶段 2：SVO2 录制

至少录制：

- 空桌静态场景。
- 球桌有遮挡和缺角。
- 静止橙球位于不同距离。
- 慢速抛球。
- 高速发球/接球。
- 球员、球拍、橙色衣物和背景干扰。
- 明暗不同的现场光照。

每个关键几何/匹配场景先录一份短 lossless golden sample，再按容量需要录长时工程样本。开始长录前验证 compression mode、实际写盘速率、帧序号连续性、传感器数据、完整回放和文件关闭行为。

### 阶段 3：二维迁移验收

1. 固定左目图像格式。
2. 标注可见球帧和二维球心。
3. 重标 HSV、像素尺寸和拖影参数。
4. 对独立录像报告召回、误检、中心误差和处理延迟。

### 阶段 4：三维迁移验收

1. 先验收 rectified 左右图 stereo matching/三角化主路径。
2. 比较 `NEURAL_LIGHT`、`NEURAL` 和必要时 `NEURAL_PLUS` 的辅助 depth/XYZ。
3. 比较“每帧 grab”与 `read()`/周期性 `grab()` split-process 的 image/depth 吞吐和延迟。
4. 在同一 lossless SVO2 上比较 SDK depth/XYZ 与左右 stereo，并把同源结果放入同一 `correlation_group`。
5. 标注或测量独立三维真值。
6. 报告 image FPS、stereo 3D FPS、depth FPS、三维有效覆盖、错误更新、位置误差和延迟。
7. 验证遮挡、出画、落台和高速拖影期间不会长时间输出虚假预测。

### 阶段 5：球桌与坐标验收

1. 标定相机到球桌。
2. 验证球桌尺寸、平面残差和方向。
3. 移动相机后确认旧标定失效。
4. 标定相机到机器人 base。
5. 验证 `table_frame_id` 变化时下游会断开旧轨迹。

### 阶段 6：架构性能对照

分别测量：

```text
Jetson 本地感知
Jetson → 有线网络流 → RTX PC 感知
```

选择依据：

- 正确三维测量覆盖率。
- 误检率。
- P50/P95/P99 端到端延迟。
- 丢帧率和最长连续丢帧。
- GPU、CPU、内存和温度。
- 录像开启后的性能退化。
- 连续运行稳定性。

### 阶段 7：下游与 policy 接入

感知稳定后，再完成：

1. ZMQ 或 ROS 2 消息适配。
2. camera/table/base 坐标转换。
3. 输入健康状态和超时机制。
4. policy 状态机只消费经过门控的新测量。
5. 无球、旧球、坐标失效或相机掉线时进入安全状态。

## 15. 验收指标定义

### 15.1 二维检测

至少报告：

- 可见球帧数。
- 及时正确二维检出数。
- 二维召回率。
- 无球帧误检数与误检率。
- 球心误差中位数、P95。
- 连续漏检最大长度。
- 遮挡恢复时间。

### 15.2 三维检测

至少报告：

- 可见且具有三维真值的帧数。
- 新三维实测数。
- 预测输出数。
- 错误三维更新数。
- 三维位置误差中位数、P95、最大值。
- 速度误差。
- `measurement_age_s` 分布。
- 三维输出端到端延迟 P50/P95/P99。

### 15.3 95% 目标的建议定义

如果继续使用“95%”目标，必须预先固定分母和时限。例如：

> 在独立测试集所有人工确认可见、未完全遮挡且位于有效工作空间的球帧中，至少 95% 的帧在对应 `sdk_image_timestamp_ns` 后 50 ms 内产生通过几何与身份检查的新三维实测；预测保持不计入成功。该指标必须标注为“SDK 图像时间戳到输出”，不能称为“曝光到输出”。无球帧误检率和三维位置误差另设上限。

如果产品指标明确要求“曝光后 50 ms”，必须增加外部曝光事件测量，并将 `external_exposure_to_publish_ms` 作为独立指标，不能用上述 SDK 指标替代。

不能把以下指标混称为 95%：

- 包含预测的输出比例。
- 只在成功候选内部计算的正确率。
- 训练录像或调参录像上的覆盖率。
- 不具备公制真值的“看起来有坐标”。
- 不同曝光、分辨率和可见性分母的混合结果。

### 15.4 系统稳定性

- 冷启动成功率。
- 相机断开与重连行为。
- 网络断开与恢复行为。
- 运行 30 分钟、2 小时和目标演示时长的稳定性。
- 输出目录增长和磁盘保护。
- SVO2 正常关闭后的可回放性。
- 进程异常退出后相机资源能否释放。

## 16. 与 policy 切换的接口要求

ZED X 迁移不应让发球/接球 policy 直接依赖 `pyzed.sl`。推荐边界：

```text
ZedXSource
  → ZedXBallFrontend / 球桌感知
  → BallMeasurement3D
  → 标准化 Observation
  → 坐标与健康门控
  → Rally 状态机
  → 发球 policy / 接球 policy
```

标准化 Observation 至少包含：

```text
observation_id
sdk_image_timestamp_ns
timestamp_origin
publish_timestamp_ns
camera_frame_id
calibration_id
table_frame_id
ball_measured
ball_predicted
measurement_source
measurement_age_s
position_camera_m
velocity_camera_mps
position_table_m
velocity_table_mps
covariance
table_valid
camera_healthy
```

policy 切换必须以状态、时间和有效性为条件，不能仅以“ZMQ 收到一条消息”为条件。推荐的最小门控：

- 只接受 calibration ID 和 table frame ID 正确的数据。
- 只使用时间戳单调递增的观测。
- 对进入接球 policy 的首个观测要求 `measured=true`，不能仅靠预测。
- `measurement_age_s` 超限时拒绝。
- 相机或网络异常时禁止继续消费旧轨迹。
- 切换瞬间显式重置或转换 policy 内部的历史状态。

具体 policy 状态机不在本文实现范围内，但相机迁移必须维持上述输入契约。

## 17. 安全与变更控制

### 17.1 硬件到货前

- 不安装 ZED SDK、CUDA、JetPack 或 ZED Link 驱动。
- 不执行 `apt update/upgrade`。
- 不覆盖 `/home/pc/rsy/predict_node`。
- 不修改机器人 MDU/HDU。
- 不执行任何机器人运动模式或关节控制命令。

### 17.2 到货后的变更审批点

以下动作必须单独确认：

- 刷写或升级 JetPack/L4T。
- 安装/替换 ZED Link 内核驱动。
- 安装会调整 CUDA 或 NVIDIA 驱动的 ZED SDK。
- 重启 Jetson、PC 或机器人控制器。
- 打开新的网络端口或修改防火墙。
- 将候选程序替换为生产入口。
- 连接机器人并开始消费实时坐标。
- 启动任何机器人运动或 policy。

### 17.3 部署隔离

建议初始目录：

```text
/home/pc/rsy/zed_x_candidate
```

或在 Jetson 上：

```text
/home/<user>/zed_x_pingpong_candidate
```

不得直接覆盖现有 D455 目录。候选版本通过独立配置和端口运行；同一时刻只能有一个进程占用实际相机或正式输出端口。

## 18. 风险清单

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 缺少 Jetson/ZED Box 或采集卡不匹配 | 相机无法连接 | 采购前核对官方兼容矩阵 |
| JetPack、ZED Link、SDK 不匹配 | 驱动加载失败或系统不稳定 | 锁定版本组合，先只读盘点 |
| 把任意 Jetson 都视为支持 SDK 网络流 | `enableStreaming()` 不可用或只能改走高负载软件编码 | 采购前核对 NVENC/NVDEC；Orin Nano 无 NVENC，单列验证或更换架构 |
| 网络流压缩损伤小球细节 | 左右匹配与二维检测下降 | 有线网络、高质量码率、与 Jetson 本地结果对照 |
| 网络流增加延迟和抖动 | policy 输入过旧 | 记录采集与到达时间，比较 P95/P99 |
| 把 SVO2 等同于 raw/lossless | lossy 伪影污染 HSV、球边缘和 NCC 基准 | golden 数据显式使用 lossless，记录最终 compression mode |
| 假设 SVGA 120 FPS 同时等于 Neural Depth 120 Hz | 实际 3D 频率和延迟严重不达标 | 高频 stereo 作为主路径；用 `read()`/周期性 `grab()` 解耦并实测三种频率 |
| 用大一统 FrameBundle 强行覆盖 D455 与 ZED | 大量 Optional、旧帧复用和异步语义泄漏到公共层 | typed events + camera-specific frontend，在 `BallMeasurement3D` 后统一 |
| ZED depth 对高速小球无有效值 | 三维覆盖不足 | 保留同步左右图 NCC 路径 |
| 自动曝光导致球拖影或左右亮度变化 | 检测/匹配失败 | 固定曝光实验、高照度、记录实际参数 |
| 直接沿用 D455 阈值 | 漏检或误检 | 重新标注和调参 |
| 坐标系方向混淆 | 机器人动作方向错误 | 显式 frame ID、静态标定物和轴向测试 |
| 将预测当实测 | 虚假高覆盖率 | 分开统计 `measured` 与 `predicted` |
| 同时改相机后端和 policy | 故障无法定位 | 先冻结感知接口，再接 policy |
| SVO2 只记录相机数据 | 无法复现完整机器人状态 | 系统级测试并行录制 ROS bag/MCAP |
| Jetson 资源竞争 | 降帧、过热、延迟升高 | 分模块 profiling、温度监控和长稳测试 |

## 19. 预期交付物

硬件到货前：

- 本技术方案。
- SDK 无关帧与标定接口设计。
- 通用投影/反投影测试。
- ZED 后端骨架与 mock 测试。
- SVO2/网络流参数配置模板。
- 新的指标统计与报告模板。

硬件到货后：

- 平台版本盘点报告。
- 经批准的安装变更单。
- 官方相机连通与健康报告。
- lossless SVO2 golden 基准数据集及 SHA256、compression mode 和掉帧报告。
- ZED X 二维检测报告。
- SDK depth 与 NCC 三维对照报告。
- 球桌和 robot-base 标定文件。
- Jetson 本地与 RTX PC 网络模式性能对照。
- 稳定的 ZMQ/ROS 2 感知接口。
- 最终部署清单、启动脚本、回滚步骤和安全门禁。

## 20. 官方资料基线

后续实现以 Stereolabs 官方文档和目标版本 release note 为准：

1. [ZED X 产品与 GMSL2 设置](https://docs.stereolabs.com/docs/products/cameras/zedx)
2. [GMSL2 相机与 Jetson 平台兼容性](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
3. [在 PC 上通过网络流开发 ZED X](https://docs.stereolabs.com/docs/products/cameras/zedx/development-on-pc)
4. [ZED SDK Local Video Streaming](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/local-network-streaming)
5. [ZED 输出视图与 BGRA 格式](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-controls)
6. [ZED raw/rectified 标定参数](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)
7. [ZED 深度 API](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/using-the-api)
8. [ZED 深度模式](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-modes)
9. [ZED 坐标系与 Frame Transform](https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/coordinate-frames)
10. [ZED Sensors API](https://docs.stereolabs.com/docs/development/zed-sdk/modules/sensors/using-the-api/)
11. [ZED 传感器时间同步](https://docs.stereolabs.com/docs/development/zed-sdk/modules/sensors/time-synchronization)
12. [ZED SVO/SVO2 录制](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording)
13. [ZED ROS 2 入门](https://docs.stereolabs.com/docs/integrations/ros-2)
14. [ZED ROS 2 Stereo Node 与 topics](https://docs.stereolabs.com/docs/integrations/ros-2/zed-stereo-node)
15. [ZED ROS 2 录制与回放](https://docs.stereolabs.com/docs/integrations/ros-2/record-and-replay-data)
16. [ROS 2 Hardware Encoding Bridge](https://docs.stereolabs.com/docs/integrations/ros-2/hardware-encoding-bridge)
17. [NVIDIA：Orin Nano 软件编码与无 NVENC 说明](https://docs.nvidia.com/jetson/archives/r35.6.0/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html)
18. [ZED SDK Linux 安装说明](https://docs.stereolabs.com/docs/development/zed-sdk/linux)
19. [ZED SDK Docker 说明](https://docs.stereolabs.com/docs/development/zed-sdk/use-with-docker)
20. [ZED SDK Split Process：`read()` 高频图像与周期性 `grab()` 深度](https://docs.stereolabs.com/docs/tutorials/split-process)
21. [ZED ROS 2 频率调优与 SVGA 120 FPS 约束](https://docs.stereolabs.com/docs/integrations/ros-2/node-frequency-tuning)
22. [ZED X 分辨率、帧率与运行时焦距说明](https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view)

## 21. 下一步决策

当前不需要安装或部署任何 ZED 软件。下一步应按顺序完成：

1. 确定 ZED X 镜头版本、预期安装位置、球桌覆盖范围和工作距离。
2. 确定 Jetson/ZED Box 与 ZED Link 采集卡型号。
3. 确定 JetPack/L4T、ZED Link 驱动和 ZED SDK 的兼容组合。
4. 实现 typed-frame、`BallMeasurement3D`、通用标定/投影和 mock stereo 测试；涉及现有 D455 文件的实际重构另开变更并先建立回归基线。
5. 到货后分别验收 Jetson 本地和 RTX PC 网络模式，再决定最终部署位置。

最终软件架构已经确定：`ZedXSource → ZedXBallFrontend → BallMeasurement3D → shared core`，高速 rectified stereo 是球三维主路径，Neural Depth 是低频辅助路径。尚未确定的是部署拓扑。先根据具体 Jetson/ZED Box 型号核对 NVENC/NVDEC 与官方兼容矩阵；具备硬件编码能力时，可在开发阶段通过千兆有线网络把 ZED SDK 流送到 RTX PC。若采用无 NVENC 的 Orin Nano，则优先评估 Jetson 本地感知，或把软件编码/ROS 2 传输作为独立方案验收。最终部署位置由同一 lossless 数据集上的端到端延迟、三维正确性、资源占用和稳定性决定。
