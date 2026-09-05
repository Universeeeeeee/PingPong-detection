# ZED X 乒乓球视觉迁移技术方案

版本日期：2026-09-05  
文档状态：硬件到货前设计稿  
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
  ├─ 回放与指标统计
  └─ ZMQ / ROS 2 适配输出
```

优点：

- 最大化复用当前 PC 的 Python 环境和调试工具。
- 大模型或额外实验可以使用 RTX GPU。
- Jetson 只负责可靠采集、编码和可选深度计算。
- 可在 PC 上将网络流作为普通 ZED SDK 输入。

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
| IR left/right | 同一 grab 的 `VIEW.LEFT/RIGHT` BGRA 图显式转灰度 | 数据语义改变 |
| `rs.align(color)` | ZED 注册到左图的 depth/XYZ | 删除或替换 |
| RealSense intrinsics/extrinsics | ZED calibration parameters | 替换 |
| `rs2_project_point_to_pixel` | 通用针孔投影或 ZED 标定 API | 替换 |
| `rs2_deproject_pixel_to_point` | ZED `DEPTH/XYZ` 或通用反投影 | 替换 |
| D455 自定义 `capture.db3` | ZED SVO2 + sidecar metadata | 替换 |
| D455 重放适配器 | SVO2 reader / 统一 FrameBundle reader | 替换 |
| RGB `ImageBallDetector` | 左目 RGB `ImageBallDetector` | 大部分保留并重标 |
| IR NCC/三角化 | 同步左右彩色图转灰度后 NCC/三角化 | 思路保留，参数重标 |
| `BallTrackGate` | 同一状态机 | 保留 |
| `TimestampedBallFilter` | 同一时间戳滤波器 | 保留 |
| ZMQ payload | 保持字段语义并扩展 metadata | 保留 |

## 7. 建议的软件边界

不要在主循环中到处直接调用 `pyzed.sl`。应把相机相关逻辑收敛到单一后端，使检测器只消费普通 NumPy 数据和显式标定结构。

### 7.1 统一帧对象

建议定义逻辑上的 `CameraFrameBundle`：

```python
CameraFrameBundle:
    source: "zed_x"
    serial_number: str
    sequence_id: int
    sdk_image_timestamp_ns: int
    processing_host_arrival_monotonic_ns: int
    timestamp_origin: str
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    depth_m: Optional[np.ndarray]
    xyz_m: Optional[np.ndarray]
    depth_confidence: Optional[np.ndarray]
    imu: Optional[dict]
    calibration: CameraCalibration
```

约束：

- `left_bgr` 与 `right_bgr` 必须来自同一次同步 grab。
- 标准 `VIEW.LEFT` / `VIEW.RIGHT` 返回 8-bit BGRA 四通道图；后端必须显式执行 BGRA→BGR，再填充 `left_bgr` / `right_bgr`，不能把 `get_data()` 直接标成 BGR。
- 检测与滤波使用 `sdk_image_timestamp_ns` 表示帧的 SDK 时间参考，不能使用算法处理完成时间替代它。
- `TIME_REFERENCE.IMAGE` 未被官方定义为曝光开始、曝光中点或曝光结束时间，字段名、日志和指标中禁止称其为“曝光时间”。
- `processing_host_arrival_monotonic_ns` 只用于当前处理主机上的排队和处理耗时；不同主机、Epoch 与 monotonic 时钟不得直接相减。
- `timestamp_origin` 必须记录 live/SVO/stream 以及时间戳来自采集端还是接收端；实体机上确认网络流是否保留发送端时间戳后再固定协议语义。
- 深度单位在后端统一转换为米。
- 无效深度使用显式 mask，不用零值冒充有效距离。
- 序列号、分辨率、帧率和标定必须写入运行记录。

### 7.2 标定对象

建议使用 SDK 无关的标定结构：

```python
CameraCalibration:
    image_geometry: "rectified" | "unrectified"
    calibration_variant: "calibration_parameters" | "calibration_parameters_raw"
    left_K: np.ndarray          # 3x3
    right_K: np.ndarray         # 3x3
    left_distortion: np.ndarray
    right_distortion: np.ndarray
    T_left_right: np.ndarray    # 4x4，定义方向必须固定
    image_width: int
    image_height: int
    coordinate_system: str
    units: "meter"
```

必须强制以下配对：

```text
VIEW.LEFT / VIEW.RIGHT
    ↔ camera_configuration.calibration_parameters

VIEW.LEFT_UNRECTIFIED / VIEW.RIGHT_UNRECTIFIED
    ↔ camera_configuration.calibration_parameters_raw
```

前者描述已校正图像，通常是零畸变的 PINHOLE 模型；后者描述真实镜头并保留原始畸变。禁止对 `VIEW.LEFT/RIGHT` 再套用 raw distortion，也禁止使用 rectified 参数解释 unrectified 图像。后端初始化时应断言图像视图、标定变体、分辨率和 calibration ID 一致。

禁止把 D455 的 `rs.intrinsics` 或 ZED SDK 对象继续传入球桌和球检测模块。这样可以逐步移除算法层对相机厂商 SDK 的依赖。

### 7.3 建议目录结构

在保留现有 D455 快照的前提下，新实现可采用：

```text
zed_x/
├── ZED_X_MIGRATION_TECHNICAL_PLAN.md
├── zed_capture.py
├── zed_record.py
├── zed_replay.py
├── camera_types.py
├── projection.py
├── zed_table_tennis_tracker.py
├── run_zed_x_tracker.sh
└── tests/
```

原则：

- 不重命名或覆盖当前 D455 文件。
- 初期直接复用上级目录中的纯算法模块，避免复制后形成两个漂移版本。
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

### 8.2 三维路径 A：ZED SDK depth/XYZ

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

### 8.3 三维路径 B：同步左右图 NCC

保留 `ball_epipolar.py` 的基本思想，但替换标定和数据输入：

1. 左目 RGB 检测给出球中心和搜索 ROI。
2. 使用 `VIEW.LEFT/RIGHT` 获取同一 grab 的左右校正 BGRA 图，显式转换为灰度；配套使用 `calibration_parameters`，不得使用 `calibration_parameters_raw`。
3. 在右图极线附近搜索 NCC 峰值。
4. 执行亚像素视差修正。
5. 使用 ZED 标定的焦距和基线三角化。
6. 检查重投影误差、左右外观、球体物理尺寸和匹配歧义。
7. 与 SDK depth/XYZ 比较，但两者不一定是完全独立证据，因为都源于同一对图像。

ZED 左右图硬件同步后，不再需要 D455 的 RGB-to-IR 时间邻近拼接。每个 stereo pair 只允许形成一次测量，仍保留重复帧与倒序时间戳拒绝。

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

### 8.5 深度模式实验

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

- 左右图使用同一 grab 对应的图像时间戳。
- IMU 如需融合，读取与当前图像同步的传感器数据。
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

### 12.1 原始格式

ZED X 的原始实验记录建议使用 SVO2，因为它能保存同步左右视频、时间戳和相机传感器数据，并允许回放时重新选择深度配置。

每次录制同时生成 sidecar `run.json`，记录：

- 相机序列号。
- ZED SDK、ZED Link、JetPack/L4T 版本。
- 分辨率、帧率、曝光、增益和白平衡。
- 深度模式和范围。
- 是否启用深度稳定化、positional tracking 和 IMU。
- SVO2 文件 SHA256。
- 主机时间、时区和代码版本。
- 场景标签和动作有效区间。

### 12.2 ROS bag 使用边界

- 只验证相机算法时，以 SVO2 为主。
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

缓存是可再生数据，不作为唯一原始记录。评分程序应能直接消费 SVO2 或确定性导出的缓存。

## 13. 硬件到货前可完成的工作

### 阶段 A：冻结当前 D455 基线

1. 保存本地 `d455_handoff/manifest.json`。
2. 记录 Linux 当前 `/home/pc/rsy/predict_node` 的差异文件。
3. 不覆盖 D455 生产目录。
4. 明确当前 D455 指标的录像、分母、时限和代码哈希。

### 阶段 B：提取 SDK 无关算法

1. 定义 `CameraFrameBundle` 和 `CameraCalibration`。
2. 将投影、反投影和刚体变换从 `pyrealsense2` 对象改为 NumPy 数据。
3. 让 `ball_image.py`、`ball_validation.py`、`ball_motion.py` 不导入相机 SDK。
4. 为通用投影函数增加数值单元测试。
5. 保持 D455 现有入口继续工作，不在同一次修改中删除 D455 后端。

### 阶段 C：建立 ZED 后端骨架

在没有硬件时可以实现但不能宣称通过实机验证：

- 参数解析。
- `pyzed.sl` 延迟导入和清晰错误信息。
- 统一标定结构转换。
- 左右图、depth、XYZ 和 confidence 的接口形状。
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

每个场景先录制短样本，验证可完整回放后再录长样本。

### 阶段 3：二维迁移验收

1. 固定左目图像格式。
2. 标注可见球帧和二维球心。
3. 重标 HSV、像素尺寸和拖影参数。
4. 对独立录像报告召回、误检、中心误差和处理延迟。

### 阶段 4：三维迁移验收

1. 比较 `NEURAL_LIGHT`、`NEURAL` 和必要时 `NEURAL_PLUS`。
2. 比较 SDK depth/XYZ 与左右 NCC。
3. 标注或测量独立三维真值。
4. 报告三维有效覆盖、错误更新、位置误差和延迟。
5. 验证遮挡、出画、落台和高速拖影期间不会长时间输出虚假预测。

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
ZED X 后端
  → 球/球桌感知
  → 标准化 Observation
  → 坐标与健康门控
  → Rally 状态机
  → 发球 policy / 接球 policy
```

标准化 Observation 至少包含：

```text
observation_id
sensor_timestamp_ns
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
- 原始 SVO2 基准数据集及 SHA256。
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

## 21. 下一步决策

当前不需要安装或部署任何 ZED 软件。下一步应按顺序完成：

1. 确定 ZED X 镜头版本、预期安装位置、球桌覆盖范围和工作距离。
2. 确定 Jetson/ZED Box 与 ZED Link 采集卡型号。
3. 确定 JetPack/L4T、ZED Link 驱动和 ZED SDK 的兼容组合。
4. 决定第一版采用“Jetson 本地处理”还是“Jetson 采集、RTX PC 处理”。
5. 在硬件到货前实现 SDK 无关接口和测试骨架。

推荐默认方向是：先根据具体 Jetson/ZED Box 型号核对 NVENC/NVDEC 与官方兼容矩阵。具备硬件编码能力时，可在开发阶段通过千兆有线网络将 ZED SDK 流送到 RTX PC，先复用现有感知算法；若采用无 NVENC 的 Orin Nano，则优先评估 Jetson 本地感知，或把软件编码/ROS 2 传输作为独立方案验收。最终架构由同一数据集上的端到端延迟和三维正确性决定，而不是由“Jetson”这一名称预先假定。
