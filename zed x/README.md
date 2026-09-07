# ZED Mini：常用命令

本目录包含 ZED Mini 的同步双目橙色乒乓球检测、规则球桌位姿识别、SVO/SVO2 原生录制和 MP4 预览录制。
球的三维位置始终在左相机坐标系；只有 `table.valid=true` 时，球测量中的 `position_table_m` 才有效。

以下命令假设已克隆本仓库，并已准备好能导入 `pyzed.sl` 和 `cv2` 的 ZED Python 虚拟环境。
Git 不会克隆 `.venv`；下面显式使用 PC 上已有的 ZED 环境，不要把它误写成仓库内的 `./.venv`。

## 0. 进入目录并更新

```bash
cd ~/zed_mini/zed_mini/PingPong-detection/"zed x"
git pull origin main

export ZED_PYTHON="$HOME/zed_mini/zed x/.venv/bin/python"
"$ZED_PYTHON" --version
```

变量名中间是下划线：使用 `"$ZED_PYTHON"`，不要写成 `"$ZED PYTHON"`。如果已有虚拟环境的位置不同，
用下面命令查找，然后把 `ZED_PYTHON` 改成实际路径：

```bash
find "$HOME/zed_mini" -type f -path '*/.venv/bin/python' -print
```

如果克隆目录不同，还需把第一行改为实际的 `PingPong-detection/zed x` 路径。

## 1. 设备与环境检查

```bash
bash linux_readonly_preflight.sh
"$ZED_PYTHON" zed_probe.py --list-devices
```

从第二条命令输出中记录实际 `serial_number`，再设置一次变量：

```bash
ZED_SERIAL=13376675  # 替换为实际 serial_number
```

## 2. 先验证高速双目采集

不检测球、不算 SDK depth，只检查左右图、时间戳、帧率和掉帧：

```bash
"$ZED_PYTHON" zed_probe.py \
  --serial "$ZED_SERIAL" \
  --resolution VGA \
  --fps 100 \
  --frames 300
```

预期 JSON 中 `camera_model` 为 `ZED_M`、`timestamps_strictly_increasing=true`；重点看
`sdk_dropped_frame_count_delta` 和 `timestamp_estimated_dropped_frames`。

## 3. 实时球检测（图形桌面）

会显示左右目并排图、球候选和三维 `XYZ`。按 `q` 或 `Esc` 结束：

```bash
"$ZED_PYTHON" zed_mini_tracker.py \
  --serial "$ZED_SERIAL" \
  --resolution VGA \
  --fps 100 \
  --warmup-frames 20 \
  --print-measurements
```

## 4. 实时球 + 球桌检测（推荐首次完整展示）

会额外打开 `ZED Mini table pose` 窗口。球桌初始化时，请让蓝色球桌的完整外轮廓清晰可见；首次建议使用
`HD720@60`，因为球桌识别需要 SDK depth：

```bash
"$ZED_PYTHON" zed_mini_tracker.py \
  --serial "$ZED_SERIAL" \
  --resolution HD720 \
  --fps 60 \
  --warmup-frames 20 \
  --table-pose \
  --table-hz 8 \
  --print-measurements
```

球检测窗口中，青色表示二维候选、绿色表示已通过双目验证的三维测量及其短历史轨迹；球桌窗口中绿色为有效、
橙色为等待确认。窗口快捷键：`q`/`Esc` 退出，`r` 清空显示用的三维轨迹，`t` 强制重新初始化球桌，`s` 保存
当前有效球桌位姿（需先传入 `--table-pose-file`）。例如：

```bash
"$ZED_PYTHON" zed_mini_tracker.py \
  --serial "$ZED_SERIAL" \
  --resolution HD720 --fps 60 \
  --table-pose --table-hz 8 \
  --table-pose-file recordings/table_pose.npz \
  --debug-dir recordings/last_debug
```

程序正常结束时，`--debug-dir` 会写出最后一帧的 `ball_preview.jpg` 和 `table_pose.jpg`。

终端测量 JSON 中：

- `position_camera_m`：左相机坐标系位置；
- `position_table_m`：仅球桌有效时才有值；
- `table_frame_id`：球桌重新确认后会改变，不能跨该编号拼接同一条球轨迹。

若桌面颜色未被识别，只调整 HSV 阈值，不要先放宽物理尺寸检查：

```bash
"$ZED_PYTHON" zed_mini_tracker.py \
  --serial "$ZED_SERIAL" \
  --resolution HD720 --fps 60 \
  --table-pose --table-hz 8 \
  --table-hsv-low 85,80,40 \
  --table-hsv-high 135,255,255
```

当桌面边界不完整、深度不够或尺寸不符合 `2.74 × 1.525 m` 时，`table.valid=false` 是预期的安全结果，
不会输出猜测的球桌系位置。

## 5. 无图形界面 / SSH

不打开窗口，使用 `Ctrl-C` 停止：

```bash
"$ZED_PYTHON" zed_mini_tracker.py \
  --serial "$ZED_SERIAL" \
  --resolution HD720 --fps 60 \
  --table-pose --table-hz 8 \
  --headless \
  --print-measurements
```

若图形桌面运行时提示 `cv2.imshow` 不可用，说明当前环境使用的是 headless OpenCV；先改用本节命令，
并在同一虚拟环境中准备支持 GUI 的 OpenCV 后再使用第 3/4 节的预览。

## 6. 同时录制 SVO2 和 MP4

SVO/SVO2 是后续算法回放的主文件，保留同步左右图和 SDK 时间戳；MP4 只是左右目并排预览。按 `q`、`Esc`
或 `Ctrl-C` 后等待程序正常结束，以完成封装：

```bash
SESSION="zed_$(date +%Y%m%d_%H%M%S)"
mkdir -p recordings

"$ZED_PYTHON" zed_mini_tracker.py \
  --serial "$ZED_SERIAL" \
  --resolution HD720 --fps 60 \
  --table-pose --table-hz 8 \
  --record-svo "recordings/${SESSION}.svo2" \
  --record-mp4 "recordings/${SESSION}_preview.mp4"
```

MP4 使用与实时窗口一致的标注预览（球候选、已确认三维轨迹、球桌状态），而不是可用于算法回放的原始数据；
后续算法回放仍应使用 SVO/SVO2。

若只需高速双目球录像而非球桌深度，可改用 `VGA@100` 并删除 `--table-pose --table-hz 8`。

## 7. 检查 SVO/SVO2

此命令验证录制文件能打开并检查其时间戳；它不运行球检测：

```bash
"$ZED_PYTHON" zed_probe.py \
  --input svo \
  --svo /absolute/path/to/recording.svo2 \
  --frames 300
```

## 8. 离线测试

```bash
PYTHONDONTWRITEBYTECODE=1 "$ZED_PYTHON" -m unittest discover -s tests -v
```

不连接相机也可运行。其中 ZED table pose 的合成图像测试需要 OpenCV；若当前环境没有 OpenCV，相关用例会显示
为 `skipped`，不是通过。

## 注意事项

- `--serial` 必须是本机实际枚举到的序列号，文档中的 `13376675` 只是之前一台 ZED Mini 的示例。
- ZED Mini 仅用于 USB/SDK/算法链路验证，不替代 ZED X 的最终硬件与运动模糊验收。
- 相机和球桌的相对位姿需固定；移动相机后应重新初始化球桌。
- 球桌坐标系原点在台面中心，尚未和机器人 base 坐标系标定，不能直接当作机器人世界坐标系。

更详细的接入边界、停止条件和实测基线见 [QUICKSTART_ZED_MINI_LINUX.md](QUICKSTART_ZED_MINI_LINUX.md)。
