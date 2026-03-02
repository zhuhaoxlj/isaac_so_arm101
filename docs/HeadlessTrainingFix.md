# Isaac Lab 无头训练环境修复记录

> 日期：2026-02-22
> 环境：RTX 5090 / Driver 580.95.05 / CUDA 13.0 / Ubuntu 22.04 容器
> 任务：`Isaac-SO-ARM100-Reach-v0`（使用 RM_ECO65 机械臂）

---

## 问题概述

执行 `uv run train --task Isaac-SO-ARM100-Reach-v0 --headless` 时训练无法启动，依次出现三类独立故障，需按顺序全部修复才能正常训练。

---

## 故障一：USD 资产文件缺失

### 症状

```
FileNotFoundError: USD file not found at path at:
'/output/isaac_so_arm101/assets/rm_eco65/urdf/ECO65-B/ECO65-B.usd'
```

训练在场景创建阶段直接崩溃退出。

### 根本原因

Isaac Lab 要求机器人以 **USD 格式**加载，但仓库中只有原始 URDF 文件，从未执行过转换。

```
assets/rm_eco65/urdf/
├── ECO65-B.urdf   ← 存在（原始 URDF）
├── ECO65-B.csv
└── ECO65-B/
    └── ECO65-B.usd  ← 不存在（代码期望此文件）
```

### 修复方法

编写并运行 URDF → USD 转换脚本：

```bash
uv run python scripts/convert_urdf_to_usd.py --headless
```

脚本核心配置（`scripts/convert_urdf_to_usd.py`）：

```python
cfg = UrdfConverterCfg(
    asset_path=str(URDF_PATH),
    usd_dir=str(OUTPUT_DIR),
    usd_file_name="ECO65-B.usd",
    fix_base=True,           # 固定底座（等效 URDF fix_base）
    merge_fixed_joints=True,
    self_collision=False,
    force_usd_conversion=True,
)
```

转换后生成：

```
assets/rm_eco65/urdf/ECO65-B/
├── ECO65-B.usd              # 主入口（Isaac Lab 加载此文件）
├── config.yaml
└── configuration/
    ├── ECO65-B_base.usd     # 网格几何体（2.6 MB）
    ├── ECO65-B_physics.usd  # 物理属性
    ├── ECO65-B_robot.usd    # 关节 / 连杆结构
    └── ECO65-B_sensor.usd   # 传感器定义
```

---

## 故障二：OpenGL / X11 系统库缺失

### 症状

训练启动后大量报错，但可以继续运行（非致命）：

```
Error: libGL.so.1: cannot open shared object file: No such file or directory
Error: libX11.so.6: cannot open shared object file: No such file or directory
Error: libXext.so.6: cannot open shared object file
```

### 根本原因

容器基础镜像未预装 OpenGL 和 X11 运行时库，Isaac Sim 的部分渲染插件（`omni.usd.libs`、`omni.gpu_foundation`）加载失败。

### 修复方法

```bash
apt-get install -y libx11-6 libxext6 libgl1 libxt6 libglu1-mesa
```

| 包名 | 提供的库 | 用途 |
|------|----------|------|
| `libx11-6` | `libX11.so.6` | X11 基础协议 |
| `libxext6` | `libXext.so.6` | X11 扩展 |
| `libgl1` | `libGL.so.1` | OpenGL 运行时 |
| `libxt6` | `libXt.so.6` | X11 工具包 |
| `libglu1-mesa` | `libGLU.so.1` | OpenGL 工具库 |

---

## 故障三：PhysX GPU 无法启用（核心问题）

### 症状

安装上述库后，Vulkan 错误依然存在，且 **GPU 利用率始终为 0%**：

```
VkResult: ERROR_INCOMPATIBLE_DRIVER
vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
PhysXFoundation: Unable to get IGpuFoundation, GpuDevices or Graphics!
```

训练进程存在（PID 可见），占用 3.4 GB 显存，但长时间无迭代输出，表现为"卡住"。

### 根本原因分析

```
NVIDIA Vulkan ICD
 └── nvidia_icd.json
      └── libGLX_nvidia.so.0   ← GLX 路径，依赖 X11 Display
           └── 无 DISPLAY 环境变量 → vkCreateInstance 返回 ERROR_INCOMPATIBLE_DRIVER
                └── PhysX GPU 初始化失败 → 回退 CPU 物理模拟
                     └── 4096 env × CPU physics → 单次 rollout 耗时 6+ 分钟 → 表现为"卡住"
```

**关键细节：**

- RTX 5090 驱动（580.95.05）的 Vulkan ICD 使用 `libGLX_nvidia.so.0`（GLX 路径）
- GLX 路径在无 X11 Display 的 headless 容器中无法创建 Vulkan 实例
- Isaac Sim 打包了自己的 Vulkan loader（`libvulkan.so.1.3.239`），同样依赖此 ICD
- PhysX GPU 因 Vulkan 失败而回退 CPU，4096 个环境的 CPU 物理模拟极其缓慢（非真正死锁）

**验证方式：** 用 32 个环境测试时训练正常（CPU 负载可承受），4096 个环境时表现为卡住，证明并非代码 bug 而是性能问题。

### 修复方法

安装 Vulkan loader 并启动 **Xvfb 虚拟帧缓冲**为 GLX ICD 提供 X11 Display：

**步骤 1：安装依赖**

```bash
apt-get install -y libvulkan1 xvfb
```

**步骤 2：启动虚拟显示器**

```bash
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
```

**步骤 3：运行训练时指定 DISPLAY**

```bash
DISPLAY=:99 uv run train --task Isaac-SO-ARM100-Reach-v0 --headless
```

### 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Vulkan 错误 | `ERROR_INCOMPATIBLE_DRIVER` | **无错误** |
| GPU 利用率 | 0% | **14%+（随 env 数线性增长）** |
| 显存占用 | 3.4 GB | **6.2 GB** |
| GPU 功耗 | 10 W | **100 W** |
| PhysX 模式 | CPU 回退 | **GPU 加速** |

---

## 完整修复命令清单

### 首次环境配置（仅需执行一次）

```bash
# 1. 安装系统库
apt-get update
apt-get install -y libx11-6 libxext6 libgl1 libxt6 libglu1-mesa libvulkan1 xvfb

# 2. 生成 USD 资产（如不存在）
cd /output/isaac_so_arm101
uv run python scripts/convert_urdf_to_usd.py --headless
```

### 每次容器重启后（需重新执行）

```bash
# 启动虚拟显示器
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
```

> ⚠️ 系统库（`apt install`）在容器重启后会丢失，需重新安装。
> 建议将以上命令写入启动脚本或 Dockerfile。

### 训练命令

```bash
DISPLAY=:99 uv run train --task Isaac-SO-ARM100-Reach-v0 --headless
```

---

## 诊断工具备忘

```bash
# 检查 GPU 状态
nvidia-smi

# 实时监控 GPU 利用率
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
  --format=csv,noheader,nounits -l 2

# 检查 Vulkan ICD 依赖是否满足
ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 | grep "not found"

# 检查训练进程状态
cat /proc/<PID>/stat | awk '{print "state:"$3, "stime:"$15}'

# 小规模快速验证（不需要 GPU physics 也能跑通）
DISPLAY=:99 uv run train --task Isaac-SO-ARM100-Reach-v0 --headless --num_envs 32
```
