"""将 ECO65-B URDF 转换为 USD 格式，供 Isaac Lab 训练使用。

用法：
    cd /output/isaac_so_arm101
    uv run python scripts/convert_urdf_to_usd.py --headless
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="URDF to USD 转换脚本")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制 headless 模式
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------- 以下代码在 Isaac Sim 启动后执行 -------
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "rm_eco65"
URDF_PATH = ASSETS_DIR / "urdf" / "ECO65-B.urdf"
OUTPUT_DIR = ASSETS_DIR / "urdf" / "ECO65-B"

print(f"[CONVERT] URDF 路径: {URDF_PATH}")
print(f"[CONVERT] USD 输出目录: {OUTPUT_DIR}")

assert URDF_PATH.exists(), f"URDF 文件不存在: {URDF_PATH}"

cfg = UrdfConverterCfg(
    asset_path=str(URDF_PATH),
    usd_dir=str(OUTPUT_DIR),
    usd_file_name="ECO65-B.usd",
    fix_base=True,
    merge_fixed_joints=True,
    self_collision=False,
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=400.0,
            damping=40.0,
        ),
    ),
)

converter = UrdfConverter(cfg)
print(f"[CONVERT] ✅ 转换完成！USD 文件: {converter.usd_path}")

simulation_app.close()
