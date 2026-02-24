# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
自定义终止条件函数模块

本模块定义了 Reach 任务的终止条件，当满足特定条件时提前结束 episode。

Isaac Lab 的终止机制：
- 每个 episode 可以因为"成功"或"超时"而结束
- 成功结束会给予额外奖励，鼓励策略快速完成任务
- 超时结束避免策略在无解状态下浪费计算资源

使用方式：
    在 reach_env_cfg.py 中通过 DoneTerm 引用本模块的函数：
    success = DoneTerm(func=ee_reached_goal, params={...})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # PyTorch，用于并行计算（32768个环境同时计算）
from isaaclab.assets import RigidObject  # 刚体对象类（如可移动的物体）
from isaaclab.managers import SceneEntityCfg  # 场景实体配置类
from isaaclab.utils.math import combine_frame_transforms  # 坐标变换工具函数

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # RL 环境基类


# ============================================================================
# 终止条件函数 1：末端执行器到达目标
# ============================================================================

def ee_reached_goal(
    env: ManagerBasedRLEnv,              # 环境对象（包含所有并行环境）
    command_name: str = "ee_pose",       # 命令名称：目标位置的命令叫什么
    threshold: float = 0.03,             # 成功阈值：距离小于多少算成功（单位：米）
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 机器人配置（包含哪个链接是末端）
) -> torch.Tensor:                       # 返回：布尔向量 [num_envs]，每个环境是否终止
    """
    检查末端执行器是否到达目标位置（成功终止条件）

    工作原理：
    1. 获取末端执行器的当前位置（世界坐标系）
    2. 获取目标位置（从命令管理器获取）
    3. 计算两者之间的欧几里得距离
    4. 如果距离 < 阈值，返回 True（终止 episode）

    Args:
        env: RL 环境对象
        command_name: 命令名称（默认 "ee_pose" = 末端位姿命令）
        threshold: 成功阈值（默认 0.03 米 = 3 厘米）
        asset_cfg: 机器人配置，body_names 指定哪个链接是末端执行器

    Returns:
        布尔向量，形状 [num_envs]，True 表示该环境成功
        例如：[True, False, True, True, False, ...]
        → 第 0, 2, 3 号环境的机器人到达了目标
    """
    from isaaclab.assets import Articulation  # 关节链对象类（机械臂）

    # ========================================
    # Step 1: 获取机器人对象
    # ========================================
    robot: Articulation = env.scene[asset_cfg.name]
    #         ^^^^^^^^^^^^
    #         从场景中根据名字获取机器人对象

    # ========================================
    # Step 2: 获取目标位置命令
    # ========================================
    command = env.command_manager.get_command(command_name)
    #         ^^^^^^^^^^^^^^^^^^^^^^^^
    #         从命令管理器获取目标位置的命令
    #         返回形状：[num_envs, 7]（x, y, z, qx, qy, qz, qw）

    # 提取目标位置（前3列：x, y, z），坐标系：机器人基座
    des_pos_b = command[:, :3]
    #           ^^^^^^^ ^^^^
    #           取所有环境的 取前3列（位置，不取姿态）
    #           命令          形状：[num_envs, 3]

    # ========================================
    # Step 3: 将目标位置转换到世界坐标系
    # ========================================
    # combine_frame_transforms 的作用：
    # 将一个点从一个坐标系转换到另一个坐标系
    # 公式：P_world = T_root_to_world × P_base × P_point
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,      # 机器人基座的世界坐标 [num_envs, 3]
        robot.data.root_quat_w,     # 机器人基座的世界姿态（四元数）[num_envs, 4]
        des_pos_b                    # 目标位置（基座坐标系）[num_envs, 3]
    )
    # ^^^^^^^^
    # 目标位置的世界坐标，形状：[num_envs, 3]

    # ========================================
    # Step 4: 获取末端执行器当前位置（世界坐标系）
    # ========================================
    curr_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    #           ^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^
    #           机器人所有刚体的世界坐标  末端执行器的 body ID
    #           形状：[num_envs, num_bodies, 3]
    #                                          ↑
    #                                          只取末端执行器这一个
    #           最终形状：[num_envs, 3]

    # ========================================
    # Step 5: 计算距离
    # ========================================
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    #         ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^ ^^^^^
    #         欧几里得距离     当前位置 - 目标位置  沿最后一维（x,y,z）计算
    #         公式：√[(x1-x2)² + (y1-y2)² + (z1-z2)²]
    #         形状：[num_envs]

    # ========================================
    # Step 6: 返回布尔值（距离是否小于阈值）
    # ========================================
    return distance < threshold
    #      ^^^^^^^^^^^^^^^^^^^^^^
    #      返回形状：[num_envs] 的布尔向量
    #      例如：[True, False, True, ...]
    #      → 第 0 号环境：距离 0.02m < 0.03m → True（成功！）
    #      → 第 1 号环境：距离 0.05m > 0.03m → False（继续）


# ============================================================================
# 终止条件函数 2：物体到达目标（用于 Lift 任务，本项目 Reach 未使用）
# ============================================================================

def object_reached_goal(
    env: ManagerBasedRLEnv,                       # 环境对象
    command_name: str = "object_pose",            # 命令名称：物体目标位姿
    threshold: float = 0.02,                      # 成功阈值（2厘米）
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # 机器人配置
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),  # 物体配置
) -> torch.Tensor:                                 # 返回：布尔向量 [num_envs]
    """
    检查物体是否到达目标位置（用于 Lift 提升任务的终止条件）

    注意：本项目是 Reach 任务（机械臂末端到达目标），不涉及物体提升
          这个函数是为 Lift 任务准备的，当前未使用

    Args:
        env: RL 环境对象
        command_name: 命令名称（物体目标位姿）
        threshold: 成功阈值（默认 0.02 米 = 2 厘米）
        robot_cfg: 机器人配置
        object_cfg: 物体配置（要提升的物体）

    Returns:
        布尔向量，形状 [num_envs]，True 表示物体到达目标
    """
    # ========================================
    # Step 1: 获取机器人和物体对象
    # ========================================
    robot: RigidObject = env.scene[robot_cfg.name]
    #     ^^^^^^^^^^^
    #     RigidObject = 刚体（6自由度，可平移+旋转）
    #     用于可移动的物体（不是关节链）

    object: RigidObject = env.scene[object_cfg.name]
    #      ^^^^^^^^^^^
    #      要提升的物体（比如立方体、杯子）

    # ========================================
    # Step 2: 获取目标位置命令
    # ========================================
    command = env.command_manager.get_command(command_name)
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #         获取物体的目标位姿命令

    # 提取目标位置（前3列：x, y, z），坐标系：机器人基座
    des_pos_b = command[:, :3]
    #           ^^^^^^^ ^^^^
    #           形状：[num_envs, 3]

    # ========================================
    # Step 3: 将目标位置转换到世界坐标系
    # ========================================
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3],   # 机器人基座位置 [num_envs, 3]
        robot.data.root_state_w[:, 3:7],  # 机器人基座姿态 [num_envs, 4]
        des_pos_b                         # 目标位置（基座坐标系）
    )
    # ^^^^^^^^
    # 目标位置的世界坐标，形状：[num_envs, 3]

    # ========================================
    # Step 4: 获取物体当前位置（世界坐标系）
    # ========================================
    # object.data.root_pos_w[:, :3] = 物体的世界坐标
    # 形状：[num_envs, 3]

    # ========================================
    # Step 5: 计算物体与目标位置的距离
    # ========================================
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #         目标位置 - 物体当前位置 的欧几里得距离
    #         形状：[num_envs]

    # ========================================
    # Step 6: 返回布尔值（物体是否到达目标）
    # ========================================
    return distance < threshold
    #      ^^^^^^^^^^^^^^^^^^^^^^
    #      返回形状：[num_envs] 的布尔向量
    #      → True 表示物体被提升到目标位置

