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
自定义观测函数模块

本模块定义了 Reach 任务的自定义观测函数，扩展 Isaac Lab 的内置观测。

为什么需要自定义观测？
- Isaac Lab 内置函数可能不满足特定需求
- 需要特殊的数据处理或坐标转换
- 提供更丰富的信息给策略

使用方式：
    在 reach_env_cfg.py 中通过 ObsTerm 引用本模块的函数：
    ee_pos = ObsTerm(func=custom_mdp.ee_pos_b, params={...})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # PyTorch，用于并行计算
from isaaclab.assets import Articulation, RigidObject  # 资产类
from isaaclab.managers import SceneEntityCfg  # 场景实体配置
from isaaclab.utils.math import subtract_frame_transforms  # 坐标变换工具

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # RL 环境基类


# ============================================================================
# 自定义观测函数 1：物体在机器人基座坐标系中的位置（用于 Lift 任务）
# ============================================================================


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    物体在机器人基座坐标系中的位置（用于 Lift 任务）

    注意：本项目是 Reach 任务（机械臂末端到达目标），不涉及物体提升
          这个函数是为 Lift 任务准备的，当前未使用

    Args:
        env: RL 环境对象
        robot_cfg: 机器人配置
        object_cfg: 物体配置

    Returns:
        物体位置（基座坐标系），形状 [num_envs, 3]
    """
    # 获取机器人和物体对象
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # 获取物体世界坐标
    object_pos_w = object.data.root_pos_w[:, :3]
    #                ^^^^^^^^^^^^^^^^^^^^ ^^^^
    #                物体的世界坐标     取前3列（x,y,z）
    #                形状：[num_envs, 3]

    # 转换到机器人基座坐标系
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],  # 机器人基座位置 [num_envs, 3]
        robot.data.root_state_w[:, 3:7],  # 机器人基座姿态 [num_envs, 4]
        object_pos_w,  # 物体世界坐标 [num_envs, 3]
    )
    # ^^^^^^^^^^^^
    # 物体在基座坐标系中的位置
    # 形状：[num_envs, 3]

    return object_pos_b


# ============================================================================
# 自定义观测函数 2：末端执行器在基座坐标系中的位置（本项目核心）
# ============================================================================


def ee_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    末端执行器在机器人基座坐标系中的 3D 位置

    ========== 设计思想 ==========
    为什么要提供这个观测？

    问题：如果只提供目标位置和关节角度，策略需要隐式学习"正运动学"
          即：关节角度 → 末端位置的映射关系

    解决：直接提供末端执行器的当前位置
          策略可以显式计算误差向量 = 目标位置 - 当前位置
          大大简化学习难度

    坐标系选择：
    - 使用基座坐标系（而非世界坐标系）
    - 原因：目标位置命令也在基座坐标系
    - 好处：两者在同一坐标系，策略可直接相减计算误差

    ========== 参数说明 ==========
    Args:
        env: RL 环境对象
        asset_cfg: 机器人配置
            body_names 需设为末端执行器链接名称
            （SO-ARM101 = "gripper_link", RM-ECO65 = "link_6"）

    ========== 返回值 ==========
    Returns:
        末端执行器位置（基座坐标系），形状 [num_envs, 3]
        例如：[[0.2, -0.1, 0.3],   ← 环境 0 的末端位置
              [0.15, -0.2, 0.25],  ← 环境 1 的末端位置
              ...]
    """
    # ========================================
    # Step 1: 获取机器人对象
    # ========================================
    robot: Articulation = env.scene[asset_cfg.name]
    #         ^^^^^^^^^^^^
    #         从场景中根据名字获取机器人对象

    # ========================================
    # Step 2: 获取末端执行器的世界坐标
    # ========================================
    # 末端执行器在世界坐标系中的位置
    ee_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :3]
    #           ^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^ ^^^^
    #           机器人所有刚体的世界坐标   末端执行器的 body ID  取前3列
    #           形状：[num_envs, num_bodies, 3]           形状：[num_envs, 3]

    # ========================================
    # Step 3: 转换到机器人基座坐标系
    # ========================================
    # subtract_frame_transforms 的作用：
    # 将一个点从一个坐标系转换到另一个坐标系
    # 公式：P_base = T_world_to_base × P_world
    ee_pos_b_frame, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],  # 机器人基座位置 [num_envs, 3]
        robot.data.root_state_w[:, 3:7],  # 机器人基座姿态 [num_envs, 4]
        ee_pos_w,  # 末端世界坐标 [num_envs, 3]
    )
    # 末端在基座坐标系中的位置
    # 形状：[num_envs, 3]

    # ========================================
    # Step 4: 返回结果
    # ========================================
    return ee_pos_b_frame
