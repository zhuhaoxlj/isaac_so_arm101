# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def ee_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """末端执行器在机器人基座坐标系中的3D位置 (num_envs, 3)。

    与 generated_commands 中的目标位置（也在基座坐标系）保持一致，
    让策略可直接计算 target_pos_b - ee_pos_b 误差向量，无需隐式学习正运动学。

    Args:
        env: RL 环境。
        asset_cfg: 机器人关节配置，body_names 需设为末端执行器链接名称。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    # 末端执行器在世界坐标系中的位置 (num_envs, 3)
    ee_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :3]
    # 转换到机器人基座坐标系
    ee_pos_b_frame, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        ee_pos_w,
    )
    return ee_pos_b_frame
