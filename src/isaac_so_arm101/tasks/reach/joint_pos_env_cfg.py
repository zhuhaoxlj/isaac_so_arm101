# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.robots.rm_eco65.rm_eco65 import RM_ECO65_CFG
from isaac_so_arm101.tasks.reach.reach_env_cfg import ReachEnvCfg

##
# Scene definition
##


@configclass
class SoArm100ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        # self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = RM_ECO65_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards - ECO65 末端执行器为 link_6
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.success_bonus.params["asset_cfg"].body_names = ["link_6"]
        # override terminations - 成功终止检测 link_6 是否到达目标
        self.terminations.success.params["asset_cfg"].body_names = ["link_6"]
        # override observations - ee_pos 观察末端 link_6 当前位置
        self.observations.policy.ee_pos.params["asset_cfg"].body_names = ["link_6"]

        # override actions - ECO65 6轴关节名为 joint1~joint6
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["link_6"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm100ReachEnvCfg_PLAY(SoArm100ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.success_bonus.params["asset_cfg"].body_names = ["gripper_link"]
        # override terminations
        self.terminations.success.params["asset_cfg"].body_names = ["gripper_link"]
        # override observations - ee_pos 观察末端 gripper_link 当前位置
        self.observations.policy.ee_pos.params["asset_cfg"].body_names = ["gripper_link"]

        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["gripper_link"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm101ReachEnvCfg_PLAY(SoArm101ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
