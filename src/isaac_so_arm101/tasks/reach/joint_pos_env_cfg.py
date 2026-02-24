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
机器人特化配置模块

本模块定义了不同机器人的具体配置，通过继承 ReachEnvCfg 基类，
只修改机器人相关的字段，其他配置（奖励、终止等）保持不变。

设计模式：模板方法模式
- 基类（ReachEnvCfg）定义通用的任务结构
- 子类（SoArm100ReachEnvCfg 等）只替换机器人特定参数

为什么要这样设计？
- 避免重复代码
- 切换机器人只需创建新子类
- 便于对比不同机器人的性能
"""

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
    """
    SO-ARM100 机器人的 Reach 任务配置

    注意：当前配置实际使用的是 RM-ECO65 机器人（第 37 行）
          名称 "SoArm100" 是历史遗留，实际功能是 RM-ECO65 配置

    如果要使用真正的 SO-ARM100，需要：
    1. 取消第 36 行注释
    2. 注释第 37 行
    3. 修改 body_names 为 ["ee_link"]
    """
    def __post_init__(self):
        # ========================================
        # 先执行父类的初始化
        # ========================================
        # post init of parent
        super().__post_init__()

        # ========================================
        # 替换机器人资产
        # ========================================
        # switch robot to franka
        # self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                              被注释了，实际不使用 SO-ARM100

        self.scene.robot = RM_ECO65_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #                  ^^^^^^^^^^^
        #                  实际使用 RM-ECO65 配置
        #                  .replace() 方法：替换 prim_path（USD 路径模板）
        #                  {ENV_REGEX_NS} 会被替换为 "envs/env_0", "envs/env_1", ...
        #                  最终路径：/World/envs/env_0/Robot, ...

        # ========================================
        # 覆盖奖励配置：指定末端执行器链接名
        # ========================================
        # override rewards - ECO65 末端执行器为 link_6
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_6"]
        #                                                      ^^^^^^^^^^^^^^
        #                                                      RM-ECO65 的末端是 link_6
        #                                                      （不是 SO-ARM101 的 gripper_link）

        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.success_bonus.params["asset_cfg"].body_names = ["link_6"]

        # ========================================
        # 覆盖终止配置：指定成功终止检测的链接
        # ========================================
        # override terminations - 成功终止检测 link_6 是否到达目标
        self.terminations.success.params["asset_cfg"].body_names = ["link_6"]
        #                                            ^^^^^^^^^^^^^^
        #                                            ee_reached_goal 函数需要知道
        #                                            哪个链接代表末端执行器

        # ========================================
        # 覆盖观测配置：指定 ee_pos 观测的链接
        # ========================================
        # override observations - ee_pos 观察末端 link_6 当前位置
        self.observations.policy.ee_pos.params["asset_cfg"].body_names = ["link_6"]
        #                                           ^^^^^^^^^^^^^^
        #                                           ee_pos_b 函数需要知道
        #                                           观测哪个链接的位置

        # ========================================
        # 覆盖动作配置：指定关节名
        # ========================================
        # override actions - ECO65 6轴关节名为 joint1~joint6
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],          # ← 正则表达式：匹配所有 "joint" 开头的关节
            #               ^^^^^^^^^^
            #               匹配：joint_1, joint_2, ..., joint_6
            #               不匹配：gripper, link_6 等

            scale=0.5,                        # ← 动作缩放因子
            #     ^^^^
            #     PPO 输出的动作乘以 0.5
            #     例如：PPO 输出 1.0 → 实际关节角度偏移 0.5 弧度
            #     目的：限制单步动作幅度，提高稳定性

            use_default_offset=True,          # ← 是否使用默认偏移
            #                     ^^^^
            #                     True = 动作是相对于当前关节角度的偏移
            #                     False = 动作是绝对目标角度
        )

        # ========================================
        # 覆盖命令配置：指定目标位置是针对哪个链接
        # ========================================
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["link_6"]
        #                         ^^^^^^^^^^
        #                         目标位置是针对 link_6 的
        #                         即：让 link_6 到达目标位置
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm100ReachEnvCfg_PLAY(SoArm100ReachEnvCfg):
    """
    SO-ARM100 (实际 RM-ECO65) 的 Play 配置

    Play 模式 = 训练后的模型推理/演示模式

    与训练模式的不同：
    - 环境数量少（50 vs 32768）
    - 禁用随机化（观测不加噪声）
    - 适合可视化演示和性能评估
    """
    def __post_init__(self):
        # ========================================
        # 先执行父类（SoArm100ReachEnvCfg）的初始化
        # ========================================
        # post init of parent
        super().__post_init__()

        # ========================================
        # Play 模式特化设置
        # ========================================
        # make a smaller scene for play
        self.scene.num_envs = 50            # ← 环境数量：50（训练是 32768）
        #                       ^^^
        #                       Play 不需要那么多环境，
        #                       50 个足够评估性能，且计算量小

        self.scene.env_spacing = 2.5         # ← 环境间距：2.5 米
        #                        ^^^^
        #                        每个机器人之间间隔 2.5 米

        # ========================================
        # 禁用随机化（确定性推理）
        # ========================================
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        #                                       ^^^^^^
        #                                       False = 不添加噪声
        #                                       训练时是 True（添加噪声提高鲁棒性）
        #                                       Play 时是 False（干净观测，展示真实性能）


@configclass
class SoArm101ReachEnvCfg(ReachEnvCfg):
    """
    SO-ARM101 机器人的 Reach 任务配置

    与 SoArm100ReachEnvCfg 的区别：
    - 使用真正的 SO-ARM101 机器人（不是 RM-ECO65）
    - 末端执行器是 gripper_link（不是 link_6）
    - 关节名不同（shoulder_*, elbow_flex, wrist_*）
    - 姿态追踪权重设为 0（SO-ARM101 不关心姿态）
    """
    def __post_init__(self):
        # ========================================
        # 先执行父类的初始化
        # ========================================
        # post init of parent
        super().__post_init__()

        # ========================================
        # 替换机器人资产
        # ========================================
        # switch robot to franka
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #                  ^^^^^^^^^^^^^
        #                  使用 SO-ARM101 配置
        #                  {ENV_REGEX_NS} = 环境命名空间占位符

        # ========================================
        # 覆盖奖励配置：指定末端执行器链接名
        # ========================================
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        #                                                      ^^^^^^^^^^^^^^
        #                                                      SO-ARM101 的末端是 gripper_link

        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.success_bonus.params["asset_cfg"].body_names = ["gripper_link"]

        # ========================================
        # 覆盖终止配置：指定成功终止检测的链接
        # ========================================
        # override terminations
        self.terminations.success.params["asset_cfg"].body_names = ["gripper_link"]

        # ========================================
        # 覆盖观测配置：指定 ee_pos 观测的链接
        # ========================================
        # override observations - ee_pos 观察末端 gripper_link 当前位置
        self.observations.policy.ee_pos.params["asset_cfg"].body_names = ["gripper_link"]
        #                                           ^^^^^^^^^^^^^^
        #                                           ee_pos_b 函数观测 gripper_link 位置

        # ========================================
        # 特殊设置：关闭姿态追踪
        # ========================================
        self.rewards.end_effector_orientation_tracking.weight = 0.0
        #                                              ^^^^
        #                                              SO-ARM101 不关心姿态，
        #                                              只关注位置（Reach 任务）

        # ========================================
        # 覆盖动作配置：指定关节名
        # ========================================
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],                # ← 正则表达式：匹配所有关节
            #               ^^^
            #               SO-ARM101 的关节名：
            #               - shoulder_pan, shoulder_lift, elbow_flex
            #               - wrist_flex, wrist_roll, gripper
            #               ".*" 匹配所有名字（简单有效）

            scale=0.5,                         # ← 动作缩放因子
            use_default_offset=True,           # ← 使用相对偏移模式
        )

        # ========================================
        # 覆盖命令配置：指定目标位置是针对哪个链接
        # ========================================
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["gripper_link"]
        #                         ^^^^^^^^^^^^^^
        #                         目标位置是针对 gripper_link 的


@configclass
class SoArm101ReachEnvCfg_PLAY(SoArm101ReachEnvCfg):
    """
    SO-ARM101 的 Play 配置

    与 SoArm100ReachEnvCfg_PLAY 相同：
    - 环境数量少（50 vs 32768）
    - 禁用随机化（观测不加噪声）
    - 适合可视化演示和性能评估
    """
    def __post_init__(self):
        # ========================================
        # 先执行父类（SoArm101ReachEnvCfg）的初始化
        # ========================================
        # post init of parent
        super().__post_init__()

        # ========================================
        # Play 模式特化设置
        # ========================================
        # make a smaller scene for play
        self.scene.num_envs = 50            # ← 环境数量：50（训练是 32768）
        self.scene.env_spacing = 2.5         # ← 环境间距：2.5 米

        # ========================================
        # 禁用随机化（确定性推理）
        # ========================================
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        #                                       ^^^^^^
        #                                       False = 不添加噪声
        #                                       展示训练后模型的真实性能
