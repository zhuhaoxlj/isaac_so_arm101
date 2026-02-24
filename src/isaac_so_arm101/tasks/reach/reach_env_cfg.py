# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaac_so_arm101.tasks.reach.mdp as custom_mdp
from isaac_so_arm101.robots.trs_so100.so_arm100 import SO_ARM100_CFG
from isaac_so_arm101.tasks.reach.mdp.terminations import ee_reached_goal

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """
    命令配置（目标位置生成规则）

    Command（命令）= 环境告诉策略"要到达的目标位置"

    命令的作用：
    - 定义任务目标（ Reach 任务 = 末端到达某个 3D 位置）
    - 动态生成目标（每个 episode 或周期性重新采样）
    - 提供奖励信号的基础（计算与目标的距离）

    命令类型：
    - UniformPoseCommandCfg：均匀随机采样（最常用）
    - DesiredPoseCommandCfg：固定目标（调试用）
    """

    # ========================================
    # 命令：末端执行器目标位姿
    # ========================================
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",  # 机器人名称
        body_name=MISSING,  # 末端执行器链接名（由子类指定）
        resampling_time_range=(3.0, 3.0),  # 目标重采样时间范围（秒）
        #                          (3.0, 3.0) = 固定每 3 秒重采样一次
        #                          含义：episode 开始时生成目标，
        #                                 之后每 3 秒换一个新目标
        #                          run14 优化：从 5.0 → 3.0
        #                          目的：强迫策略在 3 秒内完成到达
        debug_vis=True,  # 是否在可视化中显示目标位置
        #         ^^^^
        #         True = 显示一个红色球体代表目标位置
        #         False = 不显示（训练时通常关闭以提高性能）
        ranges=mdp.UniformPoseCommandCfg.Ranges(  # 目标位置的采样范围
            pos_x=(-0.25, 0.25),  # X 范围（左右）：-25cm ~ +25cm
            pos_y=(-0.35, -0.05),  # Y 范围（前后）：-35cm ~ -5cm
            #                               （机械臂前方区域）
            pos_z=(0.05, 0.40),  # Z 范围（高度）：5cm ~ 40cm
            #                               （桌面到空中）
            roll=(0.0, 0.0),  # Roll 角（绕 X 轴旋转）：固定 0
            pitch=(0.0, 0.0),  # Pitch 角（绕 Y 轴旋转）：固定 0
            yaw=(0.0, 0.0),  # Yaw 角（绕 Z 轴旋转）：固定 0
            #                               Reach 任务只关心位置，不关心姿态
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)
        # 末端执行器当前位置（基座坐标系，3D）
        # 与 pose_command 中目标位置坐标系一致，策略可直接感知误差向量
        # body_names 由子类（joint_pos_env_cfg.py）覆写为具体末端链接名
        ee_pos = ObsTerm(
            func=custom_mdp.ee_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """
    事件配置（场景重置规则）

    Event（事件）= Episode 结束时的重置操作

    工作时机：
    - Episode 成功结束时（到达目标）
    - Episode 失败结束时（超时）
    - 手动调用 env.reset() 时

    事件类型：
    - reset：重置到默认值 + 随机偏移
    - randomize：完全随机化
    - interval：周期性触发（非 episode 结束时）
    """

    # ========================================
    # 事件：重置机器人关节
    # ========================================
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,  # Isaac Lab 内置函数
        #                                  功能：将关节重置为 默认值 + 随机偏移
        #                                  位置偏移范围：position_range
        #                                  速度偏移范围：velocity_range
        mode="reset",  # 事件模式
        #    "reset" = 重置模式（episode 结束时触发）
        #    其他可选："interval"（周期性）、"apply"（手动触发）
        params={
            "position_range": (-0.5, 0.5),  # 关节角度随机偏移范围（单位：弧度）
            #                              ±0.5 弧度 ≈ ±29 度
            #                              例如：默认角度 0.0 → 重置后范围 [-0.5, 0.5]
            #                              目的：让每次 episode 从不同姿态开始
            #                              好处：增加训练多样性，提高泛化能力
            "velocity_range": (0.0, 0.0),  # 关节速度重置范围（单位：弧度/秒）
            #                              (0.0, 0.0) = 固定为 0（静止开始）
            #                              如果设为 (-1.0, 1.0)，重置时会有随机初速度
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
        },
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.02,  # -0.1→-0.02：减少姿态追踪占比，集中资源打位置精度
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
        },
    )

    # action penalty
    # run14 根因修复：直接使用终态权重，消除 iter=250 的课程阶跃
    # 根因数据：run10 在 iter250 action_rate:-0.005→-0.01 后 Value Loss 立刻从 0.013→0.022+
    # σ_eq = entropy_coef / (4×|action_rate|) = 0.01/(4×0.01) = 0.25（理论）
    action_rate = RewTerm(
        func=mdp.action_rate_l2, weight=-0.01
    )  # -0.005→-0.01（消除课程）
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,  # -0.0001→-0.001（消除课程）
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # success bonus: run17 修复：std=0.03→0.02，收紧奖励峰值区
    # 目标：迫使均值 μ 向 <2cm 收敛，而非停在 3cm 边界（play 时无噪声，μ 精度决定成功率）
    # 梯度分析：std=0.02 时 @3cm reward=1.43（vs std=0.03 时 3.57），@2cm reward=3.57（梯度集中在内圈）
    # 终止阈值仍为 3cm（TerminationsCfg），训练成功定义不变，仅优化梯度方向
    success_bonus = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.02,
            "command_name": "ee_pose",
        },
    )
    # 时间压力：直接使用终态 -0.05/步，消除 iter=3000 的课程阶跃
    # 激励机制验证（run10 峰值 iter~280，ep_len=55步=1.83s）：
    #   -0.05 × 55步 = -2.75（快速成功）vs -0.05 × 220步 = -11.0（超时）
    #   差值 8.25 >> success_bonus ≈ 3.6（@3cm 末步）→ 充足的速度压力
    # 目标 <2s = 60步：-0.05×60 = -3.0；<3s = 90步：-0.05×90 = -4.5，差值 1.5/episode
    step_alive = RewTerm(
        func=custom_mdp.step_alive_penalty, weight=0.05
    )  # 0.02→0.05（消除课程）


@configclass
class TerminationsCfg:
    """
    终止条件配置（Episode 结束规则）

    Termination（终止）= Episode 提前结束的条件

    终止类型：
    - success：成功终止（到达目标，给予奖励）
    - time_out：超时终止（超过最大时间）

    终止的作用：
    1. 定义任务完成的条件
    2. 给予明确的奖励信号
    3. 避免策略在无解状态下浪费时间
    """

    # ========================================
    # 终止条件 1：超时终止
    # ========================================
    time_out = DoneTerm(
        func=mdp.time_out,  # Isaac Lab 内置函数
        #                    功能：检查是否超过 episode_length_s
        #                    返回：超过最大时间的环境索引
        time_out=True,  # 参数：超过时间后立即终止
        #           True = 触发终止，False = 只标记不终止
    )

    # ========================================
    # 终止条件 2：成功终止（末端执行器到达目标）
    # ========================================
    # 成功判定标准：末端执行器与目标位置的距离 < 阈值（默认 3cm）

    # run18 实验结论（2cm vs 3cm 阈值对比）：
    # - 2cm 阈值：play 成功率从 88% 暴跌至 38%
    # - 原因 1：2cm 阈值本身是"几何 Curriculum"—— Critic 在边界处发散
    # - 原因 2：play 判定收紧后，原本 2~3cm 范围内的成功全部算失败（50% 受影响）
    # - 结论：3cm 是结构性上限，保持 3cm 才能保证 88%+ 的 play 成功率

    success = DoneTerm(
        func=ee_reached_goal,  # 调用自定义函数（terminations.py:30）
        #                                  功能：计算末端执行器与目标的距离
        #                                  返回：距离 < 阈值 的布尔向量
        params={
            "command_name": "ee_pose",  # 命令名称：目标位置命令
            #                          对应 CommandsCfg.ee_pose
            "threshold": 0.03,  # 成功阈值：3 厘米
            #          0.03 米 = 3 厘米
            #          这是机械臂精度和任务难度的平衡点
            "asset_cfg": SceneEntityCfg(
                "robot",  # 机器人名称
                body_names=MISSING,  # 末端执行器链接名
                #               MISSING = 占位符，由子类指定具体值
                #               例如：SO-ARM101 = "gripper_link"
                #                      RM-ECO65 = "link_6"
            ),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    run14 根因修复：彻底移除所有课程变化。
    数据分析（5 run 对比）证明：所有 run 的 Critic 发散都在课程触发时启动。
      - run10 iter250 (action_rate 课程) → Value Loss 立刻从 0.013 → 0.022+
      - run13 iter3000 (step_alive 课程) → Value Loss 飙升 9x，Surrogate Loss 变正
    奖励权重已在 RewardsCfg 中直接设为终态值，无需课程渐变。
    保留空类以兼容 ReachEnvCfg 的 curriculum 字段签名。
    """

    pass


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(
        num_envs=32768, env_spacing=2.5
    )  # PhysX 64K材质上限：65536触发限制，32768安全边界（实测0.367MB/env，≈12.8GB VRAM，39%）
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """
        后初始化函数（配置类创建完成后自动调用）

        作用：设置一些动态计算的默认值
        """
        # ========================================
        # 通用设置
        # ========================================
        self.decimation = 2  # 抽稀系数：每 N 帧渲染一次
        #              物理引擎以 60 Hz 运行，但 RL 不需要那么快
        #              decimation=2 表示每 2 帧渲染 1 次 → 30 FPS
        #              好处：降低计算量，同时保持足够的控制频率

        self.sim.render_interval = self.decimation
        #     渲染间隔 = decimation（同步）

        # ========================================
        # Episode 长度设置
        # ========================================
        self.episode_length_s = 8.0  # Episode 最大长度（秒）
        #                          8.0 秒 = 每个回合最多 8 秒
        #                          超过 8 秒会触发 time_out 终止
        #                          run14 优化：从 12.0 → 8.0
        #                          目的：缩短最大 episode，迫使策略更积极

        # ========================================
        # 可视化相机位置
        # ========================================
        self.viewer.eye = (2.5, 2.5, 1.5)  # 相机位置（x, y, z）
        #              单位：米，相对于世界坐标原点

        # ========================================
        # 物理引擎设置
        # ========================================
        self.sim.dt = 1.0 / 60.0  # 物理时间步长（秒）
        #            1/60 ≈ 0.0167 秒 = 16.7 毫秒
        #            物理引擎每 16.7ms 计算一次
        #            对应 60 Hz 的物理刷新率
