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

from isaac_so_arm101.robots.trs_so100.so_arm100 import SO_ARM100_CFG

# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
import isaac_so_arm101.tasks.reach.mdp as custom_mdp
from isaac_so_arm101.tasks.reach.mdp.terminations import ee_reached_goal
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
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
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(3.0, 3.0),   # 5.0→3.0：缩短重采样窗口，强迫策略在3s内完成到达
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.25, 0.25),
            pos_y=(-0.35, -0.05),
            pos_z=(0.05, 0.40),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
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
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),  # ±0.5 rad (~±29°) 随机初始关节角，覆盖更多起始状态
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.02,                       # -0.1→-0.02：减少姿态追踪占比，集中资源打位置精度
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    # run14 根因修复：直接使用终态权重，消除 iter=250 的课程阶跃
    # 根因数据：run10 在 iter250 action_rate:-0.005→-0.01 后 Value Loss 立刻从 0.013→0.022+
    # σ_eq = entropy_coef / (4×|action_rate|) = 0.01/(4×0.01) = 0.25（理论）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)    # -0.005→-0.01（消除课程）
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,                                                # -0.0001→-0.001（消除课程）
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # success bonus: run17 修复：std=0.03→0.02，收紧奖励峰值区
    # 目标：迫使均值 μ 向 <2cm 收敛，而非停在 3cm 边界（play 时无噪声，μ 精度决定成功率）
    # 梯度分析：std=0.02 时 @3cm reward=1.43（vs std=0.03 时 3.57），@2cm reward=3.57（梯度集中在内圈）
    # 终止阈值仍为 3cm（TerminationsCfg），训练成功定义不变，仅优化梯度方向
    success_bonus = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.02, "command_name": "ee_pose"},
    )
    # 时间压力：直接使用终态 -0.05/步，消除 iter=3000 的课程阶跃
    # 激励机制验证（run10 峰值 iter~280，ep_len=55步=1.83s）：
    #   -0.05 × 55步 = -2.75（快速成功）vs -0.05 × 220步 = -11.0（超时）
    #   差值 8.25 >> success_bonus ≈ 3.6（@3cm 末步）→ 充足的速度压力
    # 目标 <2s = 60步：-0.05×60 = -3.0；<3s = 90步：-0.05×90 = -4.5，差值 1.5/episode
    step_alive = RewTerm(func=custom_mdp.step_alive_penalty, weight=0.05)  # 0.02→0.05（消除课程）


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 成功终止：末端执行器进入目标 3cm 范围内
    # run18 实验（2cm 阈值）结论：2cm 阈值使 play 从 88% 暴跌至 38%
    # 原因：2cm 阈值本身是"几何 Curriculum"—— Critic 在 2cm 边界处同样发散
    #        且 play 判定收紧后，原本 2~3cm 的 μ 全部算失败（50% 案例受影响）
    # 维持 3cm：play 88%（2~3cm μ 仍成功），3cm 上限才是结构性天花板
    success = DoneTerm(
        func=ee_reached_goal,
        params={"command_name": "ee_pose", "threshold": 0.03,
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
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
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=32768, env_spacing=2.5)  # PhysX 64K材质上限：65536触发限制，32768安全边界（实测0.367MB/env，≈12.8GB VRAM，39%）
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
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0         # 12.0→8.0：缩短最大episode，迫使策略更积极
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
