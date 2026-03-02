from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "assets" / "rm_eco65"

# 这是你 RM_ECO65 的专属配置
RM_ECO65_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/urdf/ECO65-B/ECO65-B.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 训练初期建议关闭自碰撞，防止乱动卡死
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,  # 固定基座，等效于 URDF 的 fix_base=True，防止重力下坠
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # 机械臂出生位置
        # 睿尔曼6轴机械臂的初始关节角度
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],  # 使用正则表达式匹配所有叫 joint 开头的关节
            stiffness=400.0,  # PD控制的 Kp (刚度)
            damping=40.0,     # PD控制的 Kd (阻尼)
            armature=0.01,
        ),
    },
)
