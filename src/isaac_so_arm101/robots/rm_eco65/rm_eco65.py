from pathlib import Path

import isaaclab.sim as sim_utils  # 仿真工具模块（加载 USD/URDF 文件）
from isaaclab.actuators import ImplicitActuatorCfg  # 隐式执行器配置类（PD 控制器）
from isaaclab.assets.articulation import ArticulationCfg  # 关节链配置类（机器人资产）

# ============================================================================
# 计算资产目录的绝对路径
# ============================================================================
# Path(__file__)        = 获取当前文件 (rm_eco65.py) 的路径
# .resolve()            = 转换为绝对路径（去掉 ../ 等相对符号）
# .parent               = 向上一级目录
# .parent.parent...     = 连续向上 5 级，到达项目根目录
# / "assets" / "rm_eco65" = 拼接最终的资产目录路径
TEMPLATE_ASSETS_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "assets" / "rm_eco65"
)

# ============================================================================
# RM-ECO65 机械臂的主配置对象
# ============================================================================
# ArticulationCfg = 关节链配置（Articulation 指由多个关节连接的刚体系统）
# RM_ECO65_CFG = 配置对象的名称，在其他代码中通过这个名字引用
RM_ECO65_CFG = ArticulationCfg(
    # ------------------------------------------------------------------------
    # spawn: 机器人生成配置（告诉 Isaac Lab 怎么把机器人加载到仿真中）
    # ------------------------------------------------------------------------
    spawn=sim_utils.UsdFileCfg(  # UsdFileCfg = 从 USD 文件加载（比 URDF 更快）
        # usd_path: USD 文件的完整路径
        # USD = Universal Scene Description（NVIDIA 的 3D 场景格式）
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/urdf/ECO65-B/ECO65-B.usd",
        # ------------------------------------------------------------------------
        # rigid_props: 刚体物理属性（定义每个零件的物理行为）
        # ------------------------------------------------------------------------
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # False = 启用重力（正常物理）；True = 失重模式（像太空）
            max_depenetration_velocity=10.0,  # 当两个物体重叠时，以最高 10 m/s 的速度分开
            #                                     太低会"粘"在一起，太高会飞出去，10.0 是平衡值
        ),
        # ------------------------------------------------------------------------
        # articulation_props: 关节链属性（定义整个机械臂的物理行为）
        # ------------------------------------------------------------------------
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # False = 机械臂的零件不会互相碰撞（训练初期推荐）
            # True  = 会碰撞（更真实，但容易卡死）
            solver_position_iteration_count=8,  # 物理求解器的位置迭代次数（4=快/软，8=平衡，16=慢/硬）
            solver_velocity_iteration_count=0,  # 速度求解器迭代次数（0=节省计算，大部分任务够用）
            fix_root_link=True,  # True  = 固定底座（底座被"钉"在地上）
            # False = 底座可移动（比如装在 AGV 上的机械臂）
        ),
    ),
    # ------------------------------------------------------------------------
    # init_state: 初始状态配置（机器人生出来时的样子）
    # ------------------------------------------------------------------------
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # 机器人底座在世界坐标系的位置 (x, y, z)，单位：米
        #       x  y  z
        # 睿尔曼 6 轴机械臂的初始关节角度（单位：弧度，不是度！）
        # 0 弧度 = 0°，1.57 弧度 = 90°，3.14 弧度 = 180°
        joint_pos={
            "joint_1": 0.0,  # 第 1 个关节（底座旋转）初始角度 = 0 弧度
            "joint_2": 0.0,  # 第 2 个关节（大臂）初始角度 = 0 弧度
            "joint_3": 0.0,  # 第 3 个关节（小臂）初始角度 = 0 弧度
            "joint_4": 0.0,  # 第 4 个关节（手腕旋转）初始角度 = 0 弧度
            "joint_5": 0.0,  # 第 5 个关节（手腕弯曲）初始角度 = 0 弧度
            "joint_6": 0.0,  # 第 6 个关节（法兰/夹爪）初始角度 = 0 弧度
        },
    ),
    # ------------------------------------------------------------------------
    # actuators: 执行器配置（定义电机的控制方式）
    # ------------------------------------------------------------------------
    actuators={
        "arm": ImplicitActuatorCfg(  # "arm" 是这个执行器组的名字（可以自定义）
            # joint_names_expr: 用正则表达式匹配要控制的关节
            joint_names_expr=[
                "joint.*"
            ],  # "joint." + ".*" = 匹配所有以 "joint" 开头的名字
            #                                 匹配结果: joint_1, joint_2, joint_3, joint_4, joint_5, joint_6
            #                                 不匹配: gripper, link_6 等
            stiffness=400.0,  # PD 控制器的 Kp（位置增益，刚度）
            #                 电机力矩 = Kp × (目标角度 - 当前角度) - Kd × 当前角速度
            #                 Kp 越大，响应越快、越"硬"
            #                 400 是统一值（SO-ARM101 用 200~50 分级）
            damping=40.0,  # PD 控制器的 Kd（速度增益，阻尼）
            #              Kd 越大，运动越平滑，震荡越小
            #              Kp/Kd = 400/40 = 10/1 是一个经验比例
            armature=0.01,  # 电机转子的转动惯量（考虑电机本身的惯性，0.01 是典型值）
        ),
    },
)
