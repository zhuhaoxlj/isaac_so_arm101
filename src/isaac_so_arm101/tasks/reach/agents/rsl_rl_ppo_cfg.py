# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32          # 保持 32：1M 样本/更新，覆盖稀疏成功轨迹
    max_iterations = 600            # run15：只需跑到 iter 600，峰值窗口（iter 300~450）完全覆盖后停止
    save_interval = 10              # 50→10：密集保存，精确捕获 iter 310~440 的 1.25~1.67s 峰值区
    experiment_name = "reach"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,             # 保持 0.5：充足的初始探索
        actor_hidden_dims=[512, 256],   # 保持：大容量网络学习 6DOF 隐式正运动学
        critic_hidden_dims=[512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,                 # 保持 0.1
        entropy_coef=0.008,             # run17：0.01→0.008，σ_eq=0.008/(4×0.01)=0.20，σ_actual 预期 0.38~0.42
                                        # 目的：降低噪声虚高（σ 越低→ play 均值越精确），同时远离崩溃阈值 0.005
                                        # run13 实测：entropy=0.008 在 fixed LR 下 σ 稳定 0.43，未崩溃，adaptive LR 更安全
        num_learning_epochs=6,          # 8→6：run15 教训：8 epochs 加速了熵收缩（iter300 熵比 run14 低 1.16）
                                        # 6 epochs × 4 mini-batch = 24次/rollout，与 run14 一致
        num_mini_batches=4,             # 保持 4：每 mini-batch 262K 样本（1M/4），规模充足
        learning_rate=1e-3,             # 3e-4→1e-3：run15 教训：3e-4 使 iter300 LR 仅 5.85e-4
                                        # run14 同期 LR=1.975e-3（高 3.4×），提供了突破 99% 的关键动能
                                        # run15 因 LR 不足在 86% 停滞，无法复现 run14 的突破
        schedule="adaptive",            # 保持 adaptive：KL 自调机制有效，关键是初始 LR 要足够高
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,                # 0.005→0.01：run15 教训：0.005 使 LR 减半过于频繁
                                        # 0.01 给策略更大更新空间，与 run14 一致（实现 99.93% 的参数）
        max_grad_norm=1.0,
    )