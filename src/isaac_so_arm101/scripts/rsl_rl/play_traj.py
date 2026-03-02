# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint and log end-effector + target trajectory for analysis.

Usage:
    uv run play_traj --task Isaac-SO-ARM100-Reach-Play-v0 \\
        --checkpoint logs/rsl_rl/reach/2026-02-22_00-53-16/model_150.pt \\
        --num_steps 600 --ee_body link_6 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

# ---------------------------------------------------------------------------
# CLI args - must be defined before AppLauncher
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play RL agent and log EE trajectory.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (default 1 for trajectory logging).")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
# trajectory-specific args
parser.add_argument("--num_steps", type=int, default=600, help="Total env steps to record.")
parser.add_argument("--ee_body", type=str, default="link_6", help="End-effector body name in the robot articulation.")
parser.add_argument("--output_dir", type=str, default="trajectory_logs", help="Directory to save CSV and plots.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All heavy imports AFTER simulator launch
# ---------------------------------------------------------------------------

import csv
import os
import time
from pathlib import Path

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# ---------------------------------------------------------------------------
# Helper: save CSV
# ---------------------------------------------------------------------------

def save_trajectory_csv(records: list[dict], output_path: str) -> None:
    """Save trajectory records to CSV."""
    if not records:
        print("[TRAJ] No records to save.")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[TRAJ] Saved {len(records)} rows → {output_path}")


# ---------------------------------------------------------------------------
# Helper: generate plots (requires matplotlib + pandas)
# ---------------------------------------------------------------------------

def plot_trajectory(csv_path: str) -> None:
    """Generate 4-panel trajectory analysis plot from CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend, safe for headless
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print(f"[TRAJ] Cannot generate plot (missing dependency): {e}")
        return

    df = pd.read_csv(csv_path)
    episodes = sorted(df["episode"].unique())
    max_episodes_to_plot = min(len(episodes), 6)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Trajectory Analysis\n{Path(csv_path).name}", fontsize=11)

    # ----- 1) 3D trajectory -----
    ax3d = fig.add_subplot(221, projection="3d")
    cmap = plt.get_cmap("tab10")
    for i, ep in enumerate(episodes[:max_episodes_to_plot]):
        ep_df = df[df["episode"] == ep]
        color = cmap(i % 10)
        ax3d.plot(ep_df["ee_x"], ep_df["ee_y"], ep_df["ee_z"],
                  alpha=0.75, linewidth=1.5, color=color, label=f"ep{ep}")
        # start point
        ax3d.scatter(ep_df["ee_x"].iloc[0], ep_df["ee_y"].iloc[0], ep_df["ee_z"].iloc[0],
                     marker="o", s=60, color=color, zorder=5)
        # target (mark once per episode - target may resample mid-episode)
        # use scatter for all target points (faint)
        ax3d.scatter(ep_df["tgt_x"], ep_df["tgt_y"], ep_df["tgt_z"],
                     marker="*", s=30, color=color, alpha=0.3)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D EE Trajectory (circle=start, star=target)")
    ax3d.legend(fontsize=7, loc="upper left")

    # ----- 2) Distance to target over steps -----
    ax_dist = fig.add_subplot(222)
    for i, ep in enumerate(episodes[:max_episodes_to_plot]):
        ep_df = df[df["episode"] == ep]
        ax_dist.plot(ep_df["step_in_ep"], ep_df["distance"],
                     alpha=0.8, linewidth=1.5, color=cmap(i % 10), label=f"ep{ep}")
    ax_dist.axhline(y=0.03, color="red", linestyle="--", linewidth=1.5, label="3cm threshold")
    ax_dist.set_xlabel("Step in Episode")
    ax_dist.set_ylabel("Distance to Target (m)")
    ax_dist.set_title("Distance to Target Per Episode")
    ax_dist.legend(fontsize=7)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_ylim(bottom=0)

    # ----- 3) XYZ components over time (episode 0) -----
    ax_xyz = fig.add_subplot(223)
    ep0_df = df[df["episode"] == episodes[0]] if episodes else df
    if not ep0_df.empty:
        steps = ep0_df["step_in_ep"]
        ax_xyz.plot(steps, ep0_df["ee_x"], color="C0", linewidth=1.5, label="EE_x")
        ax_xyz.plot(steps, ep0_df["ee_y"], color="C1", linewidth=1.5, label="EE_y")
        ax_xyz.plot(steps, ep0_df["ee_z"], color="C2", linewidth=1.5, label="EE_z")
        ax_xyz.plot(steps, ep0_df["tgt_x"], color="C0", linestyle=":", linewidth=1.0, alpha=0.7, label="tgt_x")
        ax_xyz.plot(steps, ep0_df["tgt_y"], color="C1", linestyle=":", linewidth=1.0, alpha=0.7, label="tgt_y")
        ax_xyz.plot(steps, ep0_df["tgt_z"], color="C2", linestyle=":", linewidth=1.0, alpha=0.7, label="tgt_z")
    ax_xyz.set_xlabel("Step in Episode")
    ax_xyz.set_ylabel("Position (m)")
    ax_xyz.set_title(f"EE XYZ vs Target (Episode {episodes[0]})")
    ax_xyz.legend(fontsize=7, ncol=2)
    ax_xyz.grid(True, alpha=0.3)

    # ----- 4) Distance distribution histogram -----
    ax_hist = fig.add_subplot(224)
    ax_hist.hist(df["distance"], bins=60, edgecolor="black", color="steelblue", alpha=0.8)
    ax_hist.axvline(x=0.03, color="red", linestyle="--", linewidth=2, label="3cm threshold")
    success_rate = (df["distance"] < 0.03).mean() * 100
    ax_hist.set_xlabel("Distance to Target (m)")
    ax_hist.set_ylabel("Step Count")
    ax_hist.set_title(f"Distance Distribution (success rate {success_rate:.1f}%)")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = csv_path.replace(".csv", ".png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[TRAJ] Saved plot → {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play policy and record EE + target trajectory."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create environment (no video recording for trajectory mode)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -----------------------------------------------------------------------
    # Load policy
    # -----------------------------------------------------------------------
    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # -----------------------------------------------------------------------
    # Access underlying ManagerBasedRLEnv for data extraction
    # env (RslRlVecEnvWrapper) → .unwrapped → ManagerBasedRLEnv
    # -----------------------------------------------------------------------
    inner_env = env.unwrapped
    # keep unwrapping until we reach ManagerBasedRLEnv (has scene attribute)
    while not hasattr(inner_env, "scene") and hasattr(inner_env, "unwrapped"):
        inner_env = inner_env.unwrapped

    robot = inner_env.scene["robot"]

    # find end-effector body index
    ee_body_name = args_cli.ee_body
    body_names = list(robot.body_names) if hasattr(robot, "body_names") else list(robot.data.body_names)
    if ee_body_name not in body_names:
        print(f"[TRAJ] WARNING: EE body '{ee_body_name}' not found in: {body_names}")
        print(f"[TRAJ] Falling back to last body: '{body_names[-1]}'")
        ee_body_name = body_names[-1]
    ee_idx = body_names.index(ee_body_name)
    print(f"[TRAJ] EE body: '{ee_body_name}' (idx={ee_idx})")

    # -----------------------------------------------------------------------
    # Output file setup
    # -----------------------------------------------------------------------
    checkpoint_stem = Path(resume_path).stem  # e.g. "model_150"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(output_dir / f"traj_{checkpoint_stem}_{timestamp}.csv")

    # -----------------------------------------------------------------------
    # Recording loop
    # -----------------------------------------------------------------------
    records: list[dict] = []
    episode = 0
    step_in_ep = 0
    global_step = 0
    num_steps = args_cli.num_steps

    obs = env.get_observations()
    print(f"[TRAJ] Recording {num_steps} steps (env 0 only)...")

    while simulation_app.is_running() and global_step < num_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # ---- extract data from env 0 ----
        with torch.no_grad():
            # EE world position: [num_envs, num_bodies, 3] → env 0, ee_idx
            ee_pos_w = robot.data.body_pos_w[0, ee_idx, :3]

            # target position: robot base frame → world frame
            command = inner_env.command_manager.get_command("ee_pose")
            des_pos_b = command[0:1, :3]  # env 0, [1, 3]
            root_pos = robot.data.root_pos_w[0:1]   # [1, 3]
            root_quat = robot.data.root_quat_w[0:1]  # [1, 4]
            des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, des_pos_b)
            des_pos_w = des_pos_w[0]  # [3]

            distance = torch.norm(ee_pos_w - des_pos_w).item()

        records.append({
            "episode": episode,
            "step_in_ep": step_in_ep,
            "global_step": global_step,
            "ee_x": ee_pos_w[0].item(),
            "ee_y": ee_pos_w[1].item(),
            "ee_z": ee_pos_w[2].item(),
            "tgt_x": des_pos_w[0].item(),
            "tgt_y": des_pos_w[1].item(),
            "tgt_z": des_pos_w[2].item(),
            "distance": distance,
        })

        # detect episode boundary (done for env 0)
        done_flag = False
        if dones is not None:
            d = dones[0]
            done_flag = d.item() if hasattr(d, "item") else bool(d)

        if done_flag:
            episode += 1
            step_in_ep = 0
        else:
            step_in_ep += 1

        global_step += 1

        if global_step % 100 == 0:
            print(f"[TRAJ] step={global_step}/{num_steps}  ep={episode}  dist={distance:.4f}m")

    print(f"[TRAJ] Done. {len(records)} steps, {episode} episode resets.")

    # save + plot
    save_trajectory_csv(records, csv_path)
    plot_trajectory(csv_path)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
