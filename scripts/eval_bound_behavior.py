#!/usr/bin/env python3
"""Bound gait diagnostics for the existing Barkour/MJX PPO policy.

This script is diagnostics-only.  It does not edit rewards, STL specs,
thresholds, command sampling, PPO settings, or training code.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math as pymath
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))
(REPO_ROOT / "runs" / ".mplconfig").mkdir(parents=True, exist_ok=True)
if sys.platform != "darwin":
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_triton_gemm_any=True" not in xla_flags:
        xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

SRC_DIR = REPO_ROOT / "src"
CONFIG_DIR = REPO_ROOT / "configs"
for path in (SRC_DIR, CONFIG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
from jax import numpy as jp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brax import math
from brax.io import html
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from Barkour import BarkourEnv
from coeff_config import (
    BOUND_TO_TROT_EXIT,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    TROT_TO_BOUND_ENTER,
    TROT_TO_WALK_EXIT,
    WALK_TO_TROT_ENTER,
)

MODE_NAMES = {MODE_WALK: "WALK", MODE_TROT: "TROT", MODE_BOUND: "BOUND"}
FOOT_NAMES = ["FL", "HL", "FR", "HR"]
NATURAL_SWEEP = [1.50, 1.65, 1.75, 1.90, 2.10, 2.25]
GRAVITY = 9.81


def _install_running_stats_compat_patch():
    try:
        from brax.training.acme import running_statistics
    except Exception:
        return lambda: None

    cls = running_statistics.RunningStatisticsState
    dataclass_fields = getattr(cls, "__dataclass_fields__", {})
    if "std_eps" in dataclass_fields:
        return lambda: None

    orig_init = cls.__init__

    def patched_init(self, *args, std_eps=None, **kwargs):
        return orig_init(self, *args, **kwargs)

    cls.__init__ = patched_init

    def restore():
        cls.__init__ = orig_init

    return restore


def load_policy(env: BarkourEnv, checkpoint: str):
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    restore_patch = _install_running_stats_compat_patch()
    try:
        make_inference_fn, params, _ = ppo.train(
            environment=env,
            num_timesteps=0,
            episode_length=1000,
            normalize_observations=True,
            restore_checkpoint_path=checkpoint,
            network_factory=functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=(128, 128, 128, 128),
            ),
        )
    finally:
        restore_patch()

    return jax.jit(make_inference_fn(params, deterministic=True))


def mode_from_command(command: jp.ndarray) -> jp.ndarray:
    vx_abs = jp.abs(command[0])
    return jp.where(
        vx_abs >= TROT_TO_BOUND_ENTER,
        jp.array(MODE_BOUND, dtype=jp.int32),
        jp.where(
            vx_abs >= WALK_TO_TROT_ENTER,
            jp.array(MODE_TROT, dtype=jp.int32),
            jp.array(MODE_WALK, dtype=jp.int32),
        ),
    )


def command_for_step(args: argparse.Namespace, t: int) -> jp.ndarray:
    if args.ramp_vx is not None:
        start, end = args.ramp_vx
        denom = max(args.episode_length - 1, 1)
        vx = start + (end - start) * t / denom
    else:
        vx = args.command_vx
    return jp.array([vx, args.command_vy, args.command_yaw], dtype=jp.float32)


def rebuild_obs_with_info(env: BarkourEnv, state, info: Dict[str, Any]):
    obs_history = state.obs
    obs = env._get_obs(state.pipeline_state, info, obs_history)
    return state.replace(obs=obs, info=info)


def reset_eval_state(env: BarkourEnv, rng: jax.Array, command: jp.ndarray, force_mode: Optional[str]):
    state = env.reset(rng)
    info = dict(state.info)
    info["command"] = command
    info["mode"] = jp.array(MODE_BOUND if force_mode == "bound" else int(mode_from_command(command)), dtype=jp.int32)
    info["history_len"] = jp.array(0, dtype=jp.int32)
    obs = env._get_obs(state.pipeline_state, info, jp.zeros_like(state.obs))
    return state.replace(obs=obs, info=info)


def set_controlled_command(env: BarkourEnv, state, command: jp.ndarray, force_mode: Optional[str]):
    info = dict(state.info)
    info["command"] = command
    if force_mode == "bound":
        info["mode"] = jp.array(MODE_BOUND, dtype=jp.int32)
    return rebuild_obs_with_info(env, state, info)


def quat_to_euler_xyz(q: Sequence[float]) -> Tuple[float, float, float]:
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = pymath.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = pymath.copysign(pymath.pi / 2, sinp)
    else:
        pitch = pymath.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = pymath.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def scalar(x: Any) -> float:
    arr = np.asarray(jax.device_get(x))
    return float(arr.reshape(-1)[0])


def get_robot_mass(env: BarkourEnv, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    body_mass = np.asarray(env.sys.mj_model.body_mass, dtype=np.float64)
    return float(body_mass[1:].sum() if body_mass.size > 1 else body_mass.sum())


def extract_step_row(
    env: BarkourEnv,
    state,
    action: np.ndarray,
    prev_action: np.ndarray,
    t: int,
    episode_idx: int,
    condition: str,
    force_mode: Optional[str],
    command_override: Optional[np.ndarray] = None,
    mode_override: Optional[int] = None,
) -> Dict[str, float | str]:
    pipeline_state = state.pipeline_state
    x = pipeline_state.x
    xd = pipeline_state.xd
    command = (
        np.asarray(command_override, dtype=np.float64)
        if command_override is not None
        else np.asarray(jax.device_get(state.info["command"]), dtype=np.float64)
    )
    mode = int(mode_override) if mode_override is not None else int(np.asarray(jax.device_get(state.info["mode"])))
    mode_one_hot = np.zeros(3, dtype=np.float64)
    if 0 <= mode < 3:
        mode_one_hot[mode] = 1.0

    local_vel = np.asarray(jax.device_get(math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))), dtype=np.float64)
    local_ang = np.asarray(jax.device_get(math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))), dtype=np.float64)
    pos = np.asarray(jax.device_get(x.pos[env._torso_idx - 1]), dtype=np.float64)
    quat = np.asarray(jax.device_get(x.rot[0]), dtype=np.float64)
    roll, pitch, yaw = quat_to_euler_xyz(quat)
    foot_xyz = np.asarray(jax.device_get(pipeline_state.site_xpos[env._feet_site_id]), dtype=np.float64)
    contacts = np.asarray(jax.device_get(state.info["contact_history"][-1]), dtype=np.float64)
    torques = np.asarray(jax.device_get(pipeline_state.qfrc_actuator), dtype=np.float64)
    qd = np.asarray(jax.device_get(pipeline_state.qd), dtype=np.float64)
    rewards = {k: scalar(v) for k, v in state.info["rewards"].items()}

    row: Dict[str, float | str] = {
        "condition": condition,
        "episode": episode_idx,
        "timestep": t,
        "time_s": t * float(env.dt),
        "forced_mode": force_mode or "",
        "command_vx": command[0],
        "command_vy": command[1],
        "command_yaw": command[2],
        "realized_vx": local_vel[0],
        "realized_vy": local_vel[1],
        "realized_vz": local_vel[2],
        "realized_yaw_rate": local_ang[2],
        "base_x": pos[0],
        "base_y": pos[1],
        "base_z": pos[2],
        "base_height": pos[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "ang_vel_x": local_ang[0],
        "ang_vel_y": local_ang[1],
        "ang_vel_z": local_ang[2],
        "active_gait_mode": mode,
        "active_gait_mode_name": MODE_NAMES.get(mode, "UNKNOWN"),
        "mode_one_hot_walk": mode_one_hot[0],
        "mode_one_hot_trot": mode_one_hot[1],
        "mode_one_hot_bound": mode_one_hot[2],
        "done": scalar(state.done),
        "reward_total": scalar(state.reward),
        "action_l2": float(np.linalg.norm(action)),
        "action_delta_l2": float(np.linalg.norm(action - prev_action)),
        "torque_l2": float(np.linalg.norm(torques)),
        "torque_abs_sum": float(np.sum(np.abs(torques))),
        "actuated_power_abs": float(np.sum(np.abs(torques[6:] * qd[6:]))),
        "slip_proxy": scalar(state.info["slipmax_history"][-1]),
    }
    for i, foot in enumerate(FOOT_NAMES):
        row[f"contact_{foot}"] = contacts[i]
        row[f"foot_{foot}_x"] = foot_xyz[i, 0]
        row[f"foot_{foot}_y"] = foot_xyz[i, 1]
        row[f"foot_{foot}_z"] = foot_xyz[i, 2]
        row[f"foot_{foot}_clearance"] = foot_xyz[i, 2] - float(env._foot_radius)
    for i, value in enumerate(action):
        row[f"action_{i:02d}"] = float(value)
    for k, v in rewards.items():
        row[f"reward_{k}"] = v
    return row


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.mean(a == b))


def corr_binary(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def count_clean_bound_cycles(contact: np.ndarray) -> int:
    # contact foot order is [FL, HL, FR, HR].
    front_only = (contact[:, 0] == 1) & (contact[:, 2] == 1) & (contact[:, 1] == 0) & (contact[:, 3] == 0)
    hind_only = (contact[:, 1] == 1) & (contact[:, 3] == 1) & (contact[:, 0] == 0) & (contact[:, 2] == 0)
    cycles = 0
    seen_hind = False
    prev_state = ""
    for f, h in zip(front_only, hind_only):
        state = "front" if f else "hind" if h else ""
        if state == "hind" and prev_state != "hind":
            seen_hind = True
        if state == "front" and prev_state != "front" and seen_hind:
            cycles += 1
            seen_hind = False
        if state:
            prev_state = state
    return cycles


def bound_quality_score(metrics: Dict[str, float]) -> float:
    vx_denom = max(abs(metrics["mean_commanded_vx"]), 0.1)
    vx_tracking = max(0.0, 1.0 - abs(metrics["vx_tracking_error"]) / vx_denom)
    flight = min(max(metrics["all_feet_airborne_fraction"] / 0.05, 0.0), 1.0)
    diag = 1.0 - np.nanmean([metrics["diag_FL_HR_agreement"], metrics["diag_FR_HL_agreement"]])
    diag = float(np.clip(diag, 0.0, 1.0))
    alternation = float(np.clip(metrics["front_hind_alternation_fraction"], 0.0, 1.0))
    pair_sync = float(np.nanmean([metrics["front_pair_agreement"], metrics["hind_pair_agreement"]]))
    stability_angle = max(metrics["max_abs_roll"], metrics["max_abs_pitch"])
    stability = max(0.0, 1.0 - stability_angle / 0.7)
    survival = float(np.clip(metrics["survival_fraction"], 0.0, 1.0))
    score = (
        0.20 * pair_sync
        + 0.15 * diag
        + 0.15 * alternation
        + 0.10 * flight
        + 0.15 * stability
        + 0.15 * vx_tracking
        + 0.10 * survival
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_episode_metrics(
    rows: List[Dict[str, float | str]],
    robot_mass: float,
    env_dt: float,
    planned_steps: int,
    executed_steps: Optional[int] = None,
    early_terminated: Optional[float] = None,
) -> Dict[str, float | str]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "condition": "empty",
            "episode": -1,
            "forced_mode": "",
            "executed_steps": executed_steps or 0,
            "survival_time_s": float((executed_steps or 0) * env_dt),
            "survival_fraction": float((executed_steps or 0) / max(planned_steps, 1)),
            "early_termination": float(early_terminated or 0.0),
            "fall_rate": float(early_terminated or 0.0),
            "bound_quality_score": 0.0,
        }
    contact = df[[f"contact_{f}" for f in FOOT_NAMES]].to_numpy(dtype=np.int32)
    mode = df["active_gait_mode"].to_numpy(dtype=np.int32)
    command_vx = df["command_vx"].to_numpy(dtype=np.float64)
    realized_vx = df["realized_vx"].to_numpy(dtype=np.float64)
    n = len(df)
    if executed_steps is None:
        done = df["done"].to_numpy(dtype=np.float64)
        first_done = np.where(done > 0.0)[0]
        executed_steps = int(first_done[0] + 1) if first_done.size else n
    if early_terminated is None:
        early_terminated = float(executed_steps < planned_steps)
    survival_time = executed_steps * env_dt
    bound_idx = np.where(mode == MODE_BOUND)[0]
    mode_switches = int(np.sum(mode[1:] != mode[:-1])) if n > 1 else 0
    drops_out = float(np.any((mode[:-1] == MODE_BOUND) & (mode[1:] != MODE_BOUND))) if n > 1 else 0.0
    n_contacts = contact.sum(axis=1)
    front_pair = (contact[:, 0] == 1) & (contact[:, 2] == 1)
    hind_pair = (contact[:, 1] == 1) & (contact[:, 3] == 1)
    front_only = front_pair & (contact[:, 1] == 0) & (contact[:, 3] == 0)
    hind_only = hind_pair & (contact[:, 0] == 0) & (contact[:, 2] == 0)
    flight = n_contacts == 0
    all4 = n_contacts == 4
    two_contact = n_contacts == 2
    diagonal_2 = (
        ((contact[:, 0] == 1) & (contact[:, 3] == 1) & (contact[:, 1] == 0) & (contact[:, 2] == 0))
        | ((contact[:, 2] == 1) & (contact[:, 1] == 1) & (contact[:, 0] == 0) & (contact[:, 3] == 0))
    )
    foot_clearance = {
        f"avg_swing_clearance_{foot}": float(df.loc[df[f"contact_{foot}"] < 0.5, f"foot_{foot}_clearance"].mean())
        for foot in FOOT_NAMES
    }
    dx = np.diff(df["base_x"].to_numpy(dtype=np.float64), prepend=df["base_x"].iloc[0])
    dy = np.diff(df["base_y"].to_numpy(dtype=np.float64), prepend=df["base_y"].iloc[0])
    planar_dist = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    energy = float(np.sum(df["actuated_power_abs"].to_numpy(dtype=np.float64)) * env_dt)
    cot = energy / max(robot_mass * GRAVITY * planar_dist, 1e-8)
    metrics: Dict[str, float | str] = {
        "condition": str(df["condition"].iloc[0]),
        "episode": int(df["episode"].iloc[0]),
        "forced_mode": str(df["forced_mode"].iloc[0]),
        "mean_commanded_vx": float(np.mean(command_vx)),
        "mean_realized_vx": float(np.mean(realized_vx)),
        "vx_tracking_error": float(np.mean(realized_vx - command_vx)),
        "abs_vx_tracking_error": float(np.mean(np.abs(realized_vx - command_vx))),
        "max_realized_vx": float(np.max(realized_vx)),
        "executed_steps": executed_steps,
        "survival_time_s": survival_time,
        "survival_fraction": float(executed_steps / max(planned_steps, 1)),
        "early_termination": early_terminated,
        "fall_rate": early_terminated,
        "bound_fraction": float(np.mean(mode == MODE_BOUND)),
        "time_to_enter_bound_s": float(bound_idx[0] * env_dt) if bound_idx.size else float("nan"),
        "mode_switches": mode_switches,
        "drops_out_of_bound": drops_out,
        "all_feet_airborne_fraction": float(np.mean(flight)),
        "front_only_fraction": float(np.mean(front_only)),
        "hind_only_fraction": float(np.mean(hind_only)),
        "all4_fraction": float(np.mean(all4)),
        "two_contact_fraction": float(np.mean(two_contact)),
        "diagonal_two_contact_fraction": float(np.mean(diagonal_2[two_contact])) if np.any(two_contact) else 0.0,
        "front_hind_alternation_fraction": float(np.mean(front_pair != hind_pair)),
        "front_pair_agreement": agreement(contact[:, 0], contact[:, 2]),
        "hind_pair_agreement": agreement(contact[:, 1], contact[:, 3]),
        "diag_FL_HR_agreement": agreement(contact[:, 0], contact[:, 3]),
        "diag_FR_HL_agreement": agreement(contact[:, 2], contact[:, 1]),
        "front_pair_corr": corr_binary(contact[:, 0], contact[:, 2]),
        "hind_pair_corr": corr_binary(contact[:, 1], contact[:, 3]),
        "diag_FL_HR_corr": corr_binary(contact[:, 0], contact[:, 3]),
        "diag_FR_HL_corr": corr_binary(contact[:, 2], contact[:, 1]),
        "clean_bound_cycles": count_clean_bound_cycles(contact),
        "mean_roll": float(df["roll"].mean()),
        "max_abs_roll": float(np.max(np.abs(df["roll"].to_numpy(dtype=np.float64)))),
        "mean_pitch": float(df["pitch"].mean()),
        "max_abs_pitch": float(np.max(np.abs(df["pitch"].to_numpy(dtype=np.float64)))),
        "pitch_oscillation_amplitude": float(df["pitch"].max() - df["pitch"].min()),
        "base_height_mean": float(df["base_height"].mean()),
        "base_height_min": float(df["base_height"].min()),
        "base_height_max": float(df["base_height"].max()),
        "action_l2_mean": float(df["action_l2"].mean()),
        "action_delta_l2_mean": float(df["action_delta_l2"].mean()),
        "torque_l2_mean": float(df["torque_l2"].mean()),
        "actuated_energy_proxy": energy,
        "cost_of_transport_proxy": cot,
        "slip_proxy_mean": float(df["slip_proxy"].mean()),
        "reward_total_mean": float(df["reward_total"].mean()),
    }
    metrics.update(foot_clearance)
    for col in [c for c in df.columns if c.startswith("reward_")]:
        metrics[f"{col}_mean"] = float(df[col].mean())
    metrics["bound_quality_score"] = bound_quality_score(metrics)  # type: ignore[arg-type]
    return metrics


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = ["condition"]
    agg = df.groupby(group_cols)[numeric].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
    return agg.reset_index()


def plot_episode(df: pd.DataFrame, out_dir: Path, condition: str, episode: int) -> None:
    ep = df[(df["condition"] == condition) & (df["episode"] == episode)].copy()
    if ep.empty:
        return
    t = ep["time_s"].to_numpy()
    safe_condition = condition.replace("/", "_")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, ep["command_vx"], label="cmd vx")
    ax.plot(t, ep["realized_vx"], label="realized vx")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("vx [m/s]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_condition}_ep{episode}_vx_tracking.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, ep["base_height"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("base height [m]")
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_condition}_ep{episode}_base_height.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, ep["roll"], label="roll")
    ax.plot(t, ep["pitch"], label="pitch")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_condition}_ep{episode}_roll_pitch.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.step(t, ep["active_gait_mode"], where="post")
    ax.set_yticks([MODE_WALK, MODE_TROT, MODE_BOUND], ["WALK", "TROT", "BOUND"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mode")
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_condition}_ep{episode}_active_gait_mode.png", dpi=160)
    plt.close(fig)

    contacts = ep[[f"contact_{f}" for f in FOOT_NAMES]].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.imshow(contacts.T, aspect="auto", interpolation="nearest", cmap="Greys", extent=[t[0], t[-1], 0, 4])
    ax.set_yticks(np.arange(0.5, 4.5), FOOT_NAMES)
    ax.set_xlabel("time [s]")
    ax.set_title("contact raster")
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_condition}_ep{episode}_contact_raster.png", dpi=160)
    plt.close(fig)

    reward_cols = [
        c
        for c in [
            "reward_rho_safety",
            "reward_rho_tracking",
            "reward_rho_pattern",
            "reward_rho_bound",
            "reward_rho_trot",
            "reward_rho_walk",
            "reward_total_stl_reward",
            "reward_bound_front_or_hind_pair_support",
            "reward_bound_diagonal_trot_penalty",
            "reward_bound_all_four_stance_penalty",
            "reward_bound_contact_pattern_reward",
            "reward_bound_contact_pattern_bonus",
            "reward_bound_raw_contact_pattern_bonus",
            "reward_bound_contact_gate",
            "reward_bound_vx_tracking_gate",
            "reward_bound_stability_gate",
            "reward_bound_transition_guard_adjustment",
            "reward_bound_pattern_gate",
            "reward_bound_pattern_gate_penalty",
            "reward_bound_tracking_guard_penalty",
            "reward_transition_tracking_guard_penalty",
            "reward_bound_forward_progress_penalty",
            "reward_bound_all_four_stall_penalty",
            "reward_bound_forward_progress_margin",
            "reward_bound_all_four_stall_fraction",
        ]
        if c in ep.columns
    ]
    if reward_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in reward_cols:
            ax.plot(t, ep[col], label=col.replace("reward_", ""))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("reward/component")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_condition}_ep{episode}_reward_components.png", dpi=160)
        plt.close(fig)


def plot_summary(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    if metrics_df.empty:
        return
    natural = metrics_df[metrics_df["condition"].str.startswith("natural")]
    if not natural.empty:
        mode_cols = ["bound_fraction"]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        natural.groupby("mean_commanded_vx")["bound_fraction"].mean().plot(kind="bar", ax=ax)
        ax.set_xlabel("command vx [m/s]")
        ax.set_ylabel("BOUND fraction")
        fig.tight_layout()
        fig.savefig(out_dir / "time_spent_in_bound_by_command.png", dpi=160)
        plt.close(fig)

    for y, ylabel, fname in [
        ("mean_realized_vx", "realized vx [m/s]", "sweep_command_vx_vs_realized_vx.png"),
        ("survival_time_s", "survival time [s]", "sweep_command_vx_vs_survival_time.png"),
        ("bound_fraction", "BOUND fraction", "sweep_command_vx_vs_bound_fraction.png"),
        ("bound_quality_score", "bound quality score", "sweep_command_vx_vs_bound_quality_score.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for condition_type, sdf in metrics_df.groupby(metrics_df["condition"].str.split("_vx").str[0]):
            grouped = sdf.groupby("mean_commanded_vx")[y].mean().reset_index()
            ax.plot(grouped["mean_commanded_vx"], grouped[y], marker="o", label=condition_type)
        ax.set_xlabel("command vx [m/s]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def save_render_html(path: Path, env: BarkourEnv, states: List[Any]) -> None:
    if not states:
        return
    html.save(str(path), env.sys, states)


def rollout_condition(
    args: argparse.Namespace,
    env: BarkourEnv,
    inference_fn,
    jit_step,
    rng: jax.Array,
    condition: str,
    force_mode: Optional[str],
    robot_mass: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Any], jax.Array]:
    trace_rows: List[Dict[str, Any]] = []
    episode_metrics: List[Dict[str, Any]] = []
    render_states: List[Any] = []
    for episode_idx in range(args.num_episodes):
        rng, reset_rng = jax.random.split(rng)
        initial_command = command_for_step(args, 0)
        state = reset_eval_state(env, reset_rng, initial_command, force_mode)
        prev_action = np.zeros(12, dtype=np.float64)
        ep_rows: List[Dict[str, Any]] = []
        for t in range(args.episode_length):
            command = command_for_step(args, t)
            state = set_controlled_command(env, state, command, force_mode)
            mode_before_step = int(np.asarray(jax.device_get(state.info["mode"])))
            action, _ = inference_fn(state.obs, rng)
            action_np = np.asarray(jax.device_get(action), dtype=np.float64)
            state = jit_step(state, action)
            done_flag = float(np.asarray(jax.device_get(state.done))) != 0.0
            if done_flag:
                # BarkourEnv clears command-conditioned histories immediately on
                # termination, so the post-step state.info no longer represents
                # the terminal physics sample.  Exclude that reset row from
                # contact/gait summaries and record termination separately.
                break
            row = extract_step_row(
                env,
                state,
                action_np,
                prev_action,
                t,
                episode_idx,
                condition,
                force_mode,
                command_override=np.asarray(jax.device_get(command), dtype=np.float64),
                mode_override=mode_before_step,
            )
            ep_rows.append(row)
            trace_rows.append(row)
            prev_action = action_np
            if episode_idx == 0 and args.render and (t % args.render_stride == 0):
                render_states.append(jax.device_get(state.pipeline_state))
        episode_metrics.append(
            compute_episode_metrics(
                ep_rows,
                robot_mass,
                float(env.dt),
                planned_steps=args.episode_length,
                executed_steps=t + 1,
                early_terminated=float(done_flag),
            )
        )
    return trace_rows, episode_metrics, render_states, rng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "--ckpt-path", dest="checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_episodes", "--num-tests", type=int, default=5)
    parser.add_argument("--episode_length", "--horizon", type=int, default=500)
    parser.add_argument("--command_vx", type=float, default=1.90)
    parser.add_argument("--command_vy", type=float, default=0.0)
    parser.add_argument("--command_yaw", type=float, default=0.0)
    parser.add_argument("--force_mode", choices=["bound"], default=None)
    parser.add_argument("--sweep_bound_commands", action="store_true")
    parser.add_argument("--include_forced_sweep", action="store_true")
    parser.add_argument("--include_stability_checks", action="store_true")
    parser.add_argument("--ramp_vx", type=float, nargs=2, default=None)
    parser.add_argument("--render", "--save_video", action="store_true")
    parser.add_argument("--render_stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs_noise", type=float, default=0.0)
    parser.add_argument("--kick_vel", type=float, default=0.0)
    parser.add_argument("--robot_mass", type=float, default=None)
    parser.add_argument("--experiment", default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    traces_dir = out_dir / "traces"
    videos_dir = out_dir / "videos"
    for d in (plots_dir, traces_dir, videos_dir):
        d.mkdir(parents=True, exist_ok=True)

    env_for_policy = BarkourEnv(
        obs_noise=args.obs_noise,
        kick_vel=args.kick_vel,
        experiment=args.experiment,
    )
    inference_fn = load_policy(env_for_policy, args.checkpoint)
    env = BarkourEnv(
        obs_noise=args.obs_noise,
        kick_vel=args.kick_vel,
        experiment=args.experiment,
    )
    jit_step = jax.jit(env.step)
    robot_mass = get_robot_mass(env, args.robot_mass)
    rng = jax.random.PRNGKey(args.seed)

    original_command = (args.command_vx, args.command_vy, args.command_yaw, args.ramp_vx)
    conditions: List[Tuple[str, Optional[str], float, float, float, Optional[Tuple[float, float]]]] = []
    if args.sweep_bound_commands:
        for vx in NATURAL_SWEEP:
            conditions.append((f"natural_vx{vx:.2f}", None, vx, 0.0, 0.0, None))
        if args.include_forced_sweep:
            for vx in NATURAL_SWEEP:
                conditions.append((f"forced_bound_vx{vx:.2f}", "bound", vx, 0.0, 0.0, None))
    else:
        label = "forced_bound" if args.force_mode else "natural"
        suffix = f"vx{args.command_vx:.2f}_vy{args.command_vy:.2f}_yaw{args.command_yaw:.2f}"
        conditions.append((f"{label}_{suffix}", args.force_mode, args.command_vx, args.command_vy, args.command_yaw, tuple(args.ramp_vx) if args.ramp_vx else None))

    if args.include_stability_checks:
        conditions.extend(
            [
                ("natural_vx1.90_yaw0.15", None, 1.90, 0.0, 0.15, None),
                ("natural_vx1.90_vy0.15", None, 1.90, 0.15, 0.0, None),
                ("natural_ramp_vx1.40_to_2.20", None, 1.40, 0.0, 0.0, (1.40, 2.20)),
            ]
        )

    all_trace_rows: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, Any]] = []
    rendered_paths: List[str] = []
    commands_run: List[str] = []

    for condition, force_mode, vx, vy, yaw, ramp in conditions:
        args.command_vx = vx
        args.command_vy = vy
        args.command_yaw = yaw
        args.ramp_vx = list(ramp) if ramp is not None else None
        trace_rows, metrics, render_states, rng = rollout_condition(
            args=args,
            env=env,
            inference_fn=inference_fn,
            jit_step=jit_step,
            rng=rng,
            condition=condition,
            force_mode=force_mode,
            robot_mass=robot_mass,
        )
        write_csv(traces_dir / f"{condition}_trace.csv", trace_rows)
        write_csv(traces_dir / f"{condition}_episode_metrics.csv", metrics)
        if render_states:
            render_path = videos_dir / f"{condition}_episode0.html"
            save_render_html(render_path, env, render_states)
            rendered_paths.append(str(render_path))
        all_trace_rows.extend(trace_rows)
        all_metrics.extend(metrics)
        commands_run.append(
            f"{condition}: vx={vx}, vy={vy}, yaw={yaw}, ramp_vx={ramp}, force_mode={force_mode or 'natural'}"
        )

    args.command_vx, args.command_vy, args.command_yaw, args.ramp_vx = original_command
    write_csv(out_dir / "all_traces.csv", all_trace_rows)
    write_csv(out_dir / "episode_metrics.csv", all_metrics)
    summary_df = aggregate_metrics(all_metrics)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    metrics_df = pd.DataFrame(all_metrics)
    trace_df = pd.DataFrame(all_trace_rows)

    if not trace_df.empty:
        for condition in trace_df["condition"].drop_duplicates():
            plot_episode(trace_df, plots_dir, str(condition), 0)
    plot_summary(metrics_df, plots_dir)

    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "out_dir": str(out_dir.resolve()),
        "num_episodes": args.num_episodes,
        "episode_length": args.episode_length,
        "seed": args.seed,
        "obs_noise": args.obs_noise,
        "kick_vel": args.kick_vel,
        "experiment": args.experiment,
        "robot_mass": robot_mass,
        "thresholds": {
            "WALK_TO_TROT_ENTER": WALK_TO_TROT_ENTER,
            "TROT_TO_WALK_EXIT": TROT_TO_WALK_EXIT,
            "TROT_TO_BOUND_ENTER": TROT_TO_BOUND_ENTER,
            "BOUND_TO_TROT_EXIT": BOUND_TO_TROT_EXIT,
        },
        "commands_run": commands_run,
        "rendered_paths": rendered_paths,
        "unavailable_fields": [
            "termination_reason: BarkourEnv exposes done but no explicit reason in state.info",
            "native force_mode: environment has no public mode-forcing API; --force_mode bound forces the evaluation observation/state mode before each policy action and is labeled forced-bound diagnostic",
        ],
        "bound_quality_score_formula": (
            "0.20*pair_sync + 0.15*(1-diagonal_agreement) + 0.15*front_hind_alternation "
            "+ 0.10*min(flight_fraction/0.05,1) + 0.15*stability "
            "+ 0.15*vx_tracking + 0.10*survival"
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved traces: {out_dir / 'all_traces.csv'}")
    print(f"Saved episode metrics: {out_dir / 'episode_metrics.csv'}")
    print(f"Saved summary metrics: {out_dir / 'summary_metrics.csv'}")
    print(f"Saved plots: {plots_dir}")
    if rendered_paths:
        print(f"Saved render HTML: {rendered_paths}")


if __name__ == "__main__":
    main()
