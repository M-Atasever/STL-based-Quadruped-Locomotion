#!/usr/bin/env python3
"""Collect expertish Barkour rollouts and fit grouped STL reward weights/alphas.

This version is aligned with the current grouped STL reward design used by
Barkour.py + stl_reward.py:

    r =
        w_safe   * tanh(rho_safety   / alpha_safe)
      + w_track  * tanh(rho_tracking / alpha_track)
      + w_timing * tanh(rho_timing   / alpha_timing)
      + w_pattern* tanh(rho_pattern  / alpha_pattern)
      - gamma_tau * tau_effort

It loads one checkpoint per regime, collects rollouts, saves datasets, and
suggests per-regime grouped weights/alphas for coeff_config.py.

Outputs under --out-dir:
  - grouped_rollouts_steps.csv
  - grouped_rollouts_trajectories.csv
  - suggested_grouped_reward_params.json
  - suggested_grouped_reward_params.txt
  - collection_metadata.json

Practical defaults are intentionally modest so the script finishes in a
reasonable amount of time for tuning:
  * obs_noise=0.0
  * kick_vel=0.0
  * stop_on_done=True
  * n_traj_per_regime=40
  * T=300

Notes:
  * This still uses ppo.train(..., num_timesteps=0, restore_checkpoint_path=...)
    because that matches your current testing / restore flow.
  * The first load per checkpoint may take a while because of JAX/Brax restore
    and compilation.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
from jax import numpy as jp
import numpy as np
import pandas as pd
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

from Barkour import BarkourEnv
from coeff_config import (
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    TROT_TO_BOUND_ENTER,
    WALK_TO_TROT_ENTER,
)
from stl_reward import reward_step

os.environ.setdefault("MUJOCO_GL", "egl")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
    _xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = _xla_flags

TARGET_TANH_OUTPUT = 0.80
ATANH_TARGET = float(np.arctanh(TARGET_TANH_OUTPUT))
MIN_ALPHA = 1e-3

@dataclass(frozen=True)
class RegimeSpec:
    name: str
    ckpt: str
    vx_min: float
    vx_max: float
    mode_id: int


def _mode_from_command(command: jp.ndarray) -> jp.ndarray:
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


def reset_eval_state_with_command(env: BarkourEnv, rng: jax.Array, command: jp.ndarray):
    state = env.reset(rng)
    info = dict(state.info)
    info["command"] = command
    info["mode"] = _mode_from_command(command)
    info["history_len"] = jp.array(0, dtype=jp.int32)
    obs_history = jp.zeros_like(state.obs)
    obs = env._get_obs(state.pipeline_state, info, obs_history)
    return state.replace(obs=obs, info=info)


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


def load_policy(env: BarkourEnv, ckpt_path: str):
    restore_patch = _install_running_stats_compat_patch()
    try:
        make_inference_fn, params, _ = ppo.train(
            environment=env,
            num_timesteps=0,
            episode_length=1000,
            normalize_observations=True,
            restore_checkpoint_path=ckpt_path,
            network_factory=functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=(128, 128, 128, 128),
            ),
        )
    finally:
        restore_patch()
    inference_fn = make_inference_fn(params)
    return jax.jit(inference_fn)


def to_scalar(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    raise ValueError(f"Expected scalar, got shape {arr.shape}")


def build_reward_input(state_info: Dict[str, jp.ndarray]) -> Dict[str, jp.ndarray]:
    return {
        "tau_history": state_info["tau_history"],
        "CoM_history": state_info["CoM_history"],
        "contact_history": state_info["contact_history"],
        "feet_history": state_info["feet_history"],
        "lin_velocity_history": state_info["lin_velocity_history"],
        "ang_velocity_history": state_info["ang_velocity_history"],
        "com_z_history": state_info["com_z_history"],
        "roll_history": state_info["roll_history"],
        "pitch_history": state_info["pitch_history"],
        "slipmax_history": state_info["slipmax_history"],
    }


@jax.jit
def _compute_reward_tuple_jit(state_info, command, mode, history_len):
    return reward_step(build_reward_input(state_info), command, mode, history_len, None)


def compute_reward_terms(state) -> Dict[str, float]:
    vals = _compute_reward_tuple_jit(state.info, state.info["command"], state.info["mode"], state.info["history_len"])
    (
        reward,
        tau_effort,
        rho_safety,
        rho_torque,
        rho_comz,
        rho_roll,
        rho_pitch,
        rho_slip,
        rho_bound,
        rho_trot,
        rho_walk,
        rho_v_x,
        rho_v_y,
        rho_yaw,
        rho_nlegs,
        rho_gait,
        rho_v_x_error,
        rho_v_y_error,
        rho_yaw_error,
        rho_diag2,
        rho_stride,
        rho_duty,
        rho_3plus_event,
        rho_support,
        rho_diag_phase,
        rho_p2,
        rho_tracking,
        rho_timing,
        rho_pattern,
        rho_front,
        rho_hind,
        rho_hindfront,
        rho_flight,
        rho_front_only,
        rho_hind_only,
        pitch,
        roll,
        FL, 
        HL, 
        FR, 
        HR,
        rho_all4,
        rho_bound_event,
     #   slip,
      #  denom,
       # support_dist,
    ) = vals
    return {
        "reward_recomputed": float(to_scalar(reward)),
        "tau_effort": float(to_scalar(tau_effort)),
        "rho_safety": float(to_scalar(rho_safety)),
        "rho_torque": float(to_scalar(rho_torque)),
        "rho_comz": float(to_scalar(rho_comz)),
        "rho_roll": float(to_scalar(rho_roll)),
        "rho_pitch": float(to_scalar(rho_pitch)),
        "rho_slip": float(to_scalar(rho_slip)),
        "rho_bound": float(to_scalar(rho_bound)),
        "rho_trot": float(to_scalar(rho_trot)),
        "rho_walk": float(to_scalar(rho_walk)),
        "rho_vx": float(to_scalar(rho_v_x)),
        "rho_vy": float(to_scalar(rho_v_y)),
        "rho_yaw": float(to_scalar(rho_yaw)),
        "rho_nlegs": float(to_scalar(rho_nlegs)),
        "rho_gait": float(to_scalar(rho_gait)),
        "vx_error_mean": float(to_scalar(rho_v_x_error)),
        "vy_error_mean": float(to_scalar(rho_v_y_error)),
        "yaw_error_mean": float(to_scalar(rho_yaw_error)),
        "rho_diag2": float(to_scalar(rho_diag2)),
        "rho_stride": float(to_scalar(rho_stride)),
        "rho_duty": float(to_scalar(rho_duty)),
        "rho_3plus_event": float(to_scalar(rho_3plus_event)),
        "rho_support": float(to_scalar(rho_support)),
        "rho_diag_phase": float(to_scalar(rho_diag_phase)),
        "rho_p2": float(to_scalar(rho_p2)),
        "rho_tracking": float(to_scalar(rho_tracking)),
        "rho_timing": float(to_scalar(rho_timing)),
        "rho_pattern": float(to_scalar(rho_pattern)),
        "rho_front": float(to_scalar(rho_front)),
        "rho_hind": float(to_scalar(rho_hind)),
        "rho_hindfront": float(to_scalar(rho_hindfront)),
        "rho_flight": float(to_scalar(rho_flight)),
        "rho_front_only": float(to_scalar(rho_front_only)),
        "rho_hind_only": float(to_scalar(rho_hind_only)),
        "pitch": float(to_scalar(pitch)),
        "roll": float(to_scalar(roll)),
        "FL": float(to_scalar(FL)), 
        "HL": float(to_scalar(HL)), 
        "FR": float(to_scalar(FR)),
        "HR": float(to_scalar(HR)),
        "rho_all4": float(to_scalar(rho_all4)),
        "rho_bound_event": float(to_scalar(rho_bound_event)),
        #"slip": float(to_scalar(slip)),
        #"denom":float(to_scalar(denom)),
        #"support_dist":float(to_scalar(support_dist)),
    }


def sample_command(rng: jax.Array, vx_min: float, vx_max: float, vy_max: float, yaw_max: float) -> Tuple[jax.Array, jp.ndarray]:
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    vx = jax.random.uniform(k1, (), minval=vx_min, maxval=vx_max)
    if vy_max > 0:
        vy = jax.random.uniform(k2, (), minval=-vy_max, maxval=vy_max)
    else:
        vy = jp.array(0.0, dtype=jp.float32)
    if yaw_max > 0:
        yaw = jax.random.uniform(k3, (), minval=-yaw_max, maxval=yaw_max)
    else:
        yaw = jp.array(0.0, dtype=jp.float32)
    cmd = jp.array([vx, vy, yaw], dtype=jp.float32)
    return rng, cmd


def _warmup_compilation(eval_env: BarkourEnv, policy_fn, rng: jax.Array, regime: RegimeSpec, vy_max: float, yaw_max: float):
    rng, k_reset = jax.random.split(rng)
    rng, cmd = sample_command(rng, regime.vx_min, regime.vx_max, vy_max, yaw_max)
    state = reset_eval_state_with_command(eval_env, k_reset, cmd)
    rng, act_rng = jax.random.split(rng)
    _ = policy_fn(state.obs, act_rng)
    _ = compute_reward_terms(state)
    return rng


def collect_regime_rollouts(regime: RegimeSpec, n_traj: int, n_steps: int, seed: int, out_vy_max: float, out_yaw_max: float, obs_noise: float, kick_vel: float, stop_on_done: bool, progress_every: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    load_t0 = time.perf_counter()
    env_for_policy = BarkourEnv(obs_noise=obs_noise, kick_vel=kick_vel)
    jit_inference_fn = load_policy(env_for_policy, regime.ckpt)
    eval_env = BarkourEnv(obs_noise=obs_noise, kick_vel=kick_vel)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(seed)
    rng = _warmup_compilation(eval_env, jit_inference_fn, rng, regime, out_vy_max, out_yaw_max)
    load_dt = time.perf_counter() - load_t0
    step_rows: List[Dict[str, float]] = []
    traj_rows: List[Dict[str, float]] = []
    print(f"[{regime.name}] policy restore + warmup took {load_dt:.1f}s")
    for local_traj_id in range(n_traj):
        traj_t0 = time.perf_counter()
        rng, k_reset = jax.random.split(rng)
        rng, cmd = sample_command(rng, regime.vx_min, regime.vx_max, out_vy_max, out_yaw_max)
        state = reset_eval_state_with_command(eval_env, k_reset, cmd)
        traj_reward_sum = 0.0
        traj_done = False
        executed_steps = 0
        last_terms: Dict[str, float] = {}
        for t in range(n_steps):
            rng, act_rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            terms = compute_reward_terms(state)
            last_terms = terms
            done_flag = bool(to_scalar(state.done))
            info = state.info
            metrics = dict(state.metrics)
            row = {
                "regime": regime.name,
                "traj_id": local_traj_id,
                "step": t,
                "done": int(done_flag),
                "mode_from_info": int(to_scalar(info["mode"])),
                "cmd_vx": float(np.asarray(info["command"])[0]),
                "cmd_vy": float(np.asarray(info["command"])[1]),
                "cmd_yaw": float(np.asarray(info["command"])[2]),
                "reward_env": float(to_scalar(state.reward)),
                "total_dist": float(metrics.get("total_dist", 0.0)),
            }
            row.update(terms)
            step_rows.append(row)
            traj_reward_sum += float(to_scalar(state.reward))
            executed_steps = t + 1
            traj_done = done_flag
            if done_flag and stop_on_done:
                break
        traj_row = {
            "regime": regime.name,
            "traj_id": local_traj_id,
            "cmd_vx": float(np.asarray(cmd)[0]),
            "cmd_vy": float(np.asarray(cmd)[1]),
            "cmd_yaw": float(np.asarray(cmd)[2]),
            "n_steps_executed": executed_steps,
            "done": int(traj_done),
            "mean_env_reward": traj_reward_sum / max(executed_steps, 1),
        }
        for k, v in last_terms.items():
            traj_row[f"last_{k}"] = v
        traj_rows.append(traj_row)
        if progress_every > 0 and ((local_traj_id + 1) % progress_every == 0 or local_traj_id == n_traj - 1):
            traj_dt = time.perf_counter() - traj_t0
            print(f"[{regime.name}] collected {local_traj_id + 1}/{n_traj} trajectories (last traj {traj_dt:.2f}s, steps {executed_steps}, done={traj_done})")
    return pd.DataFrame(step_rows), pd.DataFrame(traj_rows)


def _fit_alpha_from_positive_margins(series: pd.Series, quantile: float = 0.80) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    pos = s[s > 0.0]
    if len(pos) == 0:
        fallback = float(np.quantile(np.abs(s.to_numpy()), quantile)) if len(s) else 0.1
        return max(fallback / ATANH_TARGET, MIN_ALPHA)
    q = float(np.quantile(pos.to_numpy(), quantile))
    return max(q / ATANH_TARGET, MIN_ALPHA)


def _mean_abs_shaped(series: pd.Series, alpha: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if s.size == 0:
        return 1e-6
    return float(np.mean(np.abs(np.tanh(s / alpha)))) + 1e-6


def fit_grouped_reward_params(step_df: pd.DataFrame, regime_col: str = "regime") -> Dict[str, object]:
    components = {
        "alpha_safe": "rho_safety",
        "alpha_track": "rho_tracking",
        "alpha_timing": "rho_timing",
        "alpha_pattern": "rho_pattern",
    }
    pooled_alphas = {name: _fit_alpha_from_positive_margins(step_df[col]) for name, col in components.items()}
    mean_abs = {
        "safe": _mean_abs_shaped(step_df["rho_safety"], pooled_alphas["alpha_safe"]),
        "track": _mean_abs_shaped(step_df["rho_tracking"], pooled_alphas["alpha_track"]),
        "timing": _mean_abs_shaped(step_df["rho_timing"], pooled_alphas["alpha_timing"]),
        "pattern": _mean_abs_shaped(step_df["rho_pattern"], pooled_alphas["alpha_pattern"]),
    }
    pooled_weights = {
        "w_safe": 1.0,
        "w_track": mean_abs["safe"] / mean_abs["track"],
        "w_timing": mean_abs["safe"] / mean_abs["timing"],
        "w_pattern": mean_abs["safe"] / mean_abs["pattern"],
    }
    per_regime: Dict[str, Dict[str, Dict[str, float]]] = {}
    for regime_name, g in step_df.groupby(regime_col):
        alphas = {name: _fit_alpha_from_positive_margins(g[col]) for name, col in components.items()}
        means = {
            "safe": _mean_abs_shaped(g["rho_safety"], alphas["alpha_safe"]),
            "track": _mean_abs_shaped(g["rho_tracking"], alphas["alpha_track"]),
            "timing": _mean_abs_shaped(g["rho_timing"], alphas["alpha_timing"]),
            "pattern": _mean_abs_shaped(g["rho_pattern"], alphas["alpha_pattern"]),
        }
        weights = {
            "w_safe": 1.0,
            "w_track": means["safe"] / means["track"],
            "w_timing": means["safe"] / means["timing"],
            "w_pattern": means["safe"] / means["pattern"],
        }
        per_regime[regime_name] = {"alphas": alphas, "weights": weights, "mean_abs_shaped": means, "n_rows": int(len(g))}
    return {
        "pooled": {"alphas": pooled_alphas, "weights": pooled_weights, "mean_abs_shaped": mean_abs, "n_rows": int(len(step_df))},
        "per_regime": per_regime,
        "notes": {
            "alpha_rule": "alpha = q80(positive rho) / atanh(0.8)",
            "weight_rule": "w_group = mean_abs_shaped_safe / mean_abs_shaped_group, with w_safe fixed to 1.0",
            "group_design": "Matches current reward_step grouped reward: safety / tracking / timing / pattern.",
            "warning_track_identifiability": "If cmd_vy and cmd_yaw are always zero, rho_tracking is dominated by vx tracking. Set --vy-max / --yaw-max > 0 if you want more balanced tracking identification.",
        },
    }


def write_human_readable_summary(summary: Dict[str, object], out_path: str) -> None:
    pooled = summary["pooled"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Suggested coeff_config.py grouped values (pooled across regimes)")
        f.write("============================================================")
        f.write("[pooled alphas]")
        for k, v in pooled["alphas"].items():
            f.write(f"{k} = {v:.6f}")
        f.write("[pooled weights]")
        for k, v in pooled["weights"].items():
            f.write(f"{k} = {v:.6f}")
        f.write("# Notes")
        for k, v in summary["notes"].items():
            f.write(f"- {k}: {v}")
        f.write("# Per-regime suggestions")
        for regime, vals in summary["per_regime"].items():
            f.write(f"[{regime}]")
            for k, v in vals["alphas"].items():
                f.write(f"{k} = {v:.6f}")
            for k, v in vals["weights"].items():
                f.write(f"{k} = {v:.6f}")


def _format_coeff_arrays(summary: Dict[str, object]) -> Dict[str, List[float]]:
    order = ["slow", "mid", "high"]
    out = {"w_safe_by_mode": [], "w_track_by_mode": [], "w_timing_by_mode": [], "w_pattern_by_mode": [], "alpha_safe_by_mode": [], "alpha_track_by_mode": [], "alpha_timing_by_mode": [], "alpha_pattern_by_mode": []}
    for regime in order:
        vals = summary["per_regime"][regime]
        out["w_safe_by_mode"].append(float(vals["weights"]["w_safe"]))
        out["w_track_by_mode"].append(float(vals["weights"]["w_track"]))
        out["w_timing_by_mode"].append(float(vals["weights"]["w_timing"]))
        out["w_pattern_by_mode"].append(float(vals["weights"]["w_pattern"]))
        out["alpha_safe_by_mode"].append(float(vals["alphas"]["alpha_safe"]))
        out["alpha_track_by_mode"].append(float(vals["alphas"]["alpha_track"]))
        out["alpha_timing_by_mode"].append(float(vals["alphas"]["alpha_timing"]))
        out["alpha_pattern_by_mode"].append(float(vals["alphas"]["alpha_pattern"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-ckpt", required=True)
    ap.add_argument("--mid-ckpt", required=True)
    ap.add_argument("--high-ckpt", required=True)
    ap.add_argument("--n-traj-per-regime", type=int, default=40)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./reward_tuning_out")
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--kick-vel", type=float, default=0.0)
    ap.add_argument("--vy-max", type=float, default=0.0)
    ap.add_argument("--yaw-max", type=float, default=0.0)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--stop-on-done", dest="stop_on_done", action="store_true")
    ap.add_argument("--no-stop-on-done", dest="stop_on_done", action="store_false")
    ap.set_defaults(stop_on_done=True)
    ap.add_argument("--slow-vx-min", type=float, default=0.20)
    ap.add_argument("--slow-vx-max", type=float, default=0.78)
    ap.add_argument("--mid-vx-min", type=float, default=0.78)
    ap.add_argument("--mid-vx-max", type=float, default=1.72)
    ap.add_argument("--high-vx-min", type=float, default=1.72)
    ap.add_argument("--high-vx-max", type=float, default=2.00)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    regimes = [RegimeSpec("slow", args.slow_ckpt, args.slow_vx_min, args.slow_vx_max, MODE_WALK), RegimeSpec("mid", args.mid_ckpt, args.mid_vx_min, args.mid_vx_max, MODE_TROT), RegimeSpec("high", args.high_ckpt, args.high_vx_min, args.high_vx_max, MODE_BOUND)]
    all_step_dfs = []
    all_traj_dfs = []
    global_t0 = time.perf_counter()
    for idx, regime in enumerate(regimes):
        print(f"=== Collecting {regime.name} regime ===")
        step_df, traj_df = collect_regime_rollouts(regime=regime, n_traj=args.n_traj_per_regime, n_steps=args.T, seed=args.seed + 1000 * idx, out_vy_max=args.vy_max, out_yaw_max=args.yaw_max, obs_noise=args.obs_noise, kick_vel=args.kick_vel, stop_on_done=args.stop_on_done, progress_every=args.progress_every)
        all_step_dfs.append(step_df)
        all_traj_dfs.append(traj_df)
    step_df = pd.concat(all_step_dfs, ignore_index=True)
    traj_df = pd.concat(all_traj_dfs, ignore_index=True)
    steps_csv = os.path.join(args.out_dir, "grouped_rollouts_steps.csv")
    traj_csv = os.path.join(args.out_dir, "grouped_rollouts_trajectories.csv")
    json_path = os.path.join(args.out_dir, "suggested_grouped_reward_params.json")
    txt_path = os.path.join(args.out_dir, "suggested_grouped_reward_params.txt")
    meta_path = os.path.join(args.out_dir, "collection_metadata.json")
    step_df.to_csv(steps_csv, index=False)
    traj_df.to_csv(traj_csv, index=False)
    summary = fit_grouped_reward_params(step_df)
    summary["coeff_config_arrays"] = _format_coeff_arrays(summary)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_human_readable_summary(summary, txt_path)
    meta = {"n_traj_per_regime": args.n_traj_per_regime, "T": args.T, "seed": args.seed, "obs_noise": args.obs_noise, "kick_vel": args.kick_vel, "vy_max": args.vy_max, "yaw_max": args.yaw_max, "stop_on_done": args.stop_on_done, "total_wallclock_sec": time.perf_counter() - global_t0, "regimes": [regime.__dict__ for regime in regimes], "outputs": {"steps_csv": steps_csv, "traj_csv": traj_csv, "json": json_path, "txt": txt_path}}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved step dataset to: {steps_csv}")
    print(f"Saved trajectory dataset to: {traj_csv}")
    print(f"Saved fitted params to: {json_path}")
    print(f"Saved readable summary to: {txt_path}")
    pooled = summary["pooled"]
    print("Suggested pooled grouped coeff_config.py values:")
    for k, v in pooled["alphas"].items():
        print(f"  {k} = {v:.6f}")
    for k, v in pooled["weights"].items():
        print(f"  {k} = {v:.6f}")
    print("Suggested per-mode arrays for coeff_config.py:")
    for k, vals in summary["coeff_config_arrays"].items():
        print(f"  {k} = {vals}")

if __name__ == "__main__":
    main()
