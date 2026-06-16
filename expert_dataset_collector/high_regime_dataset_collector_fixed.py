#!/usr/bin/env python3
"""Collect a dataset only for the high-speed regime from a dedicated checkpoint.

This version is robust to two Barkour observation layouts:
  1) current obs with mode one-hot appended (34 dims per frame, 510 total)
  2) legacy obs without mode flag         (31 dims per frame, 465 total)

Why this matters:
Some older checkpoints were trained before the mode one-hot was added. In that
case, restoring the checkpoint works, but the first inference call fails with a
shape mismatch such as (510,) vs (465,). This script auto-falls back to the
legacy observation layout if that happens.

Outputs under --out-dir:
  - high_rollouts_steps.csv
  - high_rollouts_trajectories.csv
  - high_collection_metadata.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from typing import Any, Dict, List, Tuple

import jax
from jax import numpy as jp
import numpy as np
import pandas as pd
from brax import math
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

from Barkour import BarkourEnv
from coeff_config import MODE_BOUND
from stl_reward import reward_step

os.environ.setdefault("MUJOCO_GL", "egl")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
    _xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = _xla_flags


class LegacyObsBarkourEnv(BarkourEnv):
    """Barkour env with the pre-mode-flag observation layout.

    Old checkpoints may expect 31 features per frame instead of 34 because the
    3-d mode one-hot vector was not part of the observation when they were
    trained.
    """

    def _get_obs(
        self,
        pipeline_state,
        state_info: Dict[str, Any],
        obs_history,
    ):
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate([
            jp.array([local_rpyrate[2]]) * 0.25,
            math.rotate(jp.array([0, 0, -1]), inv_torso_rot),
            state_info["command"] * jp.array([2.0, 2.0, 0.25]),
            pipeline_state.q[7:] - self._default_pose,
            state_info["last_act"],
        ])

        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )
        obs = jp.roll(obs_history, obs.shape[0]).at[:obs.shape[0]].set(obs)
        return obs

    def reset(self, rng):
        state = super().reset(rng)
        obs_history = jp.zeros(15 * 31, dtype=state.obs.dtype)
        obs = self._get_obs(state.pipeline_state, state.info, obs_history)
        return state.replace(obs=obs)



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



def load_policy(env, ckpt_path: str):
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



def sample_command(
    rng: jax.Array,
    vx_min: float,
    vx_max: float,
    vy_max: float,
    yaw_max: float,
) -> Tuple[jax.Array, jp.ndarray]:
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



def reset_eval_state_with_command_and_mode(
    env,
    rng: jax.Array,
    command: jp.ndarray,
    forced_mode: int = MODE_BOUND,
):
    state = env.reset(rng)
    info = dict(state.info)
    info["command"] = command
    info["mode"] = jp.array(forced_mode, dtype=jp.int32)
    info["history_len"] = jp.array(0, dtype=jp.int32)
    obs_history = jp.zeros_like(state.obs)
    obs = env._get_obs(state.pipeline_state, info, obs_history)
    return state.replace(obs=obs, info=info)



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
    vals = _compute_reward_tuple_jit(
        state.info,
        state.info["command"],
        state.info["mode"],
        state.info["history_len"],
    )
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
        rho_all4,
        rho_bound_event,
        pitch,
        roll,
        FL,
        HL,
        FR,
        HR,
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
        "rho_all4": float(to_scalar(rho_all4)),
        "rho_bound_event": float(to_scalar(rho_bound_event)),
        "pitch": float(to_scalar(pitch)),
        "roll": float(to_scalar(roll)),
        "FL": float(to_scalar(FL)),
        "HL": float(to_scalar(HL)),
        "FR": float(to_scalar(FR)),
        "HR": float(to_scalar(HR)),
    }



def _local_velocities(state) -> Tuple[float, float, float]:
    world_lin_vel = state.pipeline_state.xd.vel[0]
    inv_rot = math.quat_inv(state.pipeline_state.x.rot[0])
    local_vel = math.rotate(world_lin_vel, inv_rot)
    return tuple(float(np.asarray(local_vel[i])) for i in range(3))



def _current_contacts_from_state(state) -> Tuple[int, int, int, int, int]:
    c = np.asarray(state.info["contact_history"][-1]).astype(np.int32)
    fl, hl, fr, hr = int(c[0]), int(c[1]), int(c[2]), int(c[3])
    return fl, hl, fr, hr, fl + hl + fr + hr



def warmup_compilation(
    eval_env,
    policy_fn,
    rng: jax.Array,
    vx_min: float,
    vx_max: float,
    vy_max: float,
    yaw_max: float,
):
    rng, k_reset = jax.random.split(rng)
    rng, cmd = sample_command(rng, vx_min, vx_max, vy_max, yaw_max)
    state = reset_eval_state_with_command_and_mode(eval_env, k_reset, cmd, MODE_BOUND)
    rng, act_rng = jax.random.split(rng)
    _ = policy_fn(state.obs, act_rng)
    _ = compute_reward_terms(state)
    return rng



def _build_env_pair(env_cls, ckpt_path, obs_noise, kick_vel, vx_min, vx_max, vy_max, yaw_max):
    env_for_policy = env_cls(obs_noise=obs_noise, kick_vel=kick_vel)
    jit_inference_fn = load_policy(env_for_policy, ckpt_path)
    eval_env = env_cls(obs_noise=obs_noise, kick_vel=kick_vel)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(0)
    rng = warmup_compilation(eval_env, jit_inference_fn, rng, vx_min, vx_max, vy_max, yaw_max)
    return env_for_policy, eval_env, jit_inference_fn, jit_step



def build_compatible_envs(ckpt_path, obs_noise, kick_vel, vx_min, vx_max, vy_max, yaw_max):
    try:
        _, eval_env, jit_inference_fn, jit_step = _build_env_pair(
            BarkourEnv, ckpt_path, obs_noise, kick_vel, vx_min, vx_max, vy_max, yaw_max
        )
        obs_layout = "with_mode_flag"
        return eval_env, jit_inference_fn, jit_step, obs_layout
    except TypeError as e:
        msg = str(e)
        if "incompatible shapes for broadcasting" not in msg:
            raise
        print("[high] current observation layout failed; retrying legacy 465-dim observation layout...")
        _, eval_env, jit_inference_fn, jit_step = _build_env_pair(
            LegacyObsBarkourEnv, ckpt_path, obs_noise, kick_vel, vx_min, vx_max, vy_max, yaw_max
        )
        obs_layout = "legacy_no_mode_flag"
        return eval_env, jit_inference_fn, jit_step, obs_layout



def collect_high_rollouts(
    ckpt_path: str,
    n_traj: int,
    n_steps: int,
    seed: int,
    vx_min: float,
    vx_max: float,
    vy_max: float,
    yaw_max: float,
    obs_noise: float,
    kick_vel: float,
    stop_on_done: bool,
    progress_every: int,
):
    load_t0 = time.perf_counter()
    eval_env, jit_inference_fn, jit_step, obs_layout = build_compatible_envs(
        ckpt_path, obs_noise, kick_vel, vx_min, vx_max, vy_max, yaw_max
    )
    rng = jax.random.PRNGKey(seed)
    load_dt = time.perf_counter() - load_t0
    print(f"[high] policy restore + warmup took {load_dt:.1f}s ({obs_layout})")

    step_rows: List[Dict[str, float]] = []
    traj_rows: List[Dict[str, float]] = []

    for traj_id in range(n_traj):
        traj_t0 = time.perf_counter()
        rng, k_reset = jax.random.split(rng)
        rng, cmd = sample_command(rng, vx_min, vx_max, vy_max, yaw_max)
        state = reset_eval_state_with_command_and_mode(eval_env, k_reset, cmd, MODE_BOUND)

        traj_reward_sum = 0.0
        executed_steps = 0
        traj_done = False
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
            local_vx, local_vy, local_vz = _local_velocities(state)
            c_fl, c_hl, c_fr, c_hr, n_contacts = _current_contacts_from_state(state)

            row = {
                "regime": "high",
                "traj_id": traj_id,
                "step": t,
                "done": int(done_flag),
                "mode_from_info": int(to_scalar(info["mode"])),
                "cmd_vx": float(np.asarray(info["command"])[0]),
                "cmd_vy": float(np.asarray(info["command"])[1]),
                "cmd_yaw": float(np.asarray(info["command"])[2]),
                "reward_env": float(to_scalar(state.reward)),
                "total_dist": float(metrics.get("total_dist", 0.0)),
                "local_vx": local_vx,
                "local_vy": local_vy,
                "local_vz": local_vz,
                "contact_FL": c_fl,
                "contact_HL": c_hl,
                "contact_FR": c_fr,
                "contact_HR": c_hr,
                "n_contacts": n_contacts,
            }
            row.update(terms)
            step_rows.append(row)

            traj_reward_sum += float(to_scalar(state.reward))
            executed_steps = t + 1
            traj_done = done_flag
            if done_flag and stop_on_done:
                break

        traj_row = {
            "regime": "high",
            "traj_id": traj_id,
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

        if progress_every > 0 and ((traj_id + 1) % progress_every == 0 or traj_id == n_traj - 1):
            traj_dt = time.perf_counter() - traj_t0
            print(
                f"[high] collected {traj_id + 1}/{n_traj} trajectories "
                f"(last traj {traj_dt:.2f}s, steps {executed_steps}, done={traj_done})"
            )

    return pd.DataFrame(step_rows), pd.DataFrame(traj_rows), obs_layout



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--high-ckpt", required=True)
    ap.add_argument("--n-traj", type=int, default=40)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./high_regime_dataset")
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--kick-vel", type=float, default=0.0)
    ap.add_argument("--vy-max", type=float, default=0.0)
    ap.add_argument("--yaw-max", type=float, default=0.0)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--stop-on-done", dest="stop_on_done", action="store_true")
    ap.add_argument("--no-stop-on-done", dest="stop_on_done", action="store_false")
    ap.set_defaults(stop_on_done=True)
    ap.add_argument("--high-vx-min", type=float, default=1.45)
    ap.add_argument("--high-vx-max", type=float, default=1.90)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global_t0 = time.perf_counter()

    step_df, traj_df, obs_layout = collect_high_rollouts(
        ckpt_path=args.high_ckpt,
        n_traj=args.n_traj,
        n_steps=args.T,
        seed=args.seed,
        vx_min=args.high_vx_min,
        vx_max=args.high_vx_max,
        vy_max=args.vy_max,
        yaw_max=args.yaw_max,
        obs_noise=args.obs_noise,
        kick_vel=args.kick_vel,
        stop_on_done=args.stop_on_done,
        progress_every=args.progress_every,
    )

    steps_csv = os.path.join(args.out_dir, "high_rollouts_steps.csv")
    traj_csv = os.path.join(args.out_dir, "high_rollouts_trajectories.csv")
    meta_path = os.path.join(args.out_dir, "high_collection_metadata.json")

    step_df.to_csv(steps_csv, index=False)
    traj_df.to_csv(traj_csv, index=False)

    meta = {
        "n_traj": args.n_traj,
        "T": args.T,
        "seed": args.seed,
        "obs_noise": args.obs_noise,
        "kick_vel": args.kick_vel,
        "vy_max": args.vy_max,
        "yaw_max": args.yaw_max,
        "stop_on_done": args.stop_on_done,
        "high_vx_min": args.high_vx_min,
        "high_vx_max": args.high_vx_max,
        "forced_mode": int(MODE_BOUND),
        "obs_layout": obs_layout,
        "total_wallclock_sec": time.perf_counter() - global_t0,
        "outputs": {
            "steps_csv": steps_csv,
            "traj_csv": traj_csv,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved step dataset to: {steps_csv}")
    print(f"Saved trajectory dataset to: {traj_csv}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
