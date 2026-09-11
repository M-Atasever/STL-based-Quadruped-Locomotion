#!/usr/bin/env python3
"""Evaluate a trained Barkour policy on fixed velocity commands.

This script matches the PPO checkpoint-loading style used in the user's
previous `testing.py` file. It runs 20 tests of 500 timesteps for each
commanded forward velocity by default and reports:
  1) cost of transportation (CoT)
  2) survive rate
  3) success rate
  4) action variation (AV)

Action variation definition:
  AV = mean_t mean_j (a[t, j] - a[t-1, j])^2
  computed over the post-warmup portion of each rollout.
  Lower AV indicates smoother control commands.

Success definition:
  - ignore the first `warmup_steps` timesteps
  - compute the average local forward speed after warmup
  - success = survive and avg_speed in [0.85 * vx_cmd, 1.15 * vx_cmd]

CoT definition:
  CoT = total mechanical work / (m g d)
  where work = sum_t sum_j |tau_j * qdot_j| dt over the 12 actuated joints.

Example:
  python test.py --ckpt-path /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import jax
from jax import numpy as jp
import numpy as np
from brax import math
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from Barkour import BarkourEnv
from coeff_config import (
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    TROT_TO_BOUND_ENTER,
    WALK_TO_TROT_ENTER,
)

os.environ.setdefault('MUJOCO_GL', 'egl')
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

np.set_printoptions(precision=3, suppress=True, linewidth=120)

GRAVITY = 9.81
DEFAULT_COMMANDS = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 1.9, 2.0, 2.1]


def _install_running_stats_compat_patch():
    try:
        from brax.training.acme import running_statistics
    except Exception:
        return lambda: None

    cls = running_statistics.RunningStatisticsState
    dataclass_fields = getattr(cls, '__dataclass_fields__', {})
    if 'std_eps' in dataclass_fields:
        return lambda: None

    orig_init = cls.__init__

    def patched_init(self, *args, std_eps=None, **kwargs):
        return orig_init(self, *args, **kwargs)

    cls.__init__ = patched_init

    def restore():
        cls.__init__ = orig_init

    return restore


def load_policy(env: BarkourEnv, ckpt_path: str):
    """Loads a PPO policy exactly in the style of the old testing script."""
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
    """Resets the environment and rebuilds observation with the desired command.

    This matches the helper used in the old testing file so the observation is
    consistent with the forced command from the very first action.
    """
    state = env.reset(rng)

    info = dict(state.info)
    info['command'] = command
    info['mode'] = _mode_from_command(command)
    info['history_len'] = jp.array(0, dtype=jp.int32)

    obs_history = jp.zeros_like(state.obs)
    obs = env._get_obs(state.pipeline_state, info, obs_history)

    return state.replace(obs=obs, info=info)


def get_robot_mass(env: BarkourEnv, override: float | None) -> float:
    if override is not None:
        return float(override)
    body_mass = np.asarray(env.sys.mj_model.body_mass, dtype=np.float64)
    if body_mass.size <= 1:
        return float(body_mass.sum())
    return float(body_mass[1:].sum())


def get_local_forward_speed(state) -> float:
    world_lin_vel = state.pipeline_state.xd.vel[0]
    local_vel = math.rotate(world_lin_vel, math.quat_inv(state.pipeline_state.x.rot[0]))
    return float(local_vel[0])


def get_xy_position(state, torso_idx: int) -> np.ndarray:
    return np.asarray(jax.device_get(state.pipeline_state.x.pos[torso_idx - 1, :2]), dtype=np.float64)


def get_actuated_joint_power(state) -> float:
    """Mechanical power over the 12 actuated joints.

    For a floating-base quadruped in MuJoCo/Brax, qd[0:6] are base DOFs and
    qd[6:] correspond to actuated joints. qfrc_actuator has matching nv layout.
    """
    tau = np.asarray(jax.device_get(state.pipeline_state.qfrc_actuator), dtype=np.float64)
    qd = np.asarray(jax.device_get(state.pipeline_state.qd), dtype=np.float64)
    n = min(len(tau), len(qd))
    tau = tau[:n]
    qd = qd[:n]
    return float(np.sum(np.abs(tau[6:] * qd[6:])))


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(np.mean(arr)),
        'var': float(np.var(arr)),
        'std': float(np.std(arr)),
    }


def rollout_once(
    env: BarkourEnv,
    jit_inference_fn,
    jit_step,
    rng: jax.Array,
    command: jp.ndarray,
    horizon: int,
    warmup_steps: int,
    robot_mass: float,
) -> Dict[str, float]:
    state = reset_eval_state_with_command(env, rng, command)

    xy_prev = get_xy_position(state, env._torso_idx)
    planar_distance = 0.0
    total_energy = 0.0
    post_warm_local_vx: List[float] = []
    post_warm_actions: List[np.ndarray] = []
    survived = 1.0
    terminated_step = horizon

    for t in range(horizon):
        rng, act_rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)

        # Store raw policy actions for steady-state action-variation evaluation.
        # We only use the post-warmup interval, matching the steady-state
        # evaluation window used for velocity tracking.
        if t >= warmup_steps:
            post_warm_actions.append(
                np.asarray(jax.device_get(ctrl), dtype=np.float64)
            )

        state = jit_step(state, ctrl)

        total_energy += get_actuated_joint_power(state) * float(env.dt)

        xy_curr = get_xy_position(state, env._torso_idx)
        planar_distance += float(np.linalg.norm(xy_curr - xy_prev))
        xy_prev = xy_curr

        local_vx = get_local_forward_speed(state)
        if t >= warmup_steps:
            post_warm_local_vx.append(local_vx)

        if float(state.done) != 0.0:
            survived = 0.0
            terminated_step = t + 1
            break

    avg_speed_post_warm = float(np.mean(post_warm_local_vx)) if post_warm_local_vx else float('nan')
    vx_cmd = float(command[0])
    tol = 0.15 * abs(vx_cmd)
    success = 1.0 if (survived == 1.0 and abs(avg_speed_post_warm - vx_cmd) <= tol) else 0.0

    cot = total_energy / max(robot_mass * GRAVITY * planar_distance, 1e-8)

    # Mean squared first difference of consecutive actions, averaged over
    # time and action dimensions. Equivalent to
    # (1 / ((T-1) * d)) * sum_t ||a_t - a_{t-1}||_2^2.
    if len(post_warm_actions) >= 2:
        action_array = np.stack(post_warm_actions, axis=0)
        action_diffs = np.diff(action_array, axis=0)
        action_variation = float(np.mean(np.square(action_diffs)))
    else:
        action_variation = float('nan')

    return {
        'vx_cmd': float(command[0]),
        'vy_cmd': float(command[1]),
        'yaw_cmd': float(command[2]),
        'executed_steps': float(terminated_step),
        'planar_distance': float(planar_distance),
        'avg_speed_post_warm': float(avg_speed_post_warm),
        'cot': float(cot),
        'action_variation': float(action_variation),
        'survive': float(survived),
        'success': float(success),
    }


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(summary_rows: List[Dict[str, float]]) -> None:
    print('\nEvaluation summary')
    print('-' * 162)
    print(
        '{:>6} | {:>10} {:>10} {:>10} | {:>12} {:>12} {:>12} | '
        '{:>12} {:>12} {:>12} | {:>12} {:>12} {:>12}'.format(
            'vx',
            'CoT mean', 'CoT var', 'CoT std',
            'AV mean', 'AV var', 'AV std',
            'Survive mean', 'Survive var', 'Survive std',
            'Success mean', 'Success var', 'Success std',
        )
    )
    print('-' * 162)
    for row in summary_rows:
        print(
            '{:6.2f} | {:10.4f} {:10.4f} {:10.4f} | '
            '{:12.6f} {:12.6f} {:12.6f} | '
            '{:12.4f} {:12.4f} {:12.4f} | '
            '{:12.4f} {:12.4f} {:12.4f}'.format(
                row['vx_cmd'],
                row['cot_mean'], row['cot_var'], row['cot_std'],
                row['action_variation_mean'], row['action_variation_var'], row['action_variation_std'],
                row['survive_mean'], row['survive_var'], row['survive_std'],
                row['success_mean'], row['success_var'], row['success_std'],
            )
        )
    print('-' * 162)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='PPO checkpoint path')
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vy-cmd', type=float, default=0.0)
    parser.add_argument('--yaw-cmd', type=float, default=0.0)
    parser.add_argument('--velocities', type=float, nargs='+', default=DEFAULT_COMMANDS)
    parser.add_argument('--robot-mass', type=float, default=None)
    parser.add_argument('--obs-noise', type=float, default=0.0)
    parser.add_argument('--kick-vel', type=float, default=0.0)
    parser.add_argument('--rollouts-csv', type=str, default='eval_rollouts.csv')
    parser.add_argument('--summary-csv', type=str, default='eval_summary.csv')
    parser.add_argument('--summary-json', type=str, default='eval_summary.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = BarkourEnv(obs_noise=args.obs_noise, kick_vel=args.kick_vel)
    robot_mass = get_robot_mass(env, args.robot_mass)
    jit_inference_fn = load_policy(env, args.ckpt_path)

    eval_env = BarkourEnv(obs_noise=args.obs_noise, kick_vel=args.kick_vel)
    jit_step = jax.jit(eval_env.step)

    print(f'Checkpoint: {args.ckpt_path}')
    print(f'Robot mass used for CoT: {robot_mass:.4f} kg')
    print(f'Commands: {args.velocities}')
    print(f'Tests per command: {args.num_tests}')
    print(f'Horizon: {args.horizon} steps | Warmup: {args.warmup_steps} steps')

    rng = jax.random.PRNGKey(args.seed)
    rollout_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    summary_json: Dict[str, Any] = {}

    for vx_cmd in args.velocities:
        command = jp.array([vx_cmd, args.vy_cmd, args.yaw_cmd], dtype=jp.float32)
        cmd_rows: List[Dict[str, float]] = []

        for test_idx in range(args.num_tests):
            rng, rrng = jax.random.split(rng)
            row = rollout_once(
                env=eval_env,
                jit_inference_fn=jit_inference_fn,
                jit_step=jit_step,
                rng=rrng,
                command=command,
                horizon=args.horizon,
                warmup_steps=args.warmup_steps,
                robot_mass=robot_mass,
            )
            row['test_idx'] = float(test_idx)
            rollout_rows.append(row)
            cmd_rows.append(row)

        cot_stats = summarize([r['cot'] for r in cmd_rows])
        action_variation_stats = summarize([r['action_variation'] for r in cmd_rows])
        survive_stats = summarize([r['survive'] for r in cmd_rows])
        success_stats = summarize([r['success'] for r in cmd_rows])

        summary_row = {
            'vx_cmd': float(vx_cmd),
            'cot_mean': cot_stats['mean'],
            'cot_var': cot_stats['var'],
            'cot_std': cot_stats['std'],
            'action_variation_mean': action_variation_stats['mean'],
            'action_variation_var': action_variation_stats['var'],
            'action_variation_std': action_variation_stats['std'],
            'survive_mean': survive_stats['mean'],
            'survive_var': survive_stats['var'],
            'survive_std': survive_stats['std'],
            'success_mean': success_stats['mean'],
            'success_var': success_stats['var'],
            'success_std': success_stats['std'],
        }
        summary_rows.append(summary_row)
        summary_json[f'vx_{vx_cmd:.2f}'] = summary_row

    save_csv(Path(args.rollouts_csv), rollout_rows)
    save_csv(Path(args.summary_csv), summary_rows)
    Path(args.summary_json).write_text(json.dumps(summary_json, indent=2))

    print_summary_table(summary_rows)
    print(f'\nSaved per-test results to: {Path(args.rollouts_csv).resolve()}')
    print(f'Saved summary CSV to:      {Path(args.summary_csv).resolve()}')
    print(f'Saved summary JSON to:     {Path(args.summary_json).resolve()}')


if __name__ == '__main__':
    main()
