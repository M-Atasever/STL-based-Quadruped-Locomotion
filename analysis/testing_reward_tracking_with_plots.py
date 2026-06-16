from brax.io import html
import functools
import os
from pathlib import Path

import jax
from jax import numpy as jp
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML
from brax import math
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

os.environ['MUJOCO_GL'] = 'egl'

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

np.set_printoptions(precision=3, suppress=True, linewidth=120)

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
local_model = True
brax_renderer = False

ckpt_path = '/home/matasever/projects/Quadrupeds_STLReward/bound5/ckpts/2026_04_26/334233600'

x_vel = 1.8
y_vel = 0.0
ang_vel = 0.0
n_steps = 500
render_every = 2
print_every = 50
warmup_steps = 50
save_csv = True
csv_path = 'testing_reward_components.csv'
plot_dir = 'testing_reward_plots'
save_plots = True

# Components to track from state.metrics. The script will only log keys that are
# actually present, so it stays robust if your BarkourEnv exposes a subset.
tracked_metric_keys = [
    'total_stl_reward',
    'rho_safety', 'rho_walk', 'rho_trot', 'rho_bound',
    'rho_tracking', 'rho_timing', 'rho_pattern', 'gait_shape',
    'rho_diag2', 'rho_stride', 'rho_duty', 'rho_3plus_event',
    'rho_support', 'rho_diag_phase', 'rho_p2',
    'rho_front', 'rho_hind', 'rho_hindfront',
    'rho_flight', 'rho_front_only', 'rho_hind_only',
    'rho_all4', 'rho_bound_event',
    'Vel_track_x', 'Vel_track_y', 'Ang_vel_track',
    'x_error', 'y_error', 'yaw_error',
    'torque_lim', 'more_legs_grounded', 'smooth_action',
    'rho_comz', 'rho_roll', 'rho_pitch', 'rho_slip',
    'pitch', 'roll', 'FL', 'HL', 'FR', 'HR',
]

# Plot groups. Only available columns will be plotted.
plot_groups = {
    'overview': ['total_stl_reward', 'reward_env', 'local_vx'],
    'grouped_rhos': ['rho_safety', 'rho_tracking', 'rho_timing', 'rho_pattern'],
    'walk_trot_shape': ['rho_diag2', 'rho_stride', 'rho_duty', 'rho_diag_phase', 'rho_p2', 'rho_3plus_event'],
    'bound_shape': ['rho_front', 'rho_hind', 'rho_hindfront', 'rho_flight', 'rho_front_only', 'rho_hind_only', 'rho_all4', 'rho_bound_event'],
    'tracking_errors': ['x_error', 'y_error', 'yaw_error', 'Vel_track_x', 'Vel_track_y', 'Ang_vel_track'],
    'safety_terms': ['rho_comz', 'rho_roll', 'rho_pitch', 'rho_slip', 'torque_lim', 'more_legs_grounded', 'rho_support'],
    'contact_signals': ['FL', 'HL', 'FR', 'HR'],
}


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
    """Reset env and rebuild observation with a forced command."""
    state = env.reset(rng)

    info = dict(state.info)
    info['command'] = command
    info['mode'] = _mode_from_command(command)
    info['history_len'] = jp.array(0, dtype=jp.int32)

    obs_history = jp.zeros_like(state.obs)
    obs = env._get_obs(state.pipeline_state, info, obs_history)

    return state.replace(obs=obs, info=info)


def _to_float(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return float(arr)
    return arr


def _extract_metric_row(state, step_idx: int, cmd_x: float):
    metrics = dict(state.metrics)
    row = {
        'step': step_idx,
        'command_vx': float(cmd_x),
        'reward_env': _to_float(state.reward),
        'done': int(_to_float(state.done)),
        'mode': int(_to_float(state.info['mode'])),
        'history_len': int(_to_float(state.info['history_len'])),
    }

    world_lin_vel = state.pipeline_state.xd.vel[0]
    local_vel = math.rotate(world_lin_vel, math.quat_inv(state.pipeline_state.x.rot[0]))
    row['local_vx'] = _to_float(local_vel[0])
    row['local_vy'] = _to_float(local_vel[1])
    row['local_vz'] = _to_float(local_vel[2])

    for key in tracked_metric_keys:
        if key in metrics:
            row[key] = _to_float(metrics[key])

    return row


def _print_component_snapshot(row):
    snapshot_keys = [
        'local_vx', 'total_stl_reward',
        'rho_tracking', 'rho_timing', 'rho_pattern',
        'rho_diag2', 'rho_stride', 'rho_p2',
        'rho_flight', 'rho_front_only', 'rho_hind_only',
        'rho_front', 'rho_hind', 'rho_hindfront', 'rho_all4',
    ]
    printable = []
    for key in snapshot_keys:
        if key in row:
            printable.append(f"{key}={row[key]:.3f}")
    print(' | '.join(printable))


def _print_post_warmup_summary(df: pd.DataFrame, warmup_steps: int):
    if df.empty:
        print('No rollout data collected.')
        return

    post = df[df['step'] >= warmup_steps].copy()
    if post.empty:
        print(f'No post-warmup rows (warmup_steps={warmup_steps}).')
        return

    summary_keys = [
        'local_vx', 'total_stl_reward',
        'rho_safety', 'rho_tracking', 'rho_timing', 'rho_pattern',
        'rho_diag2', 'rho_stride', 'rho_duty', 'rho_3plus_event',
        'rho_diag_phase', 'rho_p2',
        'rho_front', 'rho_hind', 'rho_hindfront',
        'rho_flight', 'rho_front_only', 'rho_hind_only', 'rho_all4', 'rho_bound_event',
        'x_error', 'y_error', 'yaw_error',
    ]

    available = [k for k in summary_keys if k in post.columns]
    if not available:
        print('No tracked summary columns found in DataFrame.')
        return

    print('\nPost-warmup summary (mean ± std):')
    for key in available:
        mean = post[key].mean()
        std = post[key].std(ddof=0)
        print(f'  {key:18s}: {mean: .4f} ± {std:.4f}')


def _plot_group(df: pd.DataFrame, cols, title: str, out_path: Path, warmup_steps: int):
    available = [c for c in cols if c in df.columns]
    if not available:
        return False

    plt.figure(figsize=(10, 4.5))
    for col in available:
        plt.plot(df['step'], df[col], label=col)

    plt.axvline(warmup_steps, linestyle='--', linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _make_summary_bar_plot(df: pd.DataFrame, warmup_steps: int, out_path: Path):
    post = df[df['step'] >= warmup_steps].copy()
    summary_keys = [
        'rho_safety', 'rho_tracking', 'rho_timing', 'rho_pattern',
        'rho_diag2', 'rho_stride', 'rho_p2', 'rho_flight',
        'rho_front_only', 'rho_hind_only', 'rho_front', 'rho_hind',
        'rho_hindfront', 'rho_all4', 'rho_bound_event',
    ]
    available = [k for k in summary_keys if k in post.columns]
    if not available:
        return False

    means = post[available].mean()
    stds = post[available].std(ddof=0)

    plt.figure(figsize=(11, 5))
    x = np.arange(len(available))
    plt.bar(x, means.values, yerr=stds.values, capsize=3)
    plt.xticks(x, available, rotation=45, ha='right')
    plt.ylabel('Post-warmup mean ± std')
    plt.title('Reward-component summary after warmup')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _save_plots(df: pd.DataFrame, plot_dir: str, warmup_steps: int):
    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, cols in plot_groups.items():
        out_path = out_dir / f'{name}.png'
        ok = _plot_group(df, cols, title=name.replace('_', ' ').title(), out_path=out_path, warmup_steps=warmup_steps)
        if ok:
            saved.append(out_path)

    bar_path = out_dir / 'post_warmup_summary_bar.png'
    if _make_summary_bar_plot(df, warmup_steps=warmup_steps, out_path=bar_path):
        saved.append(bar_path)

    return saved


# -----------------------------------------------------------------------------
# Load policy
# -----------------------------------------------------------------------------
env = BarkourEnv()
if local_model:
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
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
else:
    jit_inference_fn = load_policy(env, ckpt_path)


eval_env = BarkourEnv()
jit_step = jax.jit(eval_env.step)

the_command = jp.array([x_vel, y_vel, ang_vel], dtype=jp.float32)

rng = jax.random.PRNGKey(0)
state = reset_eval_state_with_command(eval_env, rng, the_command)

rollout = [state.pipeline_state]
tracked_rows = []

for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)

    row = _extract_metric_row(state, i, x_vel)
    tracked_rows.append(row)

    if i % print_every == 0:
        print(f"Step {i:3d}: command_vx={x_vel:.2f}")
        _print_component_snapshot(row)

    rollout.append(state.pipeline_state)

# Save and summarize tracked components
tracked_df = pd.DataFrame(tracked_rows)
if save_csv:
    out_path = Path(csv_path)
    tracked_df.to_csv(out_path, index=False)
    print(f"\nSaved reward-component trace to: {out_path.resolve()}")

_print_post_warmup_summary(tracked_df, warmup_steps=warmup_steps)

if save_plots:
    saved_plots = _save_plots(tracked_df, plot_dir=plot_dir, warmup_steps=warmup_steps)
    if saved_plots:
        print('\nSaved plots:')
        for p in saved_plots:
            print(f'  {p.resolve()}')
    else:
        print('\nNo plots were saved because none of the requested columns were present.')

# Render video
media.show_video(
    eval_env.render(rollout[::render_every], camera='track'),
    fps=1.0 / eval_env.dt / render_every,
)

if brax_renderer:
    HTML(html.render(eval_env.sys.tree_replace({'opt.timestep': eval_env.dt}), rollout))
