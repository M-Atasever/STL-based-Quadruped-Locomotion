from brax.io import html
import functools
import os

import jax
from jax import numpy as jp
import mediapy as media
import numpy as np
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

np.set_printoptions(precision=3, suppress=True, linewidth=100)

local_model = True
brax_renderer = False

#ckpt_path = '/home/matasever/projects/Quadrupeds_STLReward/models_to_generate_datasets/original_bound_model/400000000'
ckpt_path = '/home/matasever/projects/Quadrupeds_STLReward/models10/ckpts/2026_05_28/311377920' 

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
    """Reset the environment and rebuild observation with the desired command.

    The original testing script overwrote state.info['command'] after reset but kept
    the stale observation built from the randomly sampled reset command. This helper
    makes the command, mode, history length, and observation mutually consistent.
    """
    state = env.reset(rng)

    info = dict(state.info)
    info['command'] = command
    info['mode'] = _mode_from_command(command)
    info['history_len'] = jp.array(0, dtype=jp.int32)

    # reset observation history so the very first action matches the requested command
    obs_history = jp.zeros_like(state.obs)
    obs = env._get_obs(state.pipeline_state, info, obs_history)

    return state.replace(obs=obs, info=info)


env = BarkourEnv()
# Load trained policy.
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

# Commands only used for BarkourEnv evaluation.
x_vel = 1.2
y_vel = 0.0
ang_vel = 0.0
the_command = jp.array([x_vel, y_vel, ang_vel], dtype=jp.float32)

# Initialize state consistently with the desired command.
rng = jax.random.PRNGKey(0)
state = reset_eval_state_with_command(eval_env, rng, the_command)

rollout = [state.pipeline_state]

n_steps = 500
render_every = 2

"""for _ in range(n_steps):
    rng, act_rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)"""
    
for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)

  # 1. Extract linear velocity of the torso (body 0)
  # world_lin_vel is [vx, vy, vz] in the world frame
  world_lin_vel = state.pipeline_state.xd.vel[0]

  # 2. Calculate local velocity (relative to the robot's torso rotation)
  # This uses the rotation (x.rot[0]) to transform world velocity to local frame
  local_vel = math.rotate(world_lin_vel, math.quat_inv(state.pipeline_state.x.rot[0]))
  local_x_vel = local_vel[0] # Local forward velocity

  # 3. Print the value (printing every 50 steps to avoid slowing down simulation)
  if i % 50 == 0:
    print(f"Step {i:3}: Command x_vel={x_vel:.2f}, Actual local_x_vel={local_x_vel:.2f}")

  rollout.append(state.pipeline_state)

media.show_video(
    eval_env.render(rollout[::render_every], camera='track'),
    fps=1.0 / eval_env.dt / render_every,
)

if brax_renderer:
    HTML(html.render(eval_env.sys.tree_replace({'opt.timestep': eval_env.dt}), rollout))
