from brax.io import model, html
import jax
from brax import envs
from jax import numpy as jp
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
import functools

import mediapy as media
import matplotlib.pyplot as plt
from IPython.display import HTML

import os
import numpy as np

from Barkour import BarkourEnv

os.environ['MUJOCO_GL']='egl'  # Configure MuJoCo to use the EGL rendering backend (requires GPU)

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

brax_renderer = False
model_path = '/home/matasever/projects/Quadrupeds_STLReward/models/2026_02_02'
ckpt_path = '/home/matasever/projects/Quadrupeds_STLReward/models/ckpts/2026_02_02/178257920'

env = BarkourEnv() 
make_inference_fn, params, _= ppo.train(environment=env, 
                                        num_timesteps=0, 
                                        episode_length=1000,
                                        normalize_observations=True,
                                        restore_checkpoint_path=ckpt_path,
                                        network_factory=functools.partial(
                                        ppo_networks.make_ppo_networks,
                                        policy_hidden_layer_sizes=(128, 128, 128, 128)),
                                        )
#params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

eval_env = BarkourEnv() 

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# @markdown Commands **only used for Barkour Env**:
x_vel = 1.5 #1.0  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
ang_vel = 0.0  #@param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info['command'] = the_command
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 500
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

media.show_video(
    eval_env.render(rollout[::render_every], camera='track'),
    fps=1.0 / eval_env.dt / render_every)

if brax_renderer:
    HTML(html.render(eval_env.sys.tree_replace({'opt.timestep': eval_env.dt}), rollout))

