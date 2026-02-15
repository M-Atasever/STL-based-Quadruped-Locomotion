#%%
from etils import epath
import functools
from datetime import datetime, date
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import pandas as pd
import json, pathlib

import jax
from orbax import checkpoint as ocp
from flax.training import orbax_utils

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from Barkour import BarkourEnv

os.environ['MUJOCO_GL']='egl'  # Configure MuJoCo to use the EGL rendering backend (requires GPU)

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def domain_randomize(sys, rng):
  """Randomizes the mjx.Model."""
  @jax.vmap
  def rand(rng):
    _, key = jax.random.split(rng, 2)
    # friction
    friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
    friction = sys.geom_friction.at[:, 0].set(friction)
    # actuator
    _, key = jax.random.split(key, 2)
    gain_range = (-5, 5)
    param = jax.random.uniform(
        key, (1,), minval=gain_range[0], maxval=gain_range[1]
    ) + sys.actuator_gainprm[:, 0]
    gain = sys.actuator_gainprm.at[:, 0].set(param)
    bias = sys.actuator_biasprm.at[:, 1].set(-param)
    return friction, gain, bias

  friction, gain, bias = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_friction': 0,
      'actuator_gainprm': 0,
      'actuator_biasprm': 0,
  })

  sys = sys.tree_replace({
      'geom_friction': friction,
      'actuator_gainprm': gain,
      'actuator_biasprm': bias,
  })

  return sys, in_axes



today = date.today().strftime("%Y_%m_%d")  
path_ = f"/home/matasever/projects/Quadrupeds_STLReward/models/ckpts/{today}"
ckpt_path = epath.Path(path_)
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
  # save checkpoints
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f'{current_step}'
  orbax_checkpointer.save(path, params, force=True, save_args=save_args) 


make_networks_factory = functools.partial(ppo_networks.make_ppo_networks,
                                          policy_hidden_layer_sizes=(128, 128, 128, 128))

train_fn = functools.partial(
      ppo.train, num_timesteps=200_000_000, num_evals=10,
      reward_scaling=1, episode_length=1000, normalize_observations=True,
      action_repeat=1, unroll_length=20, num_minibatches=32,
      num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
      entropy_cost=1e-2, num_envs=8192, batch_size=256,
      network_factory=make_networks_factory,
      randomization_fn=domain_randomize,
      policy_params_fn=policy_params_fn,
      seed=0)

df_metrics = pd.DataFrame(columns=[
            "num_steps",
            "training_entropy_loss",
            "training_policy_loss",
            "training_total_loss",
            "training_v_loss",
            "eval_episode_Ang_vel_track",
           # "eval_episode_CoM_stab",
           # "eval_episode_CoP_stab",
            "eval_episode_Vel_track_x",
            "eval_episode_Vel_track_y",
           # "eval_episode_Zmp_stab",
            "eval_episode_smooth_action",
           # "eval_episode_autow_com",
           # "eval_episode_autow_cone",
           # "eval_episode_autow_cop",
           # "eval_episode_autow_nlegs",
           # "eval_episode_autow_torque",
           # "eval_episode_autow_zmp",
            "eval_episode_combined_safety",
            #"eval_episode_friction_cone",
            "eval_episode_more_legs_grounded",
            "eval_episode_reward",
           # "eval_episode_stl_penalty",
            "eval_episode_torque_lim",
            "eval_episode_total_dist",
            "eval_episode_x_error",
            "eval_episode_y_error",
            "eval_episode_yaw_error",
            "eval_avg_episode_length",])


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 1000, -100

def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])

  temp_metrics = []
  temp_metrics.append(num_steps)
  try:
    temp_metrics.append(metrics["training/entropy_loss"])
    temp_metrics.append(metrics["training/policy_loss"])
    temp_metrics.append(metrics["training/total_loss"])
    temp_metrics.append(metrics["training/v_loss"])
  except:
    temp_metrics += [0, 0, 0, 0]
  temp_metrics.append(metrics["eval/episode_Ang_vel_track"])
#  temp_metrics.append(metrics["eval/episode_CoM_stab"])
#  temp_metrics.append(metrics["eval/episode_CoP_stab"])
  temp_metrics.append(metrics["eval/episode_Vel_track_x"])
  temp_metrics.append(metrics["eval/episode_Vel_track_y"])
#  temp_metrics.append(metrics["eval/episode_Zmp_stab"])
  temp_metrics.append(metrics["eval/episode_smooth_action"])
#  temp_metrics.append(metrics["eval/episode_autow_com"])
#  temp_metrics.append(metrics["eval/episode_autow_cone"])
#  temp_metrics.append(metrics["eval/episode_autow_cop"])
#  temp_metrics.append(metrics["eval/episode_autow_nlegs"])
#  temp_metrics.append(metrics["eval/episode_autow_torque"])
#  temp_metrics.append(metrics["eval/episode_autow_zmp"])
  temp_metrics.append(metrics["eval/episode_combined_safety"])
#  temp_metrics.append(metrics["eval/episode_friction_cone"])
  temp_metrics.append(metrics["eval/episode_more_legs_grounded"])
  temp_metrics.append(metrics["eval/episode_reward"])
#  temp_metrics.append(metrics["eval/episode_stl_penalty"])
  temp_metrics.append(metrics["eval/episode_torque_lim"])
  temp_metrics.append(metrics["eval/episode_total_dist"])
  temp_metrics.append(metrics["eval/episode_x_error"])
  temp_metrics.append(metrics["eval/episode_y_error"])
  temp_metrics.append(metrics["eval/episode_yaw_error"])
  temp_metrics.append(metrics["eval/avg_episode_length"])
  
  df_metrics.loc[len(df_metrics)] = temp_metrics
  
  temp_metrics = []
  
  if num_steps >= 199999990:  # 999999999
     df_metrics.to_csv("metricssaved.csv", sep="|")

  #plt.autoscale(enable=True, axis="both", tight=True)
  
  plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
  plt.ylim([min_y, max_y])

  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.title(f'y={y_data[-1]:.3f}')

  plt.errorbar(
      x_data, y_data, yerr=ydataerr)
  plt.show()

# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = BarkourEnv() # envs.get_environment(env_name)
eval_env = BarkourEnv()   # envs.get_environment(env_name)
make_inference_fn, params, final_metrics = train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')


today = date.today().strftime("%Y_%m_%d")  
model_path = f"/home/matasever/projects/Quadrupeds_STLReward/models/{today}"
#model_path = '/home/matasever/projects/Quadrupeds_STLReward/models/barkour_params_stl1'
model.save_params(model_path, params)

#path = pathlib.Path(f"/home/matasever/projects/Quadrupeds_STLReward/models/{today}metrics.npz")
#np.savez(path, **{k: np.asarray(v) for k, v in final_metrics.items()})
    
# %%
