# %%
from etils import epath
import functools
from datetime import datetime, date
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
from orbax import checkpoint as ocp
from flax.training import orbax_utils
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

from Barkour import BarkourEnv

os.environ['MUJOCO_GL'] = 'egl'

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

np.set_printoptions(precision=3, suppress=True, linewidth=100)


def domain_randomize(sys, rng):
    """Randomizes a subset of the MJX model parameters."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)

        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)

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
    in_axes = in_axes.tree_replace(
        {
            'geom_friction': 0,
            'actuator_gainprm': 0,
            'actuator_biasprm': 0,
        }
    )

    sys = sys.tree_replace(
        {
            'geom_friction': friction,
            'actuator_gainprm': gain,
            'actuator_biasprm': bias,
        }
    )

    return sys, in_axes


today = date.today().strftime('%Y_%m_%d')
path_ = f'/home/matasever/projects/Quadrupeds_STLReward/models8/ckpts/{today}'
ckpt_path = epath.Path(path_)
ckpt_path.mkdir(parents=True, exist_ok=True)


def policy_params_fn(current_step, make_policy, params):
    """Checkpoint PPO parameters during training."""
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f'{current_step}'
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 128, 128, 128),
)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=400_000_000,
    num_evals=10,
    reward_scaling=1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=30,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.955,
    learning_rate=0.00015,
    entropy_cost=0.004,
    num_envs=8192,
    batch_size=256,
    network_factory=make_networks_factory,
    randomization_fn=domain_randomize,
    policy_params_fn=policy_params_fn,
    seed=0,
)

metric_columns = [
    'num_steps',
    'training_entropy_loss',
    'training_policy_loss',
    'training_total_loss',
    'training_v_loss',
    'eval_episode_total_stl_reward',
    'eval_episode_rho_diag2',
    'eval_episode_rho_stride',
    'eval_episode_rho_duty',
    'eval_episode_rho_3plus_event',
    'eval_episode_rho_support',
    'eval_episode_rho_diag_phase',
    'eval_episode_rho_p2',
    'eval_episode_rho_tracking',
    'eval_episode_rho_timing',
    'eval_episode_rho_pattern',
    'eval_episode_rho_front',
    'eval_episode_rho_hind',
    'eval_episode_rho_hindfront',
    'eval_episode_rho_flight',
    'eval_episode_rho_front_only',
    'eval_episode_rho_hind_only',
    'eval_episode_Ang_vel_track',
    'eval_episode_Vel_track_x',
    'eval_episode_Vel_track_y',
    'eval_episode_gait_shape',
    'eval_episode_rho_comz',
    'eval_episode_rho_roll',
    'eval_episode_rho_pitch',
    'eval_episode_rho_slip',
    'eval_episode_rho_bound',
    'eval_episode_rho_trot',
    'eval_episode_rho_walk',
    'eval_episode_rho_safety',
    'eval_episode_more_legs_grounded',
    'eval_episode_reward',
    'eval_episode_torque_lim',
    'eval_episode_total_dist',
    'eval_episode_x_error',
    'eval_episode_y_error',
    'eval_episode_yaw_error',
    'eval_avg_episode_length',
]
            
df_metrics = pd.DataFrame(columns=metric_columns)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 60, -50


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    row = [num_steps]
    row.append(metrics.get('training/entropy_loss', 0.0))
    row.append(metrics.get('training/policy_loss', 0.0))
    row.append(metrics.get('training/total_loss', 0.0))
    row.append(metrics.get('training/v_loss', 0.0))

    row.append(metrics.get('eval/episode_total_stl_reward', np.nan))
    row.append(metrics.get('eval/episode_rho_diag2', np.nan))
    row.append(metrics.get('eval/episode_rho_stride', np.nan))
    row.append(metrics.get('eval/episode_rho_duty', np.nan))
    row.append(metrics.get('eval/episode_rho_3plus_event', np.nan))
    row.append(metrics.get('eval/episode_rho_support', np.nan))
    row.append(metrics.get("eval/episode_rho_diag_phase", np.nan))
    row.append(metrics.get("eval/episode_rho_p2", np.nan))  
    row.append(metrics.get("eval/episode_rho_tracking", np.nan))  
    row.append(metrics.get("eval/episode_rho_timing", np.nan))  
    row.append(metrics.get("eval/episode_rho_pattern", np.nan))  
    row.append(metrics.get("eval/episode_rho_front", np.nan))  
    row.append(metrics.get("eval/episode_rho_hind", np.nan))  
    row.append(metrics.get("eval/episode_rho_hindfront", np.nan))  
    row.append(metrics.get("eval/episode_rho_flight", np.nan))  
    row.append(metrics.get("eval/episode_rho_front_only", np.nan))  
    row.append(metrics.get("eval/episode_rho_hind_only", np.nan))  
    row.append(metrics.get('eval/episode_Ang_vel_track', np.nan))
    row.append(metrics.get('eval/episode_Vel_track_x', np.nan))
    row.append(metrics.get('eval/episode_Vel_track_y', np.nan))
    row.append(metrics.get('eval/episode_gait_shape', np.nan))
    row.append(metrics.get('eval/episode_rho_comz', np.nan))
    row.append(metrics.get('eval/episode_rho_roll', np.nan))
    row.append(metrics.get('eval/episode_rho_pitch', np.nan))
    row.append(metrics.get('eval/episode_rho_slip', np.nan))
    row.append(metrics.get('eval/episode_rho_bound', np.nan))
    row.append(metrics.get('eval/episode_rho_trot', np.nan))
    row.append(metrics.get('eval/episode_rho_walk', np.nan))
    row.append(metrics.get('eval/episode_rho_safety', np.nan))
    row.append(metrics.get('eval/episode_more_legs_grounded', np.nan))
    row.append(metrics.get('eval/episode_reward', np.nan))
    row.append(metrics.get('eval/episode_torque_lim', np.nan))
    row.append(metrics.get('eval/episode_total_dist', np.nan))
    row.append(metrics.get('eval/episode_x_error', np.nan))
    row.append(metrics.get('eval/episode_y_error', np.nan))
    row.append(metrics.get('eval/episode_yaw_error', np.nan))
    row.append(metrics.get('eval/avg_episode_length', np.nan))

    df_metrics.loc[len(df_metrics)] = row

    if num_steps >= 19_999_999:
        df_metrics.to_csv('metricssaved.csv', sep='|', index=False)

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')
    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.show()


# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = BarkourEnv()
eval_env = BarkourEnv()
make_inference_fn, params, final_metrics = train_fn(
    environment=env,
    progress_fn=progress,
    eval_env=eval_env,
)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

today = date.today().strftime('%Y_%m_%d')
model_path = f'/home/matasever/projects/Quadrupeds_STLReward/models8/{today}'
model.save_params(model_path, params)
