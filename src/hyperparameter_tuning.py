import os
# Make sure EGL is set for headless GPU runs
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import optuna
import jax
import jax.numpy as jnp
from brax.training.agents.ppo import train as ppo
from Barkour import BarkourEnv
from brax.training.agents.ppo import networks as ppo_networks
import functools
import numpy as np
import traceback

make_networks_factory = functools.partial(ppo_networks.make_ppo_networks,
                                          policy_hidden_layer_sizes=(128, 128, 128, 128))

train_fn = functools.partial(
      ppo.train, num_timesteps=100_000_000, num_evals=10,
      reward_scaling=1, episode_length=1000, normalize_observations=True,
      action_repeat=1, num_minibatches=32,
      num_updates_per_batch=4, num_envs=512, batch_size=256,
      network_factory=make_networks_factory,
      seed=0)


def objective(trial: optuna.Trial):
    # --- 1. Reward Weight Suggestions ---
    """raw = np.array([
            trial.suggest_float("w_vx_raw",     1e-2, 3.0, log=True),
            trial.suggest_float("w_tau_raw",    1e-3, 1.0, log=True),
            trial.suggest_float("w_gait_raw",   1e-2, 3.0, log=True),
        ], dtype=np.float64)

    raw = raw / raw.sum()"""
    
    """reward_weights = {
            "w_vx": trial.suggest_float("w_vx",     1e-2, 5.0, log=True),  # float(raw[0]),
           # "w_tau": trial.suggest_float("w_tau",    1e-3, 1.0, log=True),  # float(raw[1]),
           # "w_gait": trial.suggest_float("w_gait",   1e-2, 3.0, log=True),  # float(raw[2]),
            "w_vy": trial.suggest_float("w_vy",     1e-2, 1.0, log=True),
            "w_yaw": trial.suggest_float("w_yaw",     1e-2, 1.0, log=True),
            
            "alpha_b": trial.suggest_categorical('alpha_b', [5.0, 20.0, 200.0]),
            "alpha_vx": trial.suggest_categorical('alpha_vx', [0.5, 1.0, 5.0, 10.0, 20.0]),
            "alpha_vy": trial.suggest_categorical('alpha_vy', [0.5, 1.0, 5.0, 10.0, 20.0]),
            "alpha_yaw": trial.suggest_categorical('alpha_yaw', [0.5, 1.0, 5.0, 10.0, 20.0]),
            #"alpha_tau": trial.suggest_categorical('alpha_tau', [50.0, 100.0, 300.0]),
            #"alpha_gait": trial.suggest_categorical('alpha_gait', [1.0, 10.0, 50.0]),
            
            "beta": trial.suggest_categorical('beta', [0.5, 2.0, 5.0, 10.0]),
        }"""
        
    # --- 2. PPO Hyperparameter Suggestions ---
    ppo_params = {
        'learning_rate':    trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'entropy_cost':     trial.suggest_float('entropy_cost', 1e-4, 1e-2, log=True),
        'discounting':      trial.suggest_float('discounting', 0.95, 0.99),
        'unroll_length':    trial.suggest_categorical('unroll_length', [10, 20, 30]),
        #'num_minibatches':  trial.suggest_int('num_minibatches', 16, 48, step=16),
      #  'batch_size':       256, # Keep constant to manage VRAM
    }

    # --- 3. Environment Setup ---
    # pass weights to the env so stl_reward can use them
    env = BarkourEnv(obs_noise = 0.01, kick_vel = 0.01) # BarkourEnv(reward_weights=reward_weights)
    eval_env = BarkourEnv(obs_noise = 0.01, kick_vel = 0.01) # BarkourEnv(reward_weights=reward_weights)
    
    eval_idx = 0

    # --- 4. Pruning Callback ---
    def progress(num_steps, metrics):
        # if ppo calls progress_fn at each eval, use an internal counter
        nonlocal eval_idx
        reward = float(metrics.get("eval/episode_reward", -1e9))
        trial.report(reward, step=eval_idx)
        eval_idx += 1
        if trial.should_prune():
            raise optuna.TrialPruned()

    # --- 5. Training Execution ---
    try:
        _, _, final_metrics = train_fn(
            environment=env,
            eval_env=eval_env,
            learning_rate=ppo_params['learning_rate'],
            entropy_cost=ppo_params['entropy_cost'],
            discounting=ppo_params['discounting'],
            unroll_length=ppo_params['unroll_length'],
          #  num_minibatches=ppo_params['num_minibatches'],
            progress_fn=progress
        )

        # Return a composite metric: Reward + tracking_score + total_dist
        
        if "eval/episode_x_error" not in final_metrics:
            raise RuntimeError(f"Missing x_error; available keys: {list(final_metrics.keys())[:30]}")
        else:
            x_vel_error = final_metrics.get("eval/episode_x_error", 10)
            tracking_score = float(np.exp(-float(x_vel_error)))
        
        reward_ = float(final_metrics['eval/episode_reward'])
        total_dist = float(final_metrics['eval/episode_total_dist'])
        score = tracking_score * total_dist
        
        trial.set_user_attr("avg_reward", reward_)
        trial.set_user_attr("avg_total_dist", total_dist)
        trial.set_user_attr("avg_tracking_score", tracking_score)
        
        return reward_
    
    except optuna.TrialPruned:
        raise  # Let Optuna handle the pruning
    except Exception as e:
        traceback.print_exc()
        trial.set_user_attr("exception", str(e))
        return -10000.0

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(
        direction="maximize",
        study_name="stl_reward_weights",
        sampler=sampler,
        load_if_exists=True,
        pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=45,   
                    n_warmup_steps=5,      
                    interval_steps=1)
    )

    study.optimize(objective, n_trials=70)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  attrs:", study.best_trial.user_attrs)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
        