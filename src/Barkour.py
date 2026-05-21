import os
import sys
import numpy as np
from typing import Any, Dict, Sequence, List
from etils import epath


import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

from brax import base
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model

REPO_ROOT = epath.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'configs'
if CONFIG_DIR.as_posix() not in sys.path:
  sys.path.insert(0, CONFIG_DIR.as_posix())

from reward_config import get_config, get_stl_config
from stl_reward import reward_step 
from bound_experiments import get_experiment_config
from coeff_config import (
    H,
    MODE_WALK, MODE_TROT, MODE_BOUND,
    WALK_TO_TROT_ENTER, TROT_TO_WALK_EXIT,
    TROT_TO_BOUND_ENTER, BOUND_TO_TROT_EXIT,
    w_pattern_by_mode,
    alpha_pattern_by_mode,
    cmd_vx_range,
    cmd_vy_range,
    cmd_yaw_range,
)

os.environ.setdefault('MUJOCO_GL', 'egl')  # Local Mac notebooks can set this to glfw before import.

# Tell XLA to use Triton GEMM on non-macOS GPU runs. This is not a Metal/MPS flag.
if sys.platform != 'darwin':
  xla_flags = os.environ.get('XLA_FLAGS', '')
  if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
  os.environ['XLA_FLAGS'] = xla_flags

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


BARKOUR_ROOT_PATH = REPO_ROOT / 'mujoco_menagerie/google_barkour_vb'
STL_REWARD = True

class BarkourEnv(PipelineEnv):
  """Environment for training the barkour quadruped joystick policy in MJX."""

  def __init__(
      self,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.05,
      scene_file: str = 'scene_mjx.xml',
      reward_weights=None,
      experiment: str | None = None,
      experiment_config: Dict[str, Any] | None = None,
      **kwargs,
  ):
    path = BARKOUR_ROOT_PATH / scene_file
    sys = mjcf.load(path.as_posix())
    self._dt = 0.02  # this environment is 50 fps
    sys = sys.tree_replace({'opt.timestep': 0.004})

    # override menagerie params for smoother policy
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[6:].set(0.5239),
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend='mjx', n_frames=n_frames)

    if STL_REWARD:
      self.reward_config = get_stl_config()
      self.reward_input = None
      self._H = H  # Max STL history length; active horizon is mode-dependent in reward_step.
    else:
      self.reward_config = get_config()
      # set custom from kwargs
      for k, v in kwargs.items():
        if k.endswith('_scale'):
          self.reward_config.rewards.scales[k[:-6]] = v

    self._torso_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso'
    )
    if reward_weights is None:
      self.reward_weights = None
    else:
      required_reward_keys = (
          "w_track", "w_safe", "w_timing", "w_pattern",
          "alpha_track", "alpha_safe", "alpha_timing", "alpha_pattern",
          "beta",
      )
      if not all(k in reward_weights for k in required_reward_keys):
        raise ValueError(
            "reward_weights must provide the grouped STL keys: "
            "w_track, w_safe, w_timing, w_pattern, "
            "alpha_track, alpha_safe, alpha_timing, alpha_pattern, beta"
        )
      self.reward_weights = jp.array(
          [reward_weights[k] for k in required_reward_keys],
          dtype=jp.float32,
      )
        
    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
    self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
    self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
    self.uppers = jp.array([0.52, 2.1, 2.1] * 4)
    feet_site = [
        'foot_front_left',
        'foot_hind_left',
        'foot_front_right',
        'foot_hind_right',
    ]
    feet_site_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
    self._feet_site_id = np.array(feet_site_id)
    lower_leg_body = [
        'lower_leg_front_left',
        'lower_leg_hind_left',
        'lower_leg_front_right',
        'lower_leg_hind_right',
    ]
    lower_leg_body_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
        for l in lower_leg_body
    ]
    assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
    self._lower_leg_body_id = np.array(lower_leg_body_id)
    self._foot_radius = 0.0175
    self._nv = sys.nv
    self.experiment_config = (
        experiment_config if experiment_config is not None else get_experiment_config(experiment)
    )
    self.experiment_name = (
        self.experiment_config["name"] if self.experiment_config is not None else "default"
    )
    self._sampler_config = (
        self.experiment_config.get("sampler", {"type": "default"})
        if self.experiment_config is not None
        else {"type": "default"}
    )
    self._bound_reward_config = (
        self.experiment_config.get("bound_reward", {})
        if self.experiment_config is not None
        else {}
    )
    
  
  def _update_mode_hysteresis(self, mode: jax.Array, command: jax.Array) -> jax.Array:
    vx = jp.abs(command[0])

    mode = jp.where((mode == MODE_WALK) & (vx >= WALK_TO_TROT_ENTER), jp.array(MODE_TROT, dtype=jp.int32), mode)
    mode = jp.where((mode == MODE_TROT) & (vx <= TROT_TO_WALK_EXIT),  jp.array(MODE_WALK, dtype=jp.int32), mode)

    mode = jp.where((mode == MODE_TROT) & (vx >= TROT_TO_BOUND_ENTER), jp.array(MODE_BOUND, dtype=jp.int32), mode)
    mode = jp.where((mode == MODE_BOUND) & (vx <= BOUND_TO_TROT_EXIT), jp.array(MODE_TROT, dtype=jp.int32), mode)
    return mode  # self.mode = new_mode
  
  def _mode_from_command(self, command: jax.Array) -> jax.Array:
    """Fresh mode assignment from command speed without hysteresis memory."""
    vx = jp.abs(command[0])
    return jp.where(
        vx >= TROT_TO_BOUND_ENTER,
        jp.array(MODE_BOUND, dtype=jp.int32),
        jp.where(
            vx >= WALK_TO_TROT_ENTER,
            jp.array(MODE_TROT, dtype=jp.int32),
            jp.array(MODE_WALK, dtype=jp.int32),
        ),
    )

  def _bound_soft_transition_values(
      self,
      lin_velocity_history: jax.Array,
      command: jax.Array,
      valid_len: jax.Array,
  ) -> Dict[str, jax.Array]:
    """Soft BOUND reward activation for transition experiments."""
    cfg = self._bound_reward_config
    enabled = bool(cfg.get("enable_bound_soft_transition", False))
    history_len = lin_velocity_history.shape[0]
    active_len = jp.minimum(jp.asarray(valid_len, dtype=jp.int32), history_len)
    idx = jp.arange(history_len)
    mask = idx >= (history_len - active_len)
    mask_f = mask.astype(jp.float32)
    denom = jp.maximum(jp.sum(mask_f), 1.0)
    mean_vx = jp.sum(lin_velocity_history[:, 0] * mask_f) / denom

    if not enabled:
      one = jp.array(1.0, dtype=jp.float32)
      return {
          "bound_soft_command_alpha": one,
          "bound_soft_speed_gate": one,
          "bound_effective_pattern_alpha": one,
          "bound_soft_mean_vx": mean_vx,
      }

    cmd_lo = float(cfg.get("bound_soft_command_alpha_lo", 1.55))
    cmd_hi = float(cfg.get("bound_soft_command_alpha_hi", 1.90))
    command_alpha = jp.clip(
        (jp.abs(command[0]) - cmd_lo) / jp.maximum(cmd_hi - cmd_lo, 1e-6),
        0.0,
        1.0,
    )
    speed_center = float(cfg.get("bound_soft_speed_gate_center", 0.90))
    speed_width = float(cfg.get("bound_soft_speed_gate_width", 0.15))
    speed_gate = jax.nn.sigmoid((mean_vx - speed_center) / jp.maximum(speed_width, 1e-6))
    min_valid = int(cfg.get("bound_soft_min_valid_steps", 8))
    warmup_gate = (valid_len >= min_valid).astype(jp.float32)
    effective_alpha = command_alpha * speed_gate * warmup_gate

    return {
        "bound_soft_command_alpha": command_alpha,
        "bound_soft_speed_gate": speed_gate,
        "bound_effective_pattern_alpha": effective_alpha,
        "bound_soft_mean_vx": mean_vx,
    }

  def _uniform_scalar(self, key: jax.Array, value_range: Sequence[float]) -> jax.Array:
    return jax.random.uniform(
        key,
        shape=(),
        minval=float(value_range[0]),
        maxval=float(value_range[1]),
    )

  def _sample_bound_mixture_command(self, rng: jax.Array) -> jax.Array:
    cfg = self._sampler_config
    rng, key_regime, key_vx, key_vy, key_yaw = jax.random.split(rng, 5)
    probs = jp.array(
        [
            float(cfg["nominal_probability"]),
            float(cfg["transition_probability"]),
            float(cfg["bound_probability"]),
        ],
        dtype=jp.float32,
    )
    probs = probs / jp.sum(probs)
    regime = jax.random.choice(key_regime, 3, shape=(), p=probs)

    vx_nominal = self._uniform_scalar(key_vx, cfg["nominal_vx_range"])
    vx_transition = self._uniform_scalar(key_vx, cfg["transition_vx_range"])
    vx_bound = self._uniform_scalar(key_vx, cfg["bound_vx_range"])
    vx = jp.where(regime == 0, vx_nominal, jp.where(regime == 1, vx_transition, vx_bound))

    vy_nominal = self._uniform_scalar(key_vy, cfg["nominal_vy_range"])
    vy_bound = self._uniform_scalar(key_vy, cfg["bound_vy_range"])
    yaw_nominal = self._uniform_scalar(key_yaw, cfg["nominal_yaw_range"])
    yaw_bound = self._uniform_scalar(key_yaw, cfg["bound_yaw_range"])
    near_bound = regime != 0
    vy = jp.where(near_bound, vy_bound, vy_nominal)
    yaw = jp.where(near_bound, yaw_bound, yaw_nominal)
    return jp.array([vx, vy, yaw], dtype=jp.float32)

  def _sample_bound_curriculum_command(
      self, rng: jax.Array, command_resets: jax.Array
  ) -> jax.Array:
    cfg = self._sampler_config
    _, key_vx, key_vy, key_yaw = jax.random.split(rng, 4)
    reset_count = jp.asarray(command_resets, dtype=jp.int32)
    stage_a_end = jp.asarray(int(cfg["stage_a_reset_count"]), dtype=jp.int32)
    stage_b_end = jp.asarray(int(cfg["stage_b_reset_count"]), dtype=jp.int32)

    vx_a = self._uniform_scalar(key_vx, cfg["stage_a_vx_range"])
    vx_b = self._uniform_scalar(key_vx, cfg["stage_b_vx_range"])
    vx_c = self._uniform_scalar(key_vx, cfg["stage_c_vx_range"])
    vx = jp.where(reset_count < stage_a_end, vx_a, jp.where(reset_count < stage_b_end, vx_b, vx_c))
    vy = self._uniform_scalar(key_vy, cfg["vy_range"])
    yaw = self._uniform_scalar(key_yaw, cfg["yaw_range"])
    return jp.array([vx, vy, yaw], dtype=jp.float32)

  def sample_command(self, rng: jax.Array, command_resets: jax.Array | int = 0) -> jax.Array:
    sampler_type = self._sampler_config.get("type", "default")
    if sampler_type == "bound_mixture":
      return self._sample_bound_mixture_command(rng)
    if sampler_type == "bound_curriculum":
      return self._sample_bound_curriculum_command(rng, command_resets)

    lin_vel_x = [0.0, 1.69]  # min max [m/s]
    lin_vel_y = [-0.2, 0.2]  # min max [m/s]
    ang_vel_yaw = [-0.2, 0.2]  # min max [rad/s]

    _, key1, key2, key3 = jax.random.split(rng, 4)
    lin_vel_x = jax.random.uniform(
        key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
    )
    new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
    return new_cmd

  def _bound_contact_pattern_terms(
      self,
      contact_history: jax.Array,
      lin_velocity_history: jax.Array,
      com_z_history: jax.Array,
      roll_history: jax.Array,
      pitch_history: jax.Array,
      command: jax.Array,
      valid_len: jax.Array,
      mode: jax.Array,
  ) -> Dict[str, jax.Array]:
    """BOUND-only contact pattern shaping terms from [FL, HL, FR, HR]."""
    c = contact_history.astype(jp.float32)
    active_len = jp.minimum(jp.asarray(valid_len, dtype=jp.int32), c.shape[0])
    idx = jp.arange(c.shape[0])
    mask = idx >= (c.shape[0] - active_len)
    mask_f = mask.astype(jp.float32)
    denom = jp.maximum(jp.sum(mask_f), 1.0)

    c_fl, c_hl, c_fr, c_hr = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    front_pair = c_fl * c_fr * (1.0 - c_hl) * (1.0 - c_hr)
    hind_pair = c_hl * c_hr * (1.0 - c_fl) * (1.0 - c_fr)
    diagonal_1 = c_fl * c_hr * (1.0 - c_fr) * (1.0 - c_hl)
    diagonal_2 = c_fr * c_hl * (1.0 - c_fl) * (1.0 - c_hr)
    all_four = c_fl * c_fr * c_hl * c_hr
    any_contact = jp.minimum(c_fl + c_hl + c_fr + c_hr, 1.0)

    def masked_mean(x):
      return jp.sum(x * mask_f) / denom

    front_sync = masked_mean((1.0 - jp.abs(c_fl - c_fr)) * any_contact)
    hind_sync = masked_mean((1.0 - jp.abs(c_hl - c_hr)) * any_contact)
    pair_support = masked_mean(front_pair + hind_pair)
    diagonal_penalty = masked_mean(diagonal_1 + diagonal_2)
    all_four_penalty = masked_mean(all_four)
    contact_pattern_reward = pair_support - diagonal_penalty - 0.5 * all_four_penalty

    is_bound = (mode == MODE_BOUND).astype(jp.float32)
    contact_weight = float(self._bound_reward_config.get("bound_contact_pattern_weight", 0.0))
    diagonal_weight = float(self._bound_reward_config.get("bound_diagonal_penalty_weight", 0.0))
    all_four_weight = float(self._bound_reward_config.get("bound_all_four_penalty_weight", 0.0))
    pair_sync_weight = float(self._bound_reward_config.get("bound_pair_sync_weight", 0.0))
    raw_bonus = is_bound * (
        contact_weight * pair_support
        - diagonal_weight * diagonal_penalty
        - all_four_weight * all_four_penalty
        + pair_sync_weight * 0.5 * (front_sync + hind_sync) * pair_support
    )
    gate_enabled = bool(self._bound_reward_config.get("enable_tracking_gate", False))
    mean_vx = masked_mean(lin_velocity_history[:, 0])
    if gate_enabled:
      vx_error = jp.abs(mean_vx - command[0])
      vx_full = float(self._bound_reward_config.get("gate_vx_error_full", 0.55))
      vx_zero = float(self._bound_reward_config.get("gate_vx_error_zero", 1.10))
      vx_gate = jp.clip((vx_zero - vx_error) / jp.maximum(vx_zero - vx_full, 1e-6), 0.0, 1.0)

      min_height = jp.min(jp.where(mask, com_z_history, jp.inf))
      height_full = float(self._bound_reward_config.get("gate_base_height_full", 0.22))
      height_zero = float(self._bound_reward_config.get("gate_base_height_zero", 0.18))
      height_gate = jp.clip(
          (min_height - height_zero) / jp.maximum(height_full - height_zero, 1e-6),
          0.0,
          1.0,
      )

      max_roll_pitch = jp.maximum(
          jp.max(jp.where(mask, jp.abs(roll_history), 0.0)),
          jp.max(jp.where(mask, jp.abs(pitch_history), 0.0)),
      )
      angle_full = float(self._bound_reward_config.get("gate_roll_pitch_full_rad", 0.30))
      angle_zero = float(self._bound_reward_config.get("gate_roll_pitch_zero_rad", 0.45))
      angle_gate = jp.clip(
          (angle_zero - max_roll_pitch) / jp.maximum(angle_zero - angle_full, 1e-6),
          0.0,
          1.0,
      )
      min_valid = int(self._bound_reward_config.get("gate_min_valid_steps", 8))
      warmup_gate = (valid_len >= min_valid).astype(jp.float32)
      stability_gate = height_gate * angle_gate
      contact_gate = warmup_gate * vx_gate * stability_gate
    else:
      vx_gate = jp.array(1.0, dtype=jp.float32)
      stability_gate = jp.array(1.0, dtype=jp.float32)
      contact_gate = jp.array(1.0, dtype=jp.float32)

    soft_terms = self._bound_soft_transition_values(
        lin_velocity_history=lin_velocity_history,
        command=command,
        valid_len=valid_len,
    )
    soft_alpha = soft_terms["bound_effective_pattern_alpha"]
    bonus = raw_bonus * contact_gate * soft_alpha

    return {
        "bound_front_pair_sync": is_bound * front_sync,
        "bound_hind_pair_sync": is_bound * hind_sync,
        "bound_front_or_hind_pair_support": is_bound * pair_support,
        "bound_diagonal_trot_penalty": is_bound * diagonal_penalty,
        "bound_all_four_stance_penalty": is_bound * all_four_penalty,
        "bound_contact_pattern_reward": is_bound * contact_pattern_reward,
        "bound_contact_pattern_bonus": bonus,
        "bound_raw_contact_pattern_bonus": raw_bonus,
        "bound_contact_gate": is_bound * contact_gate,
        "bound_vx_tracking_gate": is_bound * vx_gate,
        "bound_stability_gate": is_bound * stability_gate,
        "bound_soft_command_alpha": is_bound * soft_terms["bound_soft_command_alpha"],
        "bound_soft_speed_gate": is_bound * soft_terms["bound_soft_speed_gate"],
        "bound_effective_pattern_alpha": is_bound * soft_alpha,
        "bound_soft_mean_vx": is_bound * soft_terms["bound_soft_mean_vx"],
    }

  def _bound_transition_guard_terms(
      self,
      contact_history: jax.Array,
      lin_velocity_history: jax.Array,
      com_z_history: jax.Array,
      roll_history: jax.Array,
      pitch_history: jax.Array,
      command: jax.Array,
      valid_len: jax.Array,
      mode: jax.Array,
      rho_pattern: jax.Array,
  ) -> Dict[str, jax.Array]:
    """Opt-in guard against rewarding BOUND pattern while not moving/stable."""
    cfg = self._bound_reward_config
    enabled = bool(cfg.get("enable_bound_transition_guard", False))
    if not enabled:
      zero = jp.array(0.0, dtype=jp.float32)
      return {
          "bound_transition_guard_adjustment": zero,
          "bound_pattern_gate": zero,
          "bound_pattern_gate_penalty": zero,
          "bound_positive_pattern_contribution": zero,
          "bound_tracking_guard_penalty": zero,
          "transition_tracking_guard_penalty": zero,
          "bound_transition_vx_tracking_gate": zero,
          "bound_transition_stability_gate": zero,
          "bound_forward_progress_target": zero,
          "bound_forward_progress_mean_vx": zero,
          "bound_forward_progress_margin": zero,
          "bound_forward_progress_penalty": zero,
          "bound_all_four_stall_penalty": zero,
          "bound_all_four_stall_fraction": zero,
          "bound_soft_progress_reward": zero,
          "bound_soft_progress_score": zero,
          "bound_soft_anti_stall_penalty": zero,
          "bound_soft_anti_stall_margin": zero,
          "bound_all_four_after_warmup_penalty": zero,
          "bound_pair_support_after_warmup_reward": zero,
          "bound_pair_support_shortfall_penalty": zero,
          "bound_pair_support_after_warmup_fraction": zero,
          "bound_diagonal_after_warmup_penalty": zero,
          "bound_diagonal_after_warmup_fraction": zero,
          "bound_pair_balance_after_warmup_penalty": zero,
          "bound_pair_balance_after_warmup_score": zero,
          "bound_balanced_pair_after_warmup_reward": zero,
          "bound_balanced_pair_after_warmup_fraction": zero,
          "bound_hind_pair_shortfall_penalty": zero,
          "bound_front_pair_dominance_penalty": zero,
          "bound_front_pair_dominance_margin": zero,
          "bound_pair_balance_gate": zero,
          "bound_pair_gated_progress_reward": zero,
          "bound_pair_gated_speed_target_reward": zero,
          "bound_pair_gated_speed_shortfall_penalty": zero,
          "bound_pair_gated_speed_target": zero,
          "bound_pair_gated_speed_margin": zero,
      }

    history_len = lin_velocity_history.shape[0]
    active_len = jp.minimum(jp.asarray(valid_len, dtype=jp.int32), history_len)
    idx = jp.arange(history_len)
    mask = idx >= (history_len - active_len)
    mask_f = mask.astype(jp.float32)
    denom = jp.maximum(jp.sum(mask_f), 1.0)

    def masked_mean(x):
      return jp.sum(x * mask_f) / denom

    mean_vx = masked_mean(lin_velocity_history[:, 0])
    vx_error = jp.abs(mean_vx - command[0])
    vx_full = float(cfg.get("gate_vx_error_full", 0.45))
    vx_zero = float(cfg.get("gate_vx_error_zero", 0.95))
    vx_gate = jp.clip((vx_zero - vx_error) / jp.maximum(vx_zero - vx_full, 1e-6), 0.0, 1.0)

    min_height = jp.min(jp.where(mask, com_z_history, jp.inf))
    height_full = float(cfg.get("gate_base_height_full", 0.23))
    height_zero = float(cfg.get("gate_base_height_zero", 0.18))
    height_gate = jp.clip(
        (min_height - height_zero) / jp.maximum(height_full - height_zero, 1e-6),
        0.0,
        1.0,
    )

    max_roll_pitch = jp.maximum(
        jp.max(jp.where(mask, jp.abs(roll_history), 0.0)),
        jp.max(jp.where(mask, jp.abs(pitch_history), 0.0)),
    )
    angle_full = float(cfg.get("gate_roll_pitch_full_rad", 0.28))
    angle_zero = float(cfg.get("gate_roll_pitch_zero_rad", 0.45))
    angle_gate = jp.clip(
        (angle_zero - max_roll_pitch) / jp.maximum(angle_zero - angle_full, 1e-6),
        0.0,
        1.0,
    )
    min_valid = int(cfg.get("gate_min_valid_steps", 8))
    warmup_gate = (valid_len >= min_valid).astype(jp.float32)
    stability_gate = height_gate * angle_gate
    gate = warmup_gate * vx_gate * stability_gate

    abs_vx_cmd = jp.abs(command[0])
    is_bound = (mode == MODE_BOUND).astype(jp.float32)
    transition_min = float(cfg.get("transition_guard_min_vx", 1.45))
    transition_max = float(cfg.get("transition_guard_max_vx", TROT_TO_BOUND_ENTER))
    is_transition = (
        (abs_vx_cmd >= transition_min)
        & (abs_vx_cmd < transition_max)
        & (mode != MODE_BOUND)
    ).astype(jp.float32)

    pattern_alpha = float(cfg.get("bound_pattern_guard_alpha", 0.6))
    positive_pattern = jp.maximum(jp.tanh(rho_pattern / jp.maximum(pattern_alpha, 1e-6)), 0.0)
    pattern_guard_weight = float(cfg.get("bound_pattern_guard_weight", 1.0))
    pattern_penalty = -is_bound * pattern_guard_weight * (1.0 - gate) * positive_pattern

    bound_tracking_weight = float(cfg.get("bound_tracking_guard_penalty_weight", 0.0))
    transition_tracking_weight = float(cfg.get("transition_tracking_guard_penalty_weight", 0.0))
    bound_tracking_penalty = -is_bound * bound_tracking_weight * (1.0 - vx_gate)
    transition_tracking_penalty = -is_transition * transition_tracking_weight * (1.0 - vx_gate)

    forward_guard_enabled = bool(cfg.get("enable_bound_forward_progress_guard", False))
    c = contact_history.astype(jp.float32)
    c_fl, c_hl, c_fr, c_hr = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    front_pair = c_fl * c_fr * (1.0 - c_hl) * (1.0 - c_hr)
    hind_pair = c_hl * c_hr * (1.0 - c_fl) * (1.0 - c_fr)
    diagonal_1 = c_fl * c_hr * (1.0 - c_fr) * (1.0 - c_hl)
    diagonal_2 = c_fr * c_hl * (1.0 - c_fl) * (1.0 - c_hr)
    all_four = c_fl * c_hl * c_fr * c_hr
    front_pair_fraction = masked_mean(front_pair)
    hind_pair_fraction = masked_mean(hind_pair)
    pair_support_fraction = front_pair_fraction + hind_pair_fraction
    diagonal_observed_fraction = masked_mean(diagonal_1 + diagonal_2)
    all_four_observed_fraction = masked_mean(all_four)
    if forward_guard_enabled:
      min_fraction = float(cfg.get("bound_forward_progress_min_fraction", 0.45))
      min_vx = float(cfg.get("bound_forward_progress_min_vx", 0.70))
      target_vx = jp.maximum(min_vx, min_fraction * jp.abs(command[0]))
      forward_margin = mean_vx - target_vx
      forward_shortfall = jp.clip((target_vx - mean_vx) / jp.maximum(target_vx, 1e-6), 0.0, 1.0)
      progress_weight = float(cfg.get("bound_forward_progress_penalty_weight", 0.0))
      forward_progress_penalty = -is_bound * progress_weight * forward_shortfall

      all_four_fraction = all_four_observed_fraction
      all_four_weight = float(cfg.get("bound_all_four_stall_penalty_weight", 0.0))
      all_four_stall_penalty = -is_bound * all_four_weight * forward_shortfall * all_four_fraction
    else:
      target_vx = jp.array(0.0, dtype=jp.float32)
      forward_margin = jp.array(0.0, dtype=jp.float32)
      forward_progress_penalty = jp.array(0.0, dtype=jp.float32)
      all_four_fraction = all_four_observed_fraction
      all_four_stall_penalty = jp.array(0.0, dtype=jp.float32)

    soft_progress_enabled = bool(cfg.get("enable_bound_soft_progress_reward", False))
    if soft_progress_enabled:
      progress_scale = float(cfg.get("bound_soft_progress_scale_vx", 0.60))
      soft_progress_score = jp.tanh(jp.maximum(mean_vx, 0.0) / jp.maximum(progress_scale, 1e-6))
      soft_progress_weight = float(cfg.get("bound_soft_progress_reward_weight", 0.0))
      soft_progress_reward = is_bound * warmup_gate * soft_progress_weight * soft_progress_score

      anti_stall_min_vx = float(cfg.get("bound_soft_anti_stall_min_vx", 0.30))
      anti_stall_margin = mean_vx - anti_stall_min_vx
      anti_stall_shortfall = jp.clip(
          (anti_stall_min_vx - mean_vx) / jp.maximum(anti_stall_min_vx, 1e-6),
          0.0,
          1.0,
      )
      anti_stall_weight = float(cfg.get("bound_soft_anti_stall_penalty_weight", 0.0))
      soft_anti_stall_penalty = -is_bound * warmup_gate * anti_stall_weight * anti_stall_shortfall

      all_four_allowed = float(cfg.get("bound_all_four_after_warmup_allowed_fraction", 0.55))
      all_four_excess = jp.clip(
          (all_four_observed_fraction - all_four_allowed)
          / jp.maximum(1.0 - all_four_allowed, 1e-6),
          0.0,
          1.0,
      )
      all_four_after_weight = float(cfg.get("bound_all_four_after_warmup_penalty_weight", 0.0))
      all_four_after_warmup_penalty = -is_bound * warmup_gate * all_four_after_weight * all_four_excess
    else:
      soft_progress_score = jp.array(0.0, dtype=jp.float32)
      soft_progress_reward = jp.array(0.0, dtype=jp.float32)
      anti_stall_margin = jp.array(0.0, dtype=jp.float32)
      soft_anti_stall_penalty = jp.array(0.0, dtype=jp.float32)
      all_four_after_warmup_penalty = jp.array(0.0, dtype=jp.float32)

    clean_pair_enabled = bool(cfg.get("enable_bound_clean_pair_reward", False))
    if clean_pair_enabled:
      pair_reward_weight = float(cfg.get("bound_pair_support_after_warmup_reward_weight", 0.0))
      pair_support_after_warmup_reward = (
          is_bound * warmup_gate * pair_reward_weight * pair_support_fraction
      )

      min_pair_support = float(cfg.get("bound_min_pair_support_after_warmup", 0.20))
      pair_shortfall = jp.clip(
          (min_pair_support - pair_support_fraction) / jp.maximum(min_pair_support, 1e-6),
          0.0,
          1.0,
      )
      pair_shortfall_weight = float(cfg.get("bound_pair_support_shortfall_penalty_weight", 0.0))
      pair_support_shortfall_penalty = (
          -is_bound * warmup_gate * pair_shortfall_weight * pair_shortfall
      )

      diagonal_allowed = float(cfg.get("bound_diagonal_after_warmup_allowed_fraction", 0.15))
      diagonal_excess = jp.clip(
          (diagonal_observed_fraction - diagonal_allowed)
          / jp.maximum(1.0 - diagonal_allowed, 1e-6),
          0.0,
          1.0,
      )
      diagonal_after_weight = float(cfg.get("bound_diagonal_after_warmup_penalty_weight", 0.0))
      diagonal_after_warmup_penalty = (
          -is_bound * warmup_gate * diagonal_after_weight * diagonal_excess
      )

      pair_balance_score_raw = 1.0 - jp.clip(
          jp.abs(front_pair_fraction - hind_pair_fraction)
          / jp.maximum(pair_support_fraction, 1e-6),
          0.0,
          1.0,
      )
      pair_balance_score = jp.where(pair_support_fraction > 1e-6, pair_balance_score_raw, 0.0)
      min_balance = float(cfg.get("bound_pair_balance_min_score", 0.35))
      balance_shortfall = jp.clip(
          (min_balance - pair_balance_score) / jp.maximum(min_balance, 1e-6),
          0.0,
          1.0,
      )
      balance_weight = float(cfg.get("bound_pair_balance_after_warmup_penalty_weight", 0.0))
      balance_active = (pair_support_fraction >= min_pair_support).astype(jp.float32)
      pair_balance_after_warmup_penalty = (
          -is_bound * warmup_gate * balance_active * balance_weight * balance_shortfall
      )

      balanced_pair_fraction = jp.minimum(front_pair_fraction, hind_pair_fraction)
      balanced_pair_weight = float(cfg.get("bound_balanced_pair_reward_weight", 0.0))
      balanced_pair_after_warmup_reward = (
          is_bound * warmup_gate * balanced_pair_weight * balanced_pair_fraction
      )

      min_hind_pair = float(cfg.get("bound_min_hind_pair_after_warmup", 0.0))
      hind_pair_shortfall = jp.clip(
          (min_hind_pair - hind_pair_fraction) / jp.maximum(min_hind_pair, 1e-6),
          0.0,
          1.0,
      )
      hind_shortfall_weight = float(cfg.get("bound_hind_pair_shortfall_penalty_weight", 0.0))
      hind_pair_shortfall_penalty = (
          -is_bound * warmup_gate * hind_shortfall_weight * hind_pair_shortfall
      )

      dominance_allowed_gap = float(cfg.get("bound_front_pair_dominance_allowed_gap", 0.20))
      front_pair_dominance_margin = front_pair_fraction - hind_pair_fraction - dominance_allowed_gap
      front_pair_dominance_excess = jp.clip(
          front_pair_dominance_margin / jp.maximum(1.0 - dominance_allowed_gap, 1e-6),
          0.0,
          1.0,
      )
      dominance_weight = float(cfg.get("bound_front_pair_dominance_penalty_weight", 0.0))
      front_pair_dominance_penalty = (
          -is_bound * warmup_gate * dominance_weight * front_pair_dominance_excess
      )

      balanced_gate_target = float(cfg.get("bound_balanced_pair_gate_target", 0.08))
      pair_balance_gate = jp.clip(
          balanced_pair_fraction / jp.maximum(balanced_gate_target, 1e-6),
          0.0,
          1.0,
      )
      gated_progress_weight = float(cfg.get("bound_pair_gated_progress_reward_weight", 0.0))
      pair_gated_progress_reward = (
          is_bound * warmup_gate * gated_progress_weight * soft_progress_score * pair_balance_gate
      )

      gated_speed_target = float(cfg.get("bound_pair_gated_speed_target_vx", 0.0))
      gated_speed_sigma = float(cfg.get("bound_pair_gated_speed_target_sigma", 0.18))
      gated_speed_margin = mean_vx - gated_speed_target
      gated_speed_score = jp.exp(
          -jp.square(gated_speed_margin) / jp.maximum(2.0 * gated_speed_sigma * gated_speed_sigma, 1e-6)
      )
      gated_speed_reward_weight = float(cfg.get("bound_pair_gated_speed_target_reward_weight", 0.0))
      pair_gated_speed_target_reward = (
          is_bound * warmup_gate * gated_speed_reward_weight * pair_balance_gate * gated_speed_score
      )
      gated_speed_shortfall = jp.clip(
          (gated_speed_target - mean_vx) / jp.maximum(gated_speed_target, 1e-6),
          0.0,
          1.0,
      )
      gated_speed_shortfall_weight = float(
          cfg.get("bound_pair_gated_speed_shortfall_penalty_weight", 0.0)
      )
      pair_gated_speed_shortfall_penalty = (
          -is_bound
          * warmup_gate
          * gated_speed_shortfall_weight
          * pair_balance_gate
          * gated_speed_shortfall
      )
    else:
      pair_support_after_warmup_reward = jp.array(0.0, dtype=jp.float32)
      pair_support_shortfall_penalty = jp.array(0.0, dtype=jp.float32)
      diagonal_after_warmup_penalty = jp.array(0.0, dtype=jp.float32)
      pair_balance_after_warmup_penalty = jp.array(0.0, dtype=jp.float32)
      pair_balance_score = jp.array(0.0, dtype=jp.float32)
      balanced_pair_after_warmup_reward = jp.array(0.0, dtype=jp.float32)
      balanced_pair_fraction = jp.array(0.0, dtype=jp.float32)
      hind_pair_shortfall_penalty = jp.array(0.0, dtype=jp.float32)
      front_pair_dominance_penalty = jp.array(0.0, dtype=jp.float32)
      front_pair_dominance_margin = jp.array(0.0, dtype=jp.float32)
      pair_balance_gate = jp.array(0.0, dtype=jp.float32)
      pair_gated_progress_reward = jp.array(0.0, dtype=jp.float32)
      pair_gated_speed_target_reward = jp.array(0.0, dtype=jp.float32)
      pair_gated_speed_shortfall_penalty = jp.array(0.0, dtype=jp.float32)
      gated_speed_target = jp.array(0.0, dtype=jp.float32)
      gated_speed_margin = jp.array(0.0, dtype=jp.float32)

    adjustment = (
        pattern_penalty
        + bound_tracking_penalty
        + transition_tracking_penalty
        + forward_progress_penalty
        + all_four_stall_penalty
        + soft_progress_reward
        + soft_anti_stall_penalty
        + all_four_after_warmup_penalty
        + pair_support_after_warmup_reward
        + pair_support_shortfall_penalty
        + diagonal_after_warmup_penalty
        + pair_balance_after_warmup_penalty
        + balanced_pair_after_warmup_reward
        + hind_pair_shortfall_penalty
        + front_pair_dominance_penalty
        + pair_gated_progress_reward
        + pair_gated_speed_target_reward
        + pair_gated_speed_shortfall_penalty
    )

    return {
        "bound_transition_guard_adjustment": adjustment,
        "bound_pattern_gate": is_bound * gate,
        "bound_pattern_gate_penalty": pattern_penalty,
        "bound_positive_pattern_contribution": is_bound * positive_pattern,
        "bound_tracking_guard_penalty": bound_tracking_penalty,
        "transition_tracking_guard_penalty": transition_tracking_penalty,
        "bound_transition_vx_tracking_gate": (is_bound + is_transition) * vx_gate,
        "bound_transition_stability_gate": (is_bound + is_transition) * stability_gate,
        "bound_forward_progress_target": is_bound * target_vx,
        "bound_forward_progress_mean_vx": is_bound * mean_vx,
        "bound_forward_progress_margin": is_bound * forward_margin,
        "bound_forward_progress_penalty": forward_progress_penalty,
        "bound_all_four_stall_penalty": all_four_stall_penalty,
        "bound_all_four_stall_fraction": is_bound * all_four_fraction,
        "bound_soft_progress_reward": soft_progress_reward,
        "bound_soft_progress_score": is_bound * soft_progress_score,
        "bound_soft_anti_stall_penalty": soft_anti_stall_penalty,
        "bound_soft_anti_stall_margin": is_bound * anti_stall_margin,
        "bound_all_four_after_warmup_penalty": all_four_after_warmup_penalty,
        "bound_pair_support_after_warmup_reward": pair_support_after_warmup_reward,
        "bound_pair_support_shortfall_penalty": pair_support_shortfall_penalty,
        "bound_pair_support_after_warmup_fraction": is_bound * pair_support_fraction,
        "bound_diagonal_after_warmup_penalty": diagonal_after_warmup_penalty,
        "bound_diagonal_after_warmup_fraction": is_bound * diagonal_observed_fraction,
        "bound_pair_balance_after_warmup_penalty": pair_balance_after_warmup_penalty,
        "bound_pair_balance_after_warmup_score": is_bound * pair_balance_score,
        "bound_balanced_pair_after_warmup_reward": balanced_pair_after_warmup_reward,
        "bound_balanced_pair_after_warmup_fraction": is_bound * balanced_pair_fraction,
        "bound_hind_pair_shortfall_penalty": hind_pair_shortfall_penalty,
        "bound_front_pair_dominance_penalty": front_pair_dominance_penalty,
        "bound_front_pair_dominance_margin": is_bound * front_pair_dominance_margin,
        "bound_pair_balance_gate": is_bound * pair_balance_gate,
        "bound_pair_gated_progress_reward": pair_gated_progress_reward,
        "bound_pair_gated_speed_target_reward": pair_gated_speed_target_reward,
        "bound_pair_gated_speed_shortfall_penalty": pair_gated_speed_shortfall_penalty,
        "bound_pair_gated_speed_target": is_bound * gated_speed_target,
        "bound_pair_gated_speed_margin": is_bound * gated_speed_margin,
    }

  """def sample_command(self, rng: jax.Array) -> jax.Array:

    rng, key_regime, key_vx, key_vy, key_yaw = jax.random.split(rng, 5)
    sampled_regime = jax.random.randint(key_regime, shape=(), minval=0, maxval=3)
    
    probs = jp.array([0.3, 0.3, 0.4], dtype=jp.float32)
    sampled_regime = jax.random.choice(key_regime, 3, shape=(), p=probs)

    vx_min_global, vx_max_global = cmd_vx_range

    walk_lo = vx_min_global
    walk_hi = jp.maximum(vx_min_global, TROT_TO_WALK_EXIT)

    trot_lo = WALK_TO_TROT_ENTER 
    trot_hi = jp.maximum(trot_lo, BOUND_TO_TROT_EXIT)

    bound_lo = TROT_TO_BOUND_ENTER 
    bound_hi = jp.maximum(bound_lo, vx_max_global)

    def _sample_uniform(key, lo, hi):
        return jax.random.uniform(key, shape=(), minval=lo, maxval=hi)

    vx_walk = _sample_uniform(key_vx, walk_lo, walk_hi)
    vx_trot = _sample_uniform(key_vx, trot_lo, trot_hi)
    vx_bound = _sample_uniform(key_vx, bound_lo, bound_hi)

    vx = jp.where(
        sampled_regime == MODE_WALK,
        vx_walk,
        jp.where(sampled_regime == MODE_TROT, vx_trot, vx_bound),
    )

    # easier command distribution for gait learning
    vy = jp.where(
        sampled_regime == MODE_BOUND,
        0.0,
        jax.random.uniform(key_vy, shape=(), minval=-0.2, maxval=0.2),
    )

    yaw = jp.where(
        sampled_regime == MODE_BOUND,
        0.0,
        jax.random.uniform(key_yaw, shape=(), minval=-0.2, maxval=0.2),
    )

    return jp.array([vx, vy, yaw], dtype=jp.float32) """
      

  def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, key = jax.random.split(rng)

    pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))
    command = self.sample_command(key)
    mode0 = self._mode_from_command(command)

    state_info = {
        'rng': rng,
        'last_act': jp.zeros(12),
        'last_vel': jp.zeros(12),
        'command': command,
        'last_contact': jp.zeros(4, dtype=bool),
        'feet_air_time': jp.zeros(4),
        'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
        'kick': jp.array([0.0, 0.0]),
        'step': 0,
        'command_resets': jp.array(0, dtype=jp.int32),
        
        'history_len': jp.array(0, dtype=jp.int32),
        'mode': mode0,
        
        'tau_history': jp.zeros((self._H, 18)), 
        'CoM_history': jp.zeros((self._H, 2)),
        'contact_history': jp.zeros((self._H, 4)),
        'feet_history': jp.zeros((self._H, 4, 2)),
        'lin_velocity_history': jp.zeros((self._H, 3)),
        'ang_velocity_history': jp.zeros((self._H,)),
        'com_z_history': jp.zeros((self._H,)),
        'roll_history': jp.zeros((self._H,)),
        'pitch_history': jp.zeros((self._H,)),
        'slipmax_history': jp.zeros((self._H,)),
    }

    obs_history = jp.zeros(15 * 34) # jp.zeros(15 * 32)  # store 15 steps of history
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    reward, done = jp.zeros(2)
    # self.reward_input = None
    metrics = {'total_dist': 0.0}
    for k in state_info['rewards']:
      metrics[k] = state_info['rewards'][k]
    state = State(pipeline_state, obs, reward, done, metrics, state_info)  # pytype: disable=wrong-arg-types
    return state

  def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

    # kick
    push_interval = 10
    kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
    kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
    kick *= jp.mod(state.info['step'], push_interval) == 0
    qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
    qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
    state = state.tree_replace({'pipeline_state.qvel': qvel})

    # physics step
    motor_targets = self._default_pose + action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    x, xd = pipeline_state.x, pipeline_state.xd

    # observation data
    obs = self._get_obs(pipeline_state, state.info, state.obs)
    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qd[6:]

    # foot contact data based on z-position
    foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
    foot_contact_z = foot_pos[:, 2] - self._foot_radius
    contact = foot_contact_z < 1e-3  # a mm or less off the floor
    contact_filt_mm = contact | state.info['last_contact']
    contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
    first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
    state.info['feet_air_time'] += self.dt
    
    
    """# Barkour.py (inside step, replace the single-threshold contact)
    z_on  = 0.003   # 3 mm: turn contact ON
    z_off = 0.010   # 10 mm: turn contact OFF

    contact_on  = foot_contact_z < z_on
    contact_off = foot_contact_z > z_off

    contact = jp.where(contact_on, True,
            jp.where(contact_off, False, state.info['last_contact']))"""



    # done if joint limits are reached or robot is falling
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
    done |= jp.any(joint_angles < self.lowers)
    done |= jp.any(joint_angles > self.uppers)
    done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
    
    # update history buffers
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    
    inv_torso_rot = math.quat_inv(x.rot[0])
    g_body = math.rotate(jp.array([0.0, 0.0, -1.0]), inv_torso_rot)
    # roll = jp.atan2(g_body[1], g_body[2]) # Old code (returns 3.14 when robot upright perfectly)
    roll = jp.atan2(g_body[1], -g_body[2]) # New code (returns 0.0 when robot upright perfectly)
    pitch = jp.atan2(-g_body[0], jp.sqrt(g_body[1] * g_body[1] + g_body[2] * g_body[2])) 
    
    # slip proxy from finite-diff feet xy (using previous feet_history[-1])
    prev_feet_xy = state.info['feet_history'][-1]
    curr_feet_xy = pipeline_state.site_xpos[self._feet_site_id][:, :2]
    feet_vel_xy = (curr_feet_xy - prev_feet_xy) / self.dt
    feet_speed = jp.sqrt(jp.sum(feet_vel_xy * feet_vel_xy, axis=1))
    slipmax = jp.max(jp.where(contact, feet_speed, 0.0))
    
      
    tau_history = jp.roll(state.info['tau_history'], -1, axis=0).at[-1].set(pipeline_state.qfrc_actuator)
    CoM_history = jp.roll(state.info['CoM_history'], -1, axis=0).at[-1].set(pipeline_state.subtree_com[0].copy()[:2])
    contact_history = jp.roll(state.info['contact_history'], -1, axis=0).at[-1].set(contact)
    
    feet_history = jp.roll(state.info['feet_history'], -1, axis=0).at[-1].set(pipeline_state.site_xpos[self._feet_site_id][:, :2])
    lin_velocity_history = jp.roll(state.info['lin_velocity_history'], -1, axis=0).at[-1].set(local_vel)
    ang_velocity_history = jp.roll(state.info['ang_velocity_history'], -1).at[-1].set(base_ang_vel[2])
    
    com_z_history = jp.roll(state.info['com_z_history'], -1).at[-1].set(pipeline_state.subtree_com[0, 2])
    roll_history  = jp.roll(state.info['roll_history'], -1).at[-1].set(roll)
    pitch_history = jp.roll(state.info['pitch_history'], -1).at[-1].set(pitch)
    slipmax_history = jp.roll(state.info['slipmax_history'], -1).at[-1].set(slipmax)
    
    history_len = jp.minimum(state.info['history_len'] + 1, self._H)
    mode = self._update_mode_hysteresis(state.info['mode'], state.info['command'])

    # calculate reward
    
    if STL_REWARD:
      #print("torque shape:", pipeline_state.qfrc_actuator.shape)  = (18,)  /  the latter 6 dimensions comes from the base
      
      reward_input = { 'tau_history': tau_history, 
                        'CoM_history': CoM_history,
                        'contact_history': contact_history,
                        'feet_history': feet_history,
                        'lin_velocity_history': lin_velocity_history,
                        'ang_velocity_history': ang_velocity_history,
                        'com_z_history': com_z_history,
                        'roll_history': roll_history,
                        'pitch_history': pitch_history,
                        'slipmax_history': slipmax_history,
                                    }
      
      
      (r, tau_effort, rho_safety, rho_torque, rho_comz, rho_roll, rho_pitch, rho_slip, rho_bound, rho_trot, rho_walk,
        rho_v_x, rho_v_y, rho_yaw, rho_nlegs, rho_gait,
        rho_v_x_error, rho_v_y_error, rho_yaw_error,
        rho_diag2, rho_stride, rho_duty, rho_3plus_event, rho_support, rho_diag_phase, rho_p2,
        rho_tracking,
        rho_timing,
        rho_pattern,
        rho_front,
        rho_hind,
        rho_hindfront,
        rho_flight,
        rho_front_only,
        rho_hind_only,pitch,roll, FL, HL, FR, HR,
        #slip, 
        #denom,
        #support_dist,
      ) = reward_step(
          reward_input=reward_input,
          commands=state.info['command'],
          mode=mode,
          valid_len=history_len,
          weights_override=self.reward_weights,)
      bound_soft_terms = self._bound_soft_transition_values(
          lin_velocity_history=lin_velocity_history,
          command=state.info['command'],
          valid_len=history_len,
      )
      if self.reward_weights is None:
        pattern_weight = jp.asarray(w_pattern_by_mode)[mode]
        pattern_alpha = jp.asarray(alpha_pattern_by_mode)[mode]
      else:
        pattern_weight = self.reward_weights[3]
        pattern_alpha = self.reward_weights[7]
      is_bound_mode = (mode == MODE_BOUND).astype(jp.float32)
      pattern_component = pattern_weight * jp.tanh(rho_pattern / jp.maximum(pattern_alpha, 1e-6))
      strict_pattern_adjustment = (
          is_bound_mode
          * (bound_soft_terms["bound_effective_pattern_alpha"] - 1.0)
          * pattern_component
      )
      bound_contact_terms = self._bound_contact_pattern_terms(
          contact_history=contact_history,
          lin_velocity_history=lin_velocity_history,
          com_z_history=com_z_history,
          roll_history=roll_history,
          pitch_history=pitch_history,
          command=state.info['command'],
          valid_len=history_len,
          mode=mode,
      )
      bound_guard_terms = self._bound_transition_guard_terms(
          contact_history=contact_history,
          lin_velocity_history=lin_velocity_history,
          com_z_history=com_z_history,
          roll_history=roll_history,
          pitch_history=pitch_history,
          command=state.info['command'],
          valid_len=history_len,
          mode=mode,
          rho_pattern=rho_pattern,
      )
      r = (
          r
          + strict_pattern_adjustment
          + bound_contact_terms["bound_contact_pattern_bonus"]
          + bound_guard_terms["bound_transition_guard_adjustment"]
      )
      termination_penalty_weight = float(
          self._bound_reward_config.get("stl_termination_penalty", 0.0)
      )
      stl_termination_penalty = -termination_penalty_weight * done.astype(jp.float32)
      r = r + stl_termination_penalty / self.dt

      rewards = {
          'total_stl_reward': r,
          'rho_comz': rho_comz,
          'rho_roll': rho_roll,
          'rho_pitch': rho_pitch,
          'rho_slip': rho_slip,
          'rho_bound': rho_bound,
          'rho_trot': rho_trot,
          'rho_walk': rho_walk,
          'rho_safety': rho_safety,
          'torque_lim': rho_torque,
          'more_legs_grounded': rho_nlegs,
          'Vel_track_x': rho_v_x,
          'Vel_track_y': rho_v_y,
          'Ang_vel_track': rho_yaw,
          'gait_shape': rho_gait,
          'smooth_action': tau_effort,
          'x_error': rho_v_x_error,
          'y_error': rho_v_y_error,
          'yaw_error': rho_yaw_error,
          'rho_diag2': rho_diag2, 
          'rho_stride': rho_stride, 
          "rho_duty": rho_duty, 
          "rho_3plus_event": rho_3plus_event, 
          "rho_support": rho_support,
          "rho_diag_phase": rho_diag_phase,
          "rho_p2": rho_p2,
          "rho_tracking": rho_tracking,
          "rho_timing": rho_timing,
          "rho_pattern": rho_pattern,
          "rho_front": rho_front,
          "rho_hind": rho_hind,
          "rho_hindfront": rho_hindfront,
          "rho_flight": rho_flight,
          "rho_front_only": rho_front_only,
          "rho_hind_only": rho_hind_only,
          "pitch": pitch,
          "roll": roll,
          "FL": FL, 
          "HL": HL, 
          "FR": FR, 
          "HR": HR,
          "bound_strict_pattern_adjustment": strict_pattern_adjustment,
          "stl_termination_penalty": stl_termination_penalty,
         # "slip": slip, 
         # "denom": denom,
         # "support_dist": support_dist,
      }
      rewards.update(bound_contact_terms)
      rewards.update(bound_guard_terms)
      rewards.update({
          "bound_strict_pattern_adjustment": strict_pattern_adjustment,
          "bound_soft_command_alpha": is_bound_mode * bound_soft_terms["bound_soft_command_alpha"],
          "bound_soft_speed_gate": is_bound_mode * bound_soft_terms["bound_soft_speed_gate"],
          "bound_effective_pattern_alpha": is_bound_mode * bound_soft_terms["bound_effective_pattern_alpha"],
          "bound_soft_mean_vx": is_bound_mode * bound_soft_terms["bound_soft_mean_vx"],
      })
      rewards = {
          k: rewards.get(k, 0.0) for k in self.reward_config.rewards.scales.keys()
      }
      
      
      reward = jp.clip(r * self.dt, -100.0, 1000.0)
       
    else:             
      rewards = {             
          'tracking_lin_vel': (
              self._reward_tracking_lin_vel(state.info['command'], x, xd)
          ),
          'tracking_ang_vel': (
              self._reward_tracking_ang_vel(state.info['command'], x, xd)
          ),
          'lin_vel_z': self._reward_lin_vel_z(xd),
          'ang_vel_xy': self._reward_ang_vel_xy(xd),
          'orientation': self._reward_orientation(x),
          'torques': self._reward_torques(pipeline_state.qfrc_actuator),  # pytype: disable=attribute-error
          'action_rate': self._reward_action_rate(action, state.info['last_act']),
          'stand_still': self._reward_stand_still(
              state.info['command'], joint_angles,
          ),
          'feet_air_time': self._reward_feet_air_time(
              state.info['feet_air_time'],
              first_contact,
              state.info['command'],
          ),
          'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
          'termination': self._reward_termination(done, state.info['step']),
      }
      rewards = {
          k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
      }
      reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # state management
    state.info['kick'] = kick
    state.info['last_act'] = action
    state.info['last_vel'] = joint_vel
    state.info['feet_air_time'] *= ~contact_filt_mm
    state.info['last_contact'] = contact
    state.info['rewards'] = rewards
    state.info['step'] += 1
    state.info['rng'] = rng
    
    state.info['tau_history'] = tau_history
    state.info['CoM_history'] = CoM_history
    state.info['contact_history'] = contact_history
    state.info['feet_history'] = feet_history
    state.info['lin_velocity_history'] = lin_velocity_history
    state.info['ang_velocity_history'] = ang_velocity_history
    
    state.info['com_z_history'] = com_z_history
    state.info['roll_history'] = roll_history
    state.info['pitch_history'] = pitch_history
    state.info['slipmax_history'] = slipmax_history
    
    state.info['history_len'] = history_len
    state.info['mode'] = mode
    
    done_bool = done
    command_reset = done_bool | (state.info['step'] > 500)

    # On either timeout or termination, start a fresh command and clear all
    # command-conditioned histories so the next [t-H, t] window is consistent.
    next_command_resets = state.info['command_resets'] + command_reset.astype(jp.int32)
    sampled_command = self.sample_command(cmd_rng, next_command_resets)
    next_command = jp.where(command_reset, sampled_command, state.info['command'])
    next_mode = jp.where(
        command_reset,
        self._mode_from_command(next_command),
        state.info['mode'],
    )

    state.info['command'] = next_command
    state.info['step'] = jp.where(command_reset, 0, state.info['step'])
    state.info['command_resets'] = next_command_resets
  
    
    # This makes the [t-H, t] reward window consistent with the current command.
    def _maybe_clear(buf):
      return jp.where(command_reset, jp.zeros_like(buf), buf)

    state.info['tau_history'] = _maybe_clear(state.info['tau_history'])
    state.info['CoM_history'] = _maybe_clear(state.info['CoM_history'])
    state.info['contact_history'] = _maybe_clear(state.info['contact_history'])
    state.info['feet_history'] = _maybe_clear(state.info['feet_history'])
    state.info['lin_velocity_history'] = _maybe_clear(state.info['lin_velocity_history'])
    state.info['ang_velocity_history'] = _maybe_clear(state.info['ang_velocity_history'])
    state.info['com_z_history'] = _maybe_clear(state.info['com_z_history'])
    state.info['roll_history'] = _maybe_clear(state.info['roll_history'])
    state.info['pitch_history'] = _maybe_clear(state.info['pitch_history'])
    state.info['slipmax_history'] = _maybe_clear(state.info['slipmax_history'])
    state.info['history_len'] = jp.where(command_reset, jp.array(0, dtype=jp.int32), state.info['history_len'])
    state.info['mode'] = next_mode
    
    # Also clear action / velocity history terms that feed the observation stack.
    state.info['last_act'] = jp.where(command_reset, jp.zeros_like(state.info['last_act']), state.info['last_act'])
    state.info['last_vel'] = jp.where(command_reset, jp.zeros_like(state.info['last_vel']), state.info['last_vel'])
    state.info['last_contact'] = jp.where(command_reset, jp.zeros_like(state.info['last_contact']), state.info['last_contact'])
    state.info['feet_air_time'] = jp.where(command_reset, jp.zeros_like(state.info['feet_air_time']), state.info['feet_air_time'])
    state.info['kick'] = jp.where(command_reset, jp.zeros_like(state.info['kick']), state.info['kick'])
    
    obs = jp.where(command_reset, self._get_obs(pipeline_state, state.info, jp.zeros_like(state.obs)), obs)
    
    # log total displacement as a proxy metric
    state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]  # or state.metrics['total_dist'] = x.pos[self._torso_idx - 1, 0]
    state.metrics.update(state.info['rewards'])

    done = jp.float32(done_bool)
    state = state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
    return state

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
    local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
    
    mode_one_hot = jax.nn.one_hot(state_info['mode'], 3, dtype=jp.float32)  # [walk, trot, bound]

    obs = jp.concatenate([
        jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
        math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
        state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
        pipeline_state.q[7:] - self._default_pose,           # motor angles
        state_info['last_act'],                              # last action
        mode_one_hot,                                      # jp.array([state_info['mode']], dtype=jp.float32), 
    ])

    # clip, noise
    obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        state_info['rng'], obs.shape, minval=-1, maxval=1
    )
    # stack observations through time
    
    # obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)
    obs = jp.roll(obs_history, obs.shape[0]).at[:obs.shape[0]].set(obs)

    return obs

  # ------------ reward functions----------------
  def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    return jp.square(xd.vel[0, 2])

  def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    return jp.sum(jp.square(xd.ang[0, :2]))

  def _reward_orientation(self, x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))

  def _reward_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    # Penalize changes in actions
    return jp.sum(jp.square(act - last_act))

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(
        -lin_vel_error / self.reward_config.rewards.tracking_sigma
    )
    return lin_vel_reward

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= (
        math.normalize(commands[:2])[1] > 0.05
    )  # no reward for zero command
    return rew_air_time

  def _reward_stand_still(
      self,
      commands: jax.Array,
      joint_angles: jax.Array,
  ) -> jax.Array:
    # Penalize motion at zero commands
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        math.normalize(commands[:2])[1] < 0.1
    )

  def _reward_foot_slip(
      self, pipeline_state: base.State, contact_filt: jax.Array
  ) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

  def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
    return done & (step < 500)

  def render(
      self, trajectory: List[base.State], camera: str | None = None,
      width: int = 240, height: int = 320,
  ) -> Sequence[np.ndarray]:
    camera = camera or 'track'
    return super().render(trajectory, camera=camera, width=width, height=height)

#envs.register_environment('barkour', BarkourEnv)

# %%
