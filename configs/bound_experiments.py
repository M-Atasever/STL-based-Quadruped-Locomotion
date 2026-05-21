"""Opt-in BOUND gait experiment configs.

The default training path does not import any of these values unless an
experiment name is explicitly requested.
"""

from __future__ import annotations

from copy import deepcopy


DEFAULT_EXPERIMENT = {
    "name": "default",
    "description": "Baseline behavior: no sampler, reward, or PPO changes.",
    "sampler": {"type": "default"},
    "bound_reward": {
        "bound_contact_pattern_weight": 0.0,
        "bound_diagonal_penalty_weight": 0.0,
        "bound_all_four_penalty_weight": 0.0,
        "bound_pair_sync_weight": 0.0,
        "enable_tracking_gate": False,
    },
}


EXPERIMENTS = {
    "bound_v1_sampler_only": {
        "name": "bound_v1_sampler_only",
        "description": "Increase BOUND-range command exposure only; reward and PPO are unchanged.",
        "sampler": {
            "type": "bound_mixture",
            "nominal_probability": 0.10,
            "transition_probability": 0.20,
            "bound_probability": 0.70,
            "nominal_vx_range": (0.0, 1.69),
            "transition_vx_range": (1.55, 1.85),
            "bound_vx_range": (1.75, 2.25),
            "nominal_vy_range": (-0.20, 0.20),
            "nominal_yaw_range": (-0.20, 0.20),
            "bound_vy_range": (-0.15, 0.15),
            "bound_yaw_range": (-0.25, 0.25),
        },
    },
    "bound_v2_contact_pattern": {
        "name": "bound_v2_contact_pattern",
        "description": "Sampler-only experiment plus small BOUND-only contact-pattern shaping.",
        "inherits": "bound_v1_sampler_only",
        "bound_reward": {
            "bound_contact_pattern_weight": 0.50,
            "bound_diagonal_penalty_weight": 0.50,
            "bound_all_four_penalty_weight": 0.25,
            "bound_pair_sync_weight": 0.0,
        },
    },
    "bound_v3_contact_curriculum": {
        "name": "bound_v3_contact_curriculum",
        "description": "Contact-pattern reward with a per-environment command-reset curriculum.",
        "inherits": "bound_v2_contact_pattern",
        "sampler": {
            "type": "bound_curriculum",
            "stage_a_reset_count": 250,
            "stage_b_reset_count": 750,
            "stage_a_vx_range": (1.55, 1.78),
            "stage_b_vx_range": (1.75, 2.00),
            "stage_c_vx_range": (2.00, 2.25),
            "vy_range": (-0.15, 0.15),
            "yaw_range": (-0.25, 0.25),
        },
    },
    "bound_v3b_tracking_gated_contact": {
        "name": "bound_v3b_tracking_gated_contact",
        "description": (
            "Curriculum/contact experiment with smaller BOUND contact reward "
            "gated by forward tracking and base stability."
        ),
        "inherits": "bound_v3_contact_curriculum",
        "bound_reward": {
            "bound_contact_pattern_weight": 0.20,
            "bound_diagonal_penalty_weight": 0.35,
            "bound_all_four_penalty_weight": 0.20,
            "bound_pair_sync_weight": 0.0,
            "enable_tracking_gate": True,
            "gate_min_valid_steps": 8,
            "gate_vx_error_full": 0.55,
            "gate_vx_error_zero": 1.10,
            "gate_base_height_full": 0.22,
            "gate_base_height_zero": 0.18,
            "gate_roll_pitch_full_rad": 0.30,
            "gate_roll_pitch_zero_rad": 0.45,
        },
    },
    "bound_stage_b_low_bound_transition_protected": {
        "name": "bound_stage_b_low_bound_transition_protected",
        "description": (
            "Static low-BOUND/transition resume experiment that avoids the "
            "command-reset curriculum and gates BOUND pattern reward on "
            "forward tracking and stability."
        ),
        "sampler": {
            "type": "bound_mixture",
            "nominal_probability": 0.35,
            "transition_probability": 0.35,
            "bound_probability": 0.30,
            "nominal_vx_range": (1.45, 1.69),
            "transition_vx_range": (1.60, 1.80),
            "bound_vx_range": (1.75, 1.90),
            "nominal_vy_range": (-0.12, 0.12),
            "nominal_yaw_range": (-0.15, 0.15),
            "bound_vy_range": (-0.10, 0.10),
            "bound_yaw_range": (-0.15, 0.15),
        },
        "bound_reward": {
            "bound_contact_pattern_weight": 0.15,
            "bound_diagonal_penalty_weight": 0.25,
            "bound_all_four_penalty_weight": 0.15,
            "bound_pair_sync_weight": 0.0,
            "enable_tracking_gate": True,
            "enable_bound_transition_guard": True,
            "gate_min_valid_steps": 8,
            "gate_vx_error_full": 0.45,
            "gate_vx_error_zero": 0.95,
            "gate_base_height_full": 0.23,
            "gate_base_height_zero": 0.18,
            "gate_roll_pitch_full_rad": 0.28,
            "gate_roll_pitch_zero_rad": 0.45,
            "bound_pattern_guard_weight": 1.0,
            "bound_pattern_guard_alpha": 0.6,
            "bound_tracking_guard_penalty_weight": 0.40,
            "transition_tracking_guard_penalty_weight": 0.20,
            "transition_guard_min_vx": 1.45,
            "transition_guard_max_vx": 1.75,
        },
    },
    "bound_stage_b_forward_progress_protected": {
        "name": "bound_stage_b_forward_progress_protected",
        "description": (
            "Static low-BOUND transition experiment with stronger BOUND "
            "anti-stall and all-four-stance penalties."
        ),
        "inherits": "bound_stage_b_low_bound_transition_protected",
        "bound_reward": {
            "bound_contact_pattern_weight": 0.15,
            "bound_diagonal_penalty_weight": 0.25,
            "bound_all_four_penalty_weight": 0.25,
            "enable_bound_forward_progress_guard": True,
            "bound_forward_progress_min_fraction": 0.45,
            "bound_forward_progress_min_vx": 0.70,
            "bound_forward_progress_penalty_weight": 1.25,
            "bound_all_four_stall_penalty_weight": 0.75,
        },
    },
    "bound_stage_c_ramp_softmode": {
        "name": "bound_stage_c_ramp_softmode",
        "description": (
            "Transition-friendly fine-tune that keeps high-trot exposure, "
            "ramps BOUND contact pressure over command speed, and only fully "
            "activates strict BOUND pattern reward once realized speed is nontrivial."
        ),
        "inherits": "bound_stage_b_low_bound_transition_protected",
        "sampler": {
            "type": "bound_mixture",
            "nominal_probability": 0.55,
            "transition_probability": 0.30,
            "bound_probability": 0.15,
            "nominal_vx_range": (1.45, 1.69),
            "transition_vx_range": (1.50, 1.85),
            "bound_vx_range": (1.75, 1.90),
            "nominal_vy_range": (-0.12, 0.12),
            "nominal_yaw_range": (-0.15, 0.15),
            "bound_vy_range": (-0.08, 0.08),
            "bound_yaw_range": (-0.12, 0.12),
        },
        "bound_reward": {
            "bound_contact_pattern_weight": 0.12,
            "bound_diagonal_penalty_weight": 0.20,
            "bound_all_four_penalty_weight": 0.12,
            "enable_bound_soft_transition": True,
            "bound_soft_command_alpha_lo": 1.55,
            "bound_soft_command_alpha_hi": 1.90,
            "bound_soft_speed_gate_center": 0.90,
            "bound_soft_speed_gate_width": 0.15,
            "bound_soft_min_valid_steps": 8,
            "bound_tracking_guard_penalty_weight": 0.25,
            "transition_tracking_guard_penalty_weight": 0.12,
            "stl_termination_penalty": 2.0,
        },
    },
    "bound_stage_d_soft_progress": {
        "name": "bound_stage_d_soft_progress",
        "description": (
            "Stage-C soft transition plus gentle BOUND progress shaping and "
            "warmup-gated all-four suppression to escape stationary survival."
        ),
        "inherits": "bound_stage_c_ramp_softmode",
        "bound_reward": {
            "enable_bound_soft_progress_reward": True,
            "bound_soft_progress_scale_vx": 0.60,
            "bound_soft_progress_reward_weight": 0.35,
            "bound_soft_anti_stall_min_vx": 0.30,
            "bound_soft_anti_stall_penalty_weight": 0.45,
            "bound_all_four_after_warmup_allowed_fraction": 0.55,
            "bound_all_four_after_warmup_penalty_weight": 0.35,
            "bound_contact_pattern_weight": 0.14,
            "bound_diagonal_penalty_weight": 0.20,
            "bound_all_four_penalty_weight": 0.16,
            "stl_termination_penalty": 2.0,
        },
    },
    "bound_stage_e_clean_pairs": {
        "name": "bound_stage_e_clean_pairs",
        "description": (
            "Stage-D resume experiment that keeps gentle BOUND progress but "
            "adds warmup-gated pressure for clean front/hind pair support and "
            "against the diagonal crawl found in Stage D."
        ),
        "inherits": "bound_stage_d_soft_progress",
        "sampler": {
            "type": "bound_mixture",
            "nominal_probability": 0.45,
            "transition_probability": 0.25,
            "bound_probability": 0.30,
            "nominal_vx_range": (1.45, 1.69),
            "transition_vx_range": (1.55, 1.90),
            "bound_vx_range": (1.75, 2.00),
            "nominal_vy_range": (-0.10, 0.10),
            "nominal_yaw_range": (-0.12, 0.12),
            "bound_vy_range": (-0.06, 0.06),
            "bound_yaw_range": (-0.10, 0.10),
        },
        "bound_reward": {
            "bound_contact_pattern_weight": 0.22,
            "bound_diagonal_penalty_weight": 0.32,
            "bound_all_four_penalty_weight": 0.20,
            "bound_soft_progress_reward_weight": 0.28,
            "bound_soft_anti_stall_min_vx": 0.35,
            "bound_soft_anti_stall_penalty_weight": 0.35,
            "bound_all_four_after_warmup_allowed_fraction": 0.45,
            "bound_all_four_after_warmup_penalty_weight": 0.45,
            "enable_bound_clean_pair_reward": True,
            "bound_pair_support_after_warmup_reward_weight": 0.35,
            "bound_min_pair_support_after_warmup": 0.12,
            "bound_pair_support_shortfall_penalty_weight": 0.35,
            "bound_diagonal_after_warmup_allowed_fraction": 0.18,
            "bound_diagonal_after_warmup_penalty_weight": 0.55,
            "bound_pair_balance_min_score": 0.30,
            "bound_pair_balance_after_warmup_penalty_weight": 0.12,
            "stl_termination_penalty": 2.0,
        },
    },
    "bound_stage_f_balanced_pairs": {
        "name": "bound_stage_f_balanced_pairs",
        "description": (
            "Stage-E resume experiment that targets the front-pair loophole by "
            "rewarding balanced front/hind pair use, penalizing hind-pair "
            "shortfall and front-pair dominance, and only adding extra speed "
            "reward through a balanced-pair gate."
        ),
        "inherits": "bound_stage_e_clean_pairs",
        "sampler": {
            "type": "bound_mixture",
            "nominal_probability": 0.40,
            "transition_probability": 0.25,
            "bound_probability": 0.35,
            "nominal_vx_range": (1.45, 1.69),
            "transition_vx_range": (1.55, 1.90),
            "bound_vx_range": (1.75, 2.00),
            "nominal_vy_range": (-0.10, 0.10),
            "nominal_yaw_range": (-0.12, 0.12),
            "bound_vy_range": (-0.05, 0.05),
            "bound_yaw_range": (-0.08, 0.08),
        },
        "bound_reward": {
            "bound_contact_pattern_weight": 0.18,
            "bound_diagonal_penalty_weight": 0.35,
            "bound_all_four_penalty_weight": 0.22,
            "bound_soft_progress_reward_weight": 0.08,
            "bound_soft_anti_stall_min_vx": 0.30,
            "bound_soft_anti_stall_penalty_weight": 0.25,
            "bound_all_four_after_warmup_allowed_fraction": 0.40,
            "bound_all_four_after_warmup_penalty_weight": 0.50,
            "bound_pair_support_after_warmup_reward_weight": 0.15,
            "bound_min_pair_support_after_warmup": 0.20,
            "bound_pair_support_shortfall_penalty_weight": 0.25,
            "bound_diagonal_after_warmup_allowed_fraction": 0.12,
            "bound_diagonal_after_warmup_penalty_weight": 0.55,
            "bound_pair_balance_min_score": 0.55,
            "bound_pair_balance_after_warmup_penalty_weight": 0.35,
            "bound_balanced_pair_reward_weight": 0.95,
            "bound_min_hind_pair_after_warmup": 0.14,
            "bound_hind_pair_shortfall_penalty_weight": 0.65,
            "bound_front_pair_dominance_allowed_gap": 0.18,
            "bound_front_pair_dominance_penalty_weight": 0.45,
            "bound_balanced_pair_gate_target": 0.08,
            "bound_pair_gated_progress_reward_weight": 0.20,
            "stl_termination_penalty": 2.0,
        },
    },
    "bound_stage_g_pair_gated_speed": {
        "name": "bound_stage_g_pair_gated_speed",
        "description": (
            "Stage-F resume experiment that keeps the balanced front/hind pair "
            "scaffold and reintroduces modest speed pressure only through a "
            "balanced-pair gate."
        ),
        "inherits": "bound_stage_f_balanced_pairs",
        "sampler": {
            "type": "bound_mixture",
            "nominal_probability": 0.38,
            "transition_probability": 0.22,
            "bound_probability": 0.40,
            "nominal_vx_range": (1.45, 1.69),
            "transition_vx_range": (1.55, 1.90),
            "bound_vx_range": (1.75, 2.00),
            "nominal_vy_range": (-0.10, 0.10),
            "nominal_yaw_range": (-0.12, 0.12),
            "bound_vy_range": (-0.05, 0.05),
            "bound_yaw_range": (-0.08, 0.08),
        },
        "bound_reward": {
            "bound_contact_pattern_weight": 0.16,
            "bound_diagonal_penalty_weight": 0.35,
            "bound_all_four_penalty_weight": 0.24,
            "bound_soft_progress_reward_weight": 0.04,
            "bound_soft_anti_stall_min_vx": 0.25,
            "bound_soft_anti_stall_penalty_weight": 0.15,
            "bound_all_four_after_warmup_allowed_fraction": 0.36,
            "bound_all_four_after_warmup_penalty_weight": 0.50,
            "bound_pair_support_after_warmup_reward_weight": 0.10,
            "bound_min_pair_support_after_warmup": 0.22,
            "bound_pair_support_shortfall_penalty_weight": 0.20,
            "bound_pair_balance_min_score": 0.55,
            "bound_pair_balance_after_warmup_penalty_weight": 0.30,
            "bound_balanced_pair_reward_weight": 0.80,
            "bound_min_hind_pair_after_warmup": 0.15,
            "bound_hind_pair_shortfall_penalty_weight": 0.50,
            "bound_front_pair_dominance_allowed_gap": 0.18,
            "bound_front_pair_dominance_penalty_weight": 0.35,
            "bound_balanced_pair_gate_target": 0.10,
            "bound_pair_gated_progress_reward_weight": 0.18,
            "bound_pair_gated_speed_target_vx": 0.40,
            "bound_pair_gated_speed_target_sigma": 0.18,
            "bound_pair_gated_speed_target_reward_weight": 0.35,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.45,
            "stl_termination_penalty": 2.0,
        },
    },
    "bound_stage_h_pair_gated_speed_052": {
        "name": "bound_stage_h_pair_gated_speed_052",
        "description": (
            "Stage-G resume experiment that keeps the balanced-pair gate and "
            "raises the pair-gated speed target from 0.40 m/s to 0.52 m/s."
        ),
        "inherits": "bound_stage_g_pair_gated_speed",
        "bound_reward": {
            "bound_pair_gated_speed_target_vx": 0.52,
            "bound_pair_gated_speed_target_sigma": 0.22,
            "bound_pair_gated_speed_target_reward_weight": 0.40,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.42,
            "bound_pair_gated_progress_reward_weight": 0.20,
            "bound_soft_anti_stall_min_vx": 0.30,
            "bound_soft_anti_stall_penalty_weight": 0.18,
        },
    },
    "bound_stage_i_pair_gated_speed_065": {
        "name": "bound_stage_i_pair_gated_speed_065",
        "description": (
            "Stage-H resume experiment that keeps the balanced-pair gate and "
            "raises the pair-gated speed target from 0.52 m/s to 0.65 m/s."
        ),
        "inherits": "bound_stage_h_pair_gated_speed_052",
        "bound_reward": {
            "bound_pair_gated_speed_target_vx": 0.65,
            "bound_pair_gated_speed_target_sigma": 0.25,
            "bound_pair_gated_speed_target_reward_weight": 0.42,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.40,
            "bound_pair_gated_progress_reward_weight": 0.22,
            "bound_all_four_after_warmup_allowed_fraction": 0.32,
            "bound_all_four_after_warmup_penalty_weight": 0.55,
            "bound_soft_anti_stall_min_vx": 0.35,
            "bound_soft_anti_stall_penalty_weight": 0.20,
        },
    },
    "bound_stage_j_pair_gated_speed_072": {
        "name": "bound_stage_j_pair_gated_speed_072",
        "description": (
            "Stage-I resume experiment that raises the pair-gated speed target "
            "to 0.72 m/s and slightly tightens diagonal suppression."
        ),
        "inherits": "bound_stage_i_pair_gated_speed_065",
        "bound_reward": {
            "bound_pair_gated_speed_target_vx": 0.72,
            "bound_pair_gated_speed_target_sigma": 0.28,
            "bound_pair_gated_speed_target_reward_weight": 0.42,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.38,
            "bound_pair_gated_progress_reward_weight": 0.22,
            "bound_diagonal_penalty_weight": 0.42,
            "bound_diagonal_after_warmup_allowed_fraction": 0.06,
            "bound_diagonal_after_warmup_penalty_weight": 0.70,
            "bound_all_four_after_warmup_allowed_fraction": 0.30,
            "bound_all_four_after_warmup_penalty_weight": 0.58,
            "bound_soft_anti_stall_min_vx": 0.40,
            "bound_soft_anti_stall_penalty_weight": 0.20,
        },
    },
    "bound_stage_k_pair_gated_speed_080": {
        "name": "bound_stage_k_pair_gated_speed_080",
        "description": (
            "Stage-J resume experiment that continues the speed ladder by "
            "raising the pair-gated speed target to 0.80 m/s while preserving "
            "the balanced-pair contact gate."
        ),
        "inherits": "bound_stage_j_pair_gated_speed_072",
        "bound_reward": {
            "bound_pair_gated_speed_target_vx": 0.80,
            "bound_pair_gated_speed_target_sigma": 0.32,
            "bound_pair_gated_speed_target_reward_weight": 0.43,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.36,
            "bound_pair_gated_progress_reward_weight": 0.22,
            "bound_all_four_after_warmup_allowed_fraction": 0.28,
            "bound_all_four_after_warmup_penalty_weight": 0.60,
            "bound_soft_anti_stall_min_vx": 0.45,
            "bound_soft_anti_stall_penalty_weight": 0.20,
        },
    },
    "bound_stage_l_pair_gated_speed_090": {
        "name": "bound_stage_l_pair_gated_speed_090",
        "description": (
            "Stage-K resume experiment that continues the speed ladder by "
            "raising the pair-gated speed target to 0.90 m/s while preserving "
            "the balanced-pair contact gate."
        ),
        "inherits": "bound_stage_k_pair_gated_speed_080",
        "bound_reward": {
            "bound_pair_gated_speed_target_vx": 0.90,
            "bound_pair_gated_speed_target_sigma": 0.36,
            "bound_pair_gated_speed_target_reward_weight": 0.44,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.34,
            "bound_pair_gated_progress_reward_weight": 0.22,
            "bound_all_four_after_warmup_allowed_fraction": 0.27,
            "bound_all_four_after_warmup_penalty_weight": 0.62,
            "bound_soft_anti_stall_min_vx": 0.50,
            "bound_soft_anti_stall_penalty_weight": 0.20,
        },
    },
    "bound_stage_m_pair_gated_speed_085": {
        "name": "bound_stage_m_pair_gated_speed_085",
        "description": (
            "Stage-K resume experiment that retries the speed ladder with a "
            "smaller 0.85 m/s pair-gated speed target after the 0.90 m/s "
            "branch regressed high-command behavior."
        ),
        "inherits": "bound_stage_k_pair_gated_speed_080",
        "bound_reward": {
            "bound_pair_gated_speed_target_vx": 0.85,
            "bound_pair_gated_speed_target_sigma": 0.34,
            "bound_pair_gated_speed_target_reward_weight": 0.435,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.35,
            "bound_pair_gated_progress_reward_weight": 0.22,
            "bound_all_four_after_warmup_allowed_fraction": 0.27,
            "bound_all_four_after_warmup_penalty_weight": 0.61,
            "bound_soft_anti_stall_min_vx": 0.48,
            "bound_soft_anti_stall_penalty_weight": 0.20,
        },
    },
    "bound_stage_n_pair_gated_speed_085_highcmd": {
        "name": "bound_stage_n_pair_gated_speed_085_highcmd",
        "description": (
            "Stage-K resume experiment with the 0.85 m/s pair-gated speed "
            "target plus more transition and high-command BOUND exposure to "
            "avoid the ramp/high-command regression seen in Stage M."
        ),
        "inherits": "bound_stage_k_pair_gated_speed_080",
        "sampler": {
            "type": "bound_mixture",
            "nominal_probability": 0.30,
            "transition_probability": 0.25,
            "bound_probability": 0.45,
            "nominal_vx_range": (1.45, 1.69),
            "transition_vx_range": (1.55, 2.05),
            "bound_vx_range": (1.75, 2.25),
            "nominal_vy_range": (-0.10, 0.10),
            "nominal_yaw_range": (-0.12, 0.12),
            "bound_vy_range": (-0.05, 0.05),
            "bound_yaw_range": (-0.08, 0.08),
        },
        "bound_reward": {
            "bound_pair_gated_speed_target_vx": 0.85,
            "bound_pair_gated_speed_target_sigma": 0.34,
            "bound_pair_gated_speed_target_reward_weight": 0.435,
            "bound_pair_gated_speed_shortfall_penalty_weight": 0.35,
            "bound_pair_gated_progress_reward_weight": 0.22,
            "bound_all_four_after_warmup_allowed_fraction": 0.27,
            "bound_all_four_after_warmup_penalty_weight": 0.61,
            "bound_soft_anti_stall_min_vx": 0.48,
            "bound_soft_anti_stall_penalty_weight": 0.20,
            "stl_termination_penalty": 2.5,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_experiment_config(name: str | None) -> dict | None:
    """Returns a fully materialized config, or None for baseline/default."""
    if name in (None, "", "default", "baseline"):
        return None
    if name not in EXPERIMENTS:
        valid = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment '{name}'. Valid experiments: {valid}")

    raw = EXPERIMENTS[name]
    if "inherits" in raw:
        parent = get_experiment_config(raw["inherits"])
        cfg = _deep_merge(DEFAULT_EXPERIMENT, parent or {})
        cfg = _deep_merge(cfg, raw)
    else:
        cfg = _deep_merge(DEFAULT_EXPERIMENT, raw)
    cfg["name"] = name
    return cfg
