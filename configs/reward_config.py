from ml_collections import config_dict


def get_config():
  """Returns reward config for barkour quadruped environment."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            # The coefficients for all reward terms used for training. All
            # physical quantities are in SI units, if no otherwise specified,
            # i.e. joint positions are in rad, positions are measured in meters,
            # torques in Nm, and time in seconds, and forces in Newtons.
            scales=config_dict.ConfigDict(
                dict(
                    # Tracking rewards are computed using exp(-delta^2/sigma)
                    # sigma can be a hyperparameters to tune.
                    # Track the base x-y velocity (no z-velocity tracking.)
                    tracking_lin_vel=1.5,
                    # Track the angular velocity along z-axis, i.e. yaw rate.
                    tracking_ang_vel=0.8,
                    # Below are regularization terms, we roughly divide the
                    # terms to base state regularizations, joint
                    # regularizations, and other behavior regularizations.
                    # Penalize the base velocity in z direction, L2 penalty.
                    lin_vel_z=-2.0,
                    # Penalize the base roll and pitch rate. L2 penalty.
                    ang_vel_xy=-0.05,
                    # Penalize non-zero roll and pitch angles. L2 penalty.
                    orientation=-5.0,
                    # L2 regularization of joint torques, |tau|^2.
                    torques=-0.0002,
                    # Penalize the change in the action and encourage smooth
                    # actions. L2 regularization |action - last_action|^2
                    action_rate=-0.01,
                    # Encourage long swing steps.  However, it does not
                    # encourage high clearances.
                    feet_air_time=0.2,
                    # Encourage no motion at zero command, L2 regularization
                    # |q - q_default|^2.
                    stand_still=-0.5,
                    # Early termination penalty.
                    termination=-1.0,
                    # Penalizing foot slipping on the ground.
                    foot_slip=-0.1,
                )
            ),
            # Tracking reward = exp(-error^2/sigma).
            tracking_sigma=0.25,
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(
          rewards=get_default_rewards_config(),
      )
  )

  return default_config



def get_stl_config():
  """Returns reward config for barkour quadruped environment."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            # The coefficients for all reward terms used for training. All
            # physical quantities are in SI units, if no otherwise specified,
            # i.e. joint positions are in rad, positions are measured in meters,
            # torques in Nm, and time in seconds, and forces in Newtons.
            scales=config_dict.ConfigDict(
                dict(
                    # 1-) Torque Limits 
                    # 2-) Center of Mass within the Support Polygon for Static Stability
                    # 3-) Increased Size of Support Polygon (more legs on the ground)
                    # 4-) Zero Moment Point (ZMP) within the Support Polygon for Dynamic Stability
                    # 5-) Velocity tracking
                    # 6-) Heading tracking
                    # 7-) Contact force limits due to friction cone constraints
                    # 8-) Center of pressure (CoP) remaining in the support polygon constraints 
                    
                    # Final grouped terms
                    rho_safety=1.0,
                    rho_tracking=1.0,
                    rho_timing=1.0,
                    rho_pattern=1.0,
                    total_stl_reward=1.0,
                    
                   # combined_safety = 1.0,
                    torque_lim = 1.0,
                   # CoM_stab = 1.0,
                    more_legs_grounded = 1.0,
                   # Zmp_stab = 1.0,
                    Vel_track_x = 1.0,
                    Vel_track_y = 1.0,
                    Ang_vel_track = 1.0,
                   # friction_cone = 1.0,
                   # CoP_stab = 1.0,
                    smooth_action = 1.0,
                    gait_shape = 1.0,
                    rho_comz = 1.0,
                    rho_roll = 1.0,
                    rho_pitch = 1.0,
                    rho_slip = 1.0,
                    rho_bound = 1.0,
                    rho_trot = 1.0,
                    rho_walk = 1.0,
                    
                    x_error = 0.0,
                    y_error = 0.0,
                    yaw_error = 0.0,
                    
                    rho_diag2 = 1.0, 
                    rho_stride = 1.0, 
                    rho_duty = 1.0, 
                    rho_3plus_event = 1.0, 
                    rho_support = 1.0,
                    rho_diag_phase = 1.0,
                    rho_p2 = 1.0,
                    rho_front=1.0,
                    rho_hind=1.0,
                    rho_hindfront=1.0,
                    rho_flight=1.0,
                    rho_front_only=1.0,
                    rho_hind_only=1.0,
                    rho_all4=1.0,
                    rho_bound_event=1.0,
                    bound_front_pair_sync=1.0,
                    bound_hind_pair_sync=1.0,
                    bound_front_or_hind_pair_support=1.0,
                    bound_diagonal_trot_penalty=1.0,
                    bound_all_four_stance_penalty=1.0,
                    bound_contact_pattern_reward=1.0,
                    bound_contact_pattern_bonus=1.0,
                    bound_raw_contact_pattern_bonus=1.0,
                    bound_contact_gate=1.0,
                    bound_vx_tracking_gate=1.0,
                    bound_stability_gate=1.0,
                    bound_soft_command_alpha=1.0,
                    bound_soft_speed_gate=1.0,
                    bound_effective_pattern_alpha=1.0,
                    bound_soft_mean_vx=1.0,
                    bound_strict_pattern_adjustment=1.0,
                    bound_transition_guard_adjustment=1.0,
                    bound_pattern_gate=1.0,
                    bound_pattern_gate_penalty=1.0,
                    bound_positive_pattern_contribution=1.0,
                    bound_tracking_guard_penalty=1.0,
                    transition_tracking_guard_penalty=1.0,
                    bound_transition_vx_tracking_gate=1.0,
                    bound_transition_stability_gate=1.0,
                    bound_forward_progress_target=1.0,
                    bound_forward_progress_mean_vx=1.0,
                    bound_forward_progress_margin=1.0,
                    bound_forward_progress_penalty=1.0,
                    bound_all_four_stall_penalty=1.0,
                    bound_all_four_stall_fraction=1.0,
                    bound_soft_progress_reward=1.0,
                    bound_soft_progress_score=1.0,
                    bound_soft_anti_stall_penalty=1.0,
                    bound_soft_anti_stall_margin=1.0,
                    bound_all_four_after_warmup_penalty=1.0,
                    bound_pair_support_after_warmup_reward=1.0,
                    bound_pair_support_shortfall_penalty=1.0,
                    bound_pair_support_after_warmup_fraction=1.0,
                    bound_diagonal_after_warmup_penalty=1.0,
                    bound_diagonal_after_warmup_fraction=1.0,
                    bound_pair_balance_after_warmup_penalty=1.0,
                    bound_pair_balance_after_warmup_score=1.0,
                    bound_balanced_pair_after_warmup_reward=1.0,
                    bound_balanced_pair_after_warmup_fraction=1.0,
                    bound_hind_pair_shortfall_penalty=1.0,
                    bound_front_pair_dominance_penalty=1.0,
                    bound_front_pair_dominance_margin=1.0,
                    bound_pair_balance_gate=1.0,
                    bound_pair_gated_progress_reward=1.0,
                    bound_pair_gated_speed_target_reward=1.0,
                    bound_pair_gated_speed_shortfall_penalty=1.0,
                    bound_pair_gated_speed_target=1.0,
                    bound_pair_gated_speed_margin=1.0,
                    stl_termination_penalty=1.0,
                    pitch=0.0,
                    roll=0.0,
                    FL=1.0, 
                    HL=1.0, 
                    FR=1.0, 
                    HR=1.0,
                    #slip=0.0, 
                   # denom=1.0,
                   # support_dist=0.0,
      
                    
                )
            ),
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(
          rewards=get_default_rewards_config(),
      )
  )

  return default_config
