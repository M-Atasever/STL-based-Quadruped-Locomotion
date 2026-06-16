# =========================
# Barkour STL reward config
# =========================

# Barkour vb joint torque limit and joint velocity limit
# reward_step only uses tau_max directly. q_dot_max is kept for compatibility.
tau_max = 18.0  # Nm
q_dot_max = 25.0  # rad/s

# -----------------------------------------------------------------------------
# Temporal settings
# -----------------------------------------------------------------------------
DT = 0.02  # 50 Hz control step
H = 30
H_MAX = H
H_by_mode = (30, 24, 24)  # walk, trot, bound
H_WARMUP_MIN_VALID = 8

# -----------------------------------------------------------------------------
# Gait mode IDs
# -----------------------------------------------------------------------------
MODE_WALK = 0
MODE_TROT = 1
MODE_BOUND = 2

# -----------------------------------------------------------------------------
# Command-speed hysteresis on |vx| [m/s]
# -----------------------------------------------------------------------------
WALK_TO_TROT_ENTER = 0.72
TROT_TO_WALK_EXIT = 0.65
TROT_TO_BOUND_ENTER = 1.55
BOUND_TO_TROT_EXIT = 1.45

# Mixed-regime sampling and bound-focused curriculum helpers.
regime_sample_probs = (0.25, 0.35, 0.40)
cmd_vx_range = (0.0, 1.9)
cmd_vy_range = (-0.2, 0.2)
cmd_yaw_range = (-0.2, 0.2)
bound_vx_sample_range = (1.45, 1.9)

# Robust aggregation sharpness.
beta = 0.5
beta_safe = 0.5
beta_timing = 0.5
beta_pattern = 0.5

# -----------------------------------------------------------------------------
# Mode-conditioned tracking tolerances
# -----------------------------------------------------------------------------
eps_vx_by_mode = (0.55, 0.60, 0.60)
eps_vy_by_mode = (0.05, 0.05, 0.05)
eps_yaw_by_mode = (0.05, 0.05, 0.05)

# -----------------------------------------------------------------------------
# Shared safety thresholds (walk, trot, bound)
# -----------------------------------------------------------------------------
min_contacts_by_mode = (2, 2, 0)
min_contacts = 2
abs_vz_by_mode = (0.15, 0.18, 0.25)
com_z_by_mode = (0.18, 0.22, 0.16)
cop_com_xy_dist_by_mode = (0.10, 0.13, 0.15)
roll_abs_by_mode = (10.0, 7.0, 15.0)
pitch_abs_by_mode = (8.0, 7.0, 15.0)
slip_speed_by_mode = (0.40, 0.75, 0.85)

# -----------------------------------------------------------------------------
# Mode-dependent gait structure parameters
# -----------------------------------------------------------------------------
stride_period_by_mode = ((0.42, 0.54), (0.33, 0.43), (0.25, 0.31))
duty_factor_by_mode = ((0.62, 0.74), (0.52, 0.60), (0.62, 0.70))

diag_phase_error_by_mode = (0.10, 0.11, 1.00)
diag_2contact_fraction_min_by_mode = (0.92, 0.99, 0.00)
diag_2contact_fraction_max_by_mode = (1.00, 1.00, 0.03)
contact2_fraction_min_by_mode = (0.00, 0.70, 0.00)

flight_fraction_min_by_mode = (0.00, 0.00, 0.04)
front_only_fraction_min_by_mode = (0.00, 0.00, 0.18)
hind_only_fraction_min_by_mode = (0.00, 0.00, 0.16)
front_only_fraction_max_by_mode = (1.00, 1.00, 0.42)
hind_only_fraction_max_by_mode = (1.00, 1.00, 0.38)

all4_fraction_max_by_mode = (1.00, 0.10, 0.30)
all4_fraction_min_by_mode = (0.00, 0.50, 0.10)

pair_front_mismatch_max_by_mode = (1.00, 1.00, 0.10)
pair_hind_mismatch_max_by_mode = (1.00, 1.00, 0.14)
hind_to_front_lag_by_mode = ((0.00, 1.00), (0.00, 1.00), (0.33, 0.40))

# Compatibility aliases kept for older code paths / logs.
pair_phase_error_max_by_mode = pair_front_mismatch_max_by_mode
front_pair_sync_min_by_mode = (0.0, 0.0, 0.86)
hind_pair_sync_min_by_mode = (0.0, 0.0, 0.82)
lateral_2contact_fraction_max_by_mode = (0.0, 0.0, 0.12)
hind_to_front_lag_steps_by_mode = ((0, 0), (0, 0), (3, 6))
all4_fraction_min_by_mode = (0.0, 0.0, 0.18)
pair2_fraction_min_by_mode = (0.0, 0.0, 0.88)

# Optional foot-clearance predicate. The current BarkourEnv does not yet feed
# clearance_history, so reward_step automatically disables this term when absent.
clearance_min_by_mode = (0.020, 0.025, 0.030)

# Walk mode helper: require occasional >=3-contact support in the recent window.
K_REQUIRE_3PLUS = 11

# -----------------------------------------------------------------------------
# Legacy proxy tolerances retained for compatibility with older code paths.
# -----------------------------------------------------------------------------
eps_diag_sync = 0.20
eps_pair_sync = 0.20
eps_bound_overlap = 0.40
eps_walk_no_flight = 0.0

# -----------------------------------------------------------------------------
# Margin scales for normalized robustness
# -----------------------------------------------------------------------------
# Safety
tau_margin_scale = 5.0
min_contacts_margin_scale = 1.0
com_z_margin_scale = 0.03
abs_vz_margin_scale = 0.10
roll_margin_scale_deg = 5.0
pitch_margin_scale_deg = 5.0
slip_margin_scale = 0.20
support_margin_scale = 0.06

# Tracking: use the mode-dependent eps_* values as the natural normalization.
track_axis_weights = (1.0, 0.01, 0.01)

# Timing / pattern
stride_margin_scale_by_mode = (0.05, 0.04, 0.04)
duty_margin_scale_by_mode = (0.05, 0.03, 0.05)
diag_phase_margin_scale_by_mode = (0.05, 0.05, 1.00)
diag2_margin_scale_by_mode = (0.05, 0.03, 0.05)
contact2_margin_scale_by_mode = (1.00, 1.00, 1.00)
pair_mismatch_margin_scale_by_mode = (1.00, 1.00, 0.05)
hindfront_margin_scale_by_mode = (1.00, 1.00, 0.10)
flight_margin_scale_by_mode = (1.00, 1.00, 0.05)
front_only_margin_scale_by_mode = (1.00, 1.00, 0.05)
hind_only_margin_scale_by_mode = (1.00, 1.00, 0.05)
all4_margin_scale_by_mode = (1.00, 1.00, 0.08)
event3plus_margin_scale = 1.0
bound_event_margin_scale = 1.0
clearance_margin_scale_by_mode = (0.010, 0.010, 0.010)

# Compatibility alias.
pair_phase_margin_scale_by_mode = pair_mismatch_margin_scale_by_mode

# -----------------------------------------------------------------------------
# Grouped reward weights and tanh alphas
# -----------------------------------------------------------------------------
w_safe_by_mode = (1.0, 1.0, 1.0)
w_track_by_mode = (1.0, 1.0, 1.15)
#w_timing_by_mode = (0.0, 0.0, 1.2)
w_pattern_by_mode = (1.1, 1.2, 1.3)

alpha_safe_by_mode = (0.9, 0.6, 0.8)
alpha_track_by_mode = (0.8, 0.7, 0.8)
#alpha_timing_by_mode = (0.0, 0.0, 0.7)
alpha_pattern_by_mode = (1.2, 0.4, 0.7)

# Torque effort regularizer coefficient
gamma_tau = 1e-6  # 1e-5



