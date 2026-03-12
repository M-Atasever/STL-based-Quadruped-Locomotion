# =========================
# Barkour STL reward config
# =========================

# Barkour vb joint torque limit and joint acc limit
tau_max = 18.0 # Nm
q_dot_max = 25.0 # rad/s

# temporal horizon (50 Hz -> H=10 means 0.2 s)
H = 28
H_WARMUP_MIN_VALID = 5  # optional: start strict temporal mins after a few steps
DT = 0.02

# -------------------------
# gait mode IDs
# -------------------------
MODE_WALK = 0
MODE_TROT = 1
MODE_BOUND = 2

# -------------------------
# command-speed hysteresis on |vx| [m/s]
# user regime: walk [0,0.7], trot [0.7,1.7], bound >1.7
# use hysteresis to avoid mode chattering around boundaries
# -------------------------
WALK_TO_TROT_ENTER = 0.75
TROT_TO_WALK_EXIT = 0.60

TROT_TO_BOUND_ENTER = 1.80
BOUND_TO_TROT_EXIT = 1.55

# robust aggregation
beta = 10.0

# -------------------------
# mode-conditioned tracking tolerances
# (indexed by mode: walk, trot, bound)
# -------------------------
eps_vx_by_mode = (0.51, 1.05, 1.05)
eps_vy_by_mode = (0.1, 0.1, 0.1)
eps_yaw_by_mode = (0.1, 0.1, 0.1)

# -------------------------
# gait-shape proxy tolerances (contact-pattern based)
# -------------------------
# lower is better for these errors; rho = eps - err
eps_diag_sync = 0.20      # trot diagonal sync error tolerance
eps_pair_sync = 0.20      # bound front/hind pair sync error tolerance
eps_bound_overlap = 0.4  # allow some overlap between fore/hind in bound
eps_walk_no_flight = 0.0  # nlegs-2 >= 0 means no flight (>=2 contacts)

# learned stability parameters
abs_vz_by_mode = (0.148, 0.179, 0.3)
com_z_by_mode = (0.225, 0.231, 0.35)  # looser "never fall" floor (more tolerant)
cop_com_xy_dist_by_mode = (0.156, 0.351, 0.357)

roll_abs_by_mode = (3.10, 3.92, 22.0) # degree
pitch_abs_by_mode = (3.43, 3.83, 30.0) # degree
slip_speed_by_mode = (0.6, 1.3, 1.5)

min_contacts = 2
# Optional: require "3+ contacts occurs at least once every X seconds"
K_REQUIRE_3PLUS = 11  # walking-trot: “3+ contacts occurs within last 0.22s ≈ 11 steps”

# gait structure
stride_period_by_mode = ([0.42, 0.52], [0.32, 0.40], [0.28, 0.40])
duty_factor_by_mode = ([0.62, 0.69], [0.44, 0.50], [0.25, 0.40],)
diag_phase_error_by_mode = (0.12, 0.15, )
diag_2contact_fraction_min_by_mode = (0.95, 0.95, 0.0)

# -------------------------
# weights
# -------------------------
w_vx = 1.77
w_vy = 0.01
w_yaw = 0.01
#w_torque = 0.5
#w_nlegs = 0.4
w_tau = 0.04
w_gait = 0.01

# squash alphas
alpha_b = 200.0
alpha_vx = 0.5
alpha_vy = 1.0
alpha_yaw = 1.0
#alpha_torque = 10.0
#alpha_nlegs = 1.0
alpha_tau = 300.0     # because tau_dot is torque^2-mean, scale is larger
alpha_gait = 50.0

# torque cost (if torque available)
gamma_tau = 1e-6