tau_max = 18    # barkour vb torque limit

# temporal horizon
H = 10

# friction & contact
Fz_min: float = 1e-7
mu: float = 0.8             #  ??????????????????
# robust aggregation
beta: float = 10.0
alpha_b: float = 0.05 #100.0
# tolerances
eps_v: float = 0.1
eps_yaw: float = 0.05
# weights
w_vx: float = 1.0
w_vy: float = 1.0
w_yaw: float = 1.0
w_torque: float = 0.5
w_nlegs: float = 0.5
w_tau: float = 1.0
# squash alphas
alpha_vx: float = 0.3
alpha_vy: float = 0.3
alpha_yaw: float = 0.2
alpha_torque: float = 10.0
alpha_nlegs: float = 1.0
alpha_tau: float = 0.5
# torque cost (if torque available)
gamma_tau: float = 1e-6  # try 7 

