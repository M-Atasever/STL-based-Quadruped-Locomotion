import jax
import jax.numpy as jnp
from jax import lax
from coeff_config import (
    tau_max, beta,gamma_tau,
    alpha_b, w_vx, alpha_vx, w_vy, alpha_vy, w_yaw, alpha_yaw,
     w_tau, alpha_tau,
    w_gait, alpha_gait,
    eps_vx_by_mode, eps_vy_by_mode, eps_yaw_by_mode,
    eps_diag_sync, eps_pair_sync, eps_bound_overlap,
    min_contacts, com_z_by_mode, abs_vz_by_mode, roll_abs_by_mode, pitch_abs_by_mode, 
    slip_speed_by_mode, cop_com_xy_dist_by_mode, stride_period_by_mode, duty_factor_by_mode,
    K_REQUIRE_3PLUS, MODE_BOUND, MODE_TROT, MODE_WALK, H_WARMUP_MIN_VALID, DT, 
    diag_2contact_fraction_min_by_mode
)


def tanh_norm(rho, alpha):  # bound to [-1,1]
    return jnp.tanh(rho / alpha)

def smooth_min(vals, beta=10.0):
    vals = jnp.asarray(vals)
    return -jax.nn.logsumexp(-beta * vals) / beta
    # return -jnp.log(jnp.sum(jnp.exp(-beta * vals))) / beta

def _valid_mask(H, valid_len):
    # History is right-aligned (newest appended at end after jp.roll(...).at[-1].set(...))
    # valid entries are the last `valid_len` rows.
    idx = jnp.arange(H)
    return idx >= (H - valid_len)

def _masked_min(x, mask):
    x = jnp.asarray(x)
    m = jnp.asarray(mask)
    while m.ndim < x.ndim:
        m = m[..., None]
    x_masked = jnp.where(m, x, jnp.inf)
    return jnp.min(x_masked)

def _masked_mean(x, mask):
    x = jnp.asarray(x)
    m = jnp.asarray(mask, dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m[..., None]
    denom = jnp.maximum(jnp.sum(m), 1.0)
    return jnp.sum(x * m) / denom

def _gait_shape_margin(contact_hist, mode, mask,
                       eps_diag_sync, eps_pair_sync, eps_bound_overlap):
    """
    contact_hist: (H,4) with order [FL, HL, FR, HR]
    mode: int {0,1,2} = walk, trot, bound
    returns rho_gait_shape (higher is better)
    """
    c = contact_hist.astype(jnp.float32)

    # Index map based on your foot order in Barkour.py:
    # [front_left, hind_left, front_right, hind_right] = [FL, HL, FR, HR]
    FL, HL, FR, HR = 0, 1, 2, 3

    # Generic counts
    nlegs = jnp.sum(c, axis=1)  # (H,)

    # --- Walk / walking-trot proxy ---
    # Encourage >=2 stance contacts (no flight), especially in slow regime.
    walk_no_flight_margin = _masked_min(nlegs - 2.0, mask)

    # --- Trot proxy ---
    # Diagonal pair sync: FL~HR, FR~HL
    trot_diag_sync_err = 0.5 * (
        _masked_mean(jnp.abs(c[:, FL] - c[:, HR]), mask) +
        _masked_mean(jnp.abs(c[:, FR] - c[:, HL]), mask)
    )
    trot_margin = eps_diag_sync - trot_diag_sync_err

    # --- Bound proxy ---
    # Front pair sync + hind pair sync
    bound_pair_sync_err = 0.5 * (
        _masked_mean(jnp.abs(c[:, FL] - c[:, FR]), mask) +
        _masked_mean(jnp.abs(c[:, HL] - c[:, HR]), mask)
    )
    front_occ = 0.5 * (c[:, FL] + c[:, FR])
    hind_occ  = 0.5 * (c[:, HL] + c[:, HR])
    fore_hind_overlap = _masked_mean(front_occ * hind_occ, mask)
    bound_margin = smooth_min([
        eps_pair_sync - bound_pair_sync_err,
        eps_bound_overlap - fore_hind_overlap,
    ], beta=10.0)

    # mode-select
    return jnp.where(
        mode == 0,
        walk_no_flight_margin,
        jnp.where(mode == 1, trot_margin, bound_margin)
    )

def softsign_p(x, p=2.0):
    return x / jnp.power(1.0 + jnp.abs(x)**p, 1.0/p)


"""
def cop_from_cfrc_ext(model: mujoco.MjModel,
                      data: mujoco.MjData,
                      foot_body_names=( 'foot_front_left',
                                        'foot_hind_left',
                                        'foot_front_right',
                                        'foot_hind_right'),
                      plane_z=0.0,
                      eps=1e-6):
    # Sum world-frame external forces/torques on the foot bodies
    F = np.zeros(3)
    M = np.zeros(3)
    for name in foot_body_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, name)
        bid = model.site_bodyid[sid]
        fx, fy, fz, tx, ty, tz = data.cfrc_ext[bid]
        r = data.xipos[bid]  # body COM position in world
        f = np.array([fx, fy, fz])
        tau = np.array([tx, ty, tz])  # moment about body COM
        F += f 
        M += np.cross(r, f) + tau     # moment about world origin

    # If origin isn’t on the ground plane, shift the reference of the moment
    if abs(plane_z) > 1e-12:
        # M^{O'} = M^O - (O' - O) x F, with O'=[0,0,plane_z]
        M = M - np.cross(np.array([0.0, 0.0, plane_z]), F)

    Fz = max(F[2], eps)
    cop = np.array([-M[1]/Fz, M[0]/Fz])  # (x,y) on plane z=plane_z
    return cop, Fz, F, M
    #return np.array([3, 5]), 0.1, np.random.rand((4,3)), np.random.rand((4,3))
    """

"""def friction_cone_margin(model, data, Fz_min=Fz_min, delta=0.0):
    # returns the worst (minimum) margin across all active foot-ground contacts
    worst = np.inf
    tmp = np.zeros(6)  
    for k in range(data.ncon):
        con = data.contact[k]
        mujoco.mj_contactForce(model, data, k, tmp) # 3 dims for forces, 3 dims for torques
        # Common case with 3D friction: tmp[0]=fn, tmp[1]=ft1, tmp[2]=ft2
        fn, ft1, ft2 = tmp[0], tmp[1], tmp[2]
        mu1, mu2 = con.friction[0], con.friction[1]
        # Per-axis pyramid constraints: |ft1| <= mu1*fn, |ft2| <= mu2*fn
        m1 = mu1 * fn - abs(ft1)
        m2 = mu2 * fn - abs(ft2)
        mFz = fn - Fz_min
        # margin for this contact (subtract a safety buffer delta)
        m = min(m1, m2, mFz) - delta
        worst = min(worst, m)
    # If no contacts, return a large negative (violated/undefined)
    return worst if worst < np.inf else -1e6"""


def reward_step(reward_input, commands, mode, valid_len, weights_override=None):
    # histories (all right-aligned)
    tau = reward_input["tau_history"]                  # (H,12)
    c   = reward_input["contact_history"].astype(jnp.float32)  # (H,4) [FL,HL,FR,HR]
    feet_xy = reward_input["feet_history"]             # (H,4,2)
    com_xy  = reward_input["CoM_history"]              # (H,2)
    v_hist  = reward_input["lin_velocity_history"]     # (H,3)
    yaw_hist = reward_input["ang_velocity_history"]    # (H,)
    com_z_hist = reward_input["com_z_history"]         # (H,)
    roll_hist  = reward_input["roll_history"]          # (H,) radians
    pitch_hist = reward_input["pitch_history"]         # (H,) radians
    slip_hist  = reward_input["slipmax_history"]       # (H,)
            
    # reward weights for tuning
    if weights_override is None:
        # w = jnp.array([w_vx, w_tau, w_gait, alpha_b, alpha_vx, alpha_tau, alpha_gait, beta], dtype=jnp.float32)
        w = jnp.array([w_vx, alpha_b, alpha_vx, beta], dtype=jnp.float32)
    else:
        w = weights_override

    # _w_vx, _w_tau, _w_gait, _alpha_b, _alpha_vx, _alpha_tau, _alpha_gait, _beta = w
    
    _w_vx, _alpha_b, _alpha_vx, _beta = w

    H = tau.shape[0]
    valid_len = jnp.minimum(valid_len, H)
    mask = _valid_mask(H, valid_len)

    # commands
    v_x_star = commands[0]
    v_y_star = commands[1]
    yaw_star = commands[2]

    # mode-conditioned tolerances
    eps_vx = jnp.asarray(eps_vx_by_mode)[mode]
    eps_vy = jnp.asarray(eps_vy_by_mode)[mode]
    eps_yaw_m = jnp.asarray(eps_yaw_by_mode)[mode]

    # ---------- shared safety ----------
    # torque limit
    rho_torque = _masked_min(tau_max - jnp.abs(tau), mask)

    # min contacts
    n_contacts = jnp.sum(c, axis=1)  # (H,)
    rho_nlegs = _masked_min(n_contacts - float(min_contacts), mask)

    # COM z floor
    zmin = jnp.asarray(com_z_by_mode)[mode]
    rho_comz = _masked_min(com_z_hist - zmin, mask)

    # |vz| bound (use v_hist[:,2])
    vzmax = jnp.asarray(abs_vz_by_mode)[mode]
    rho_vz = _masked_min(vzmax - jnp.abs(v_hist[:, 2]), mask)

    # roll/pitch bounds
    roll_max = jnp.asarray(roll_abs_by_mode)[mode] * jnp.pi / 180.0
    pitch_max = jnp.asarray(pitch_abs_by_mode)[mode] * jnp.pi / 180.0
    rho_roll = _masked_min(roll_max - jnp.abs(roll_hist), mask)
    rho_pitch = _masked_min(pitch_max - jnp.abs(pitch_hist), mask)

    # slip bound
    slipmax = jnp.asarray(slip_speed_by_mode)[mode]
    rho_slip = _masked_min(slipmax - slip_hist, mask)

    # COM-to-stance-centroid distance proxy (JAX-safe replacement for COP–COM)
    denom = jnp.maximum(jnp.sum(c, axis=1, keepdims=True), 1.0)      # (H,1)
    centroid = jnp.sum(c[:, :, None] * feet_xy, axis=1) / denom      # (H,2)
    support_dist = jnp.linalg.norm(com_xy - centroid, axis=1)        # (H,)
    dmax = jnp.asarray(cop_com_xy_dist_by_mode)[mode]
    rho_support = _masked_min(dmax - support_dist, mask)

    rho_safety = smooth_min([rho_torque / 5.0, 
                            # rho_nlegs, 
                             rho_comz / 0.03, 
                            # rho_vz / 0.10, 
                             rho_roll / (5.0 * jnp.pi/180.0), 
                             rho_pitch / (5.0 * jnp.pi/180.0), 
                             rho_slip / 0.30, 
                             ], beta=_beta)

    # ---------- tracking ----------
    v_x_error_hist = jnp.abs(v_hist[:, 0] - v_x_star)
    v_y_error_hist = jnp.abs(v_hist[:, 1] - v_y_star)
    yaw_error_hist = jnp.abs(yaw_hist - yaw_star)

    rho_v_x = _masked_min(eps_vx - v_x_error_hist, mask)
    rho_v_y = _masked_min(eps_vy - v_y_error_hist, mask)
    rho_yaw = _masked_min(eps_yaw_m - yaw_error_hist, mask)

    rho_v_x_error = _masked_mean(v_x_error_hist, mask)
    rho_v_y_error = _masked_mean(v_y_error_hist, mask)
    rho_yaw_error = _masked_mean(yaw_error_hist, mask)

    # ---------- gait-shape ----------
    # Don’t enforce gait-shape too early (no history)
    gait_enabled = valid_len >= H_WARMUP_MIN_VALID

    FL, HL, FR, HR = 0, 1, 2, 3
    mask2 = (n_contacts == 2.0)
    diag2 = ((c[:, FL] == 1) & (c[:, HR] == 1) & (c[:, HL] == 0) & (c[:, FR] == 0)) | \
            ((c[:, FR] == 1) & (c[:, HL] == 1) & (c[:, FL] == 0) & (c[:, HR] == 0))

    # diag2 fraction over valid window
    num2 = _masked_mean(diag2.astype(jnp.float32) * mask2.astype(jnp.float32), mask)
    den2 = _masked_mean(mask2.astype(jnp.float32), mask)
    diag2_frac = num2 / jnp.maximum(den2, 1e-6)

    diag2_min = jnp.asarray(diag_2contact_fraction_min_by_mode)[mode]
    rho_diag2 = diag2_frac - diag2_min

    # stride estimate from touchdown count (rough but JAX-safe)
    c01 = c.astype(jnp.int32)
    td = (c01[1:] == 1) & (c01[:-1] == 0)        # (H-1,4)
    mask_td = mask[1:]
    td_counts = jnp.sum(td.astype(jnp.float32) * mask_td[:, None].astype(jnp.float32), axis=0)  # (4,)
    td_mean = jnp.mean(td_counts)
    valid_steps = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
    stride_steps_est = valid_steps / jnp.maximum(td_mean, 1.0)
    stride_sec_est = stride_steps_est * DT

    stride_bounds = jnp.asarray(stride_period_by_mode) # Shape (N, 2)
    duty_bounds = jnp.asarray(duty_factor_by_mode)     # Shape (N, 2)
    
    stride_lo = stride_bounds[mode, 0]
    stride_hi = stride_bounds[mode, 1]
    rho_stride = jnp.minimum(stride_sec_est - stride_lo, stride_hi - stride_sec_est)

    # “duty” proxy = per-leg contact fraction (consistent with your logged contacts)
    duty_leg = jnp.sum(c * mask[:, None].astype(jnp.float32), axis=0) / valid_steps
    duty_est = jnp.mean(duty_leg)
    duty_lo = duty_bounds[mode, 0]
    duty_hi = duty_bounds[mode, 1]
    rho_duty = jnp.minimum(duty_est - duty_lo, duty_hi - duty_est)

    # walking-trot: require 3+ contact occurs within last K steps
    K = jnp.minimum(valid_len, K_REQUIRE_3PLUS)
    idx = jnp.arange(H)
    lastK_mask = mask & (idx >= (H - K))
    # event margin: max over last K of (n_contacts-3)
    event_vals = jnp.where(lastK_mask, n_contacts - 3.0, -jnp.inf)
    rho_3plus_event = jnp.max(event_vals)

    # bound priors
    front_sync = _masked_mean(jnp.abs(c[:, FL] - c[:, FR]), mask)
    hind_sync  = _masked_mean(jnp.abs(c[:, HL] - c[:, HR]), mask)
    cF = 0.5 * (c[:, FL] + c[:, FR])
    cH = 0.5 * (c[:, HL] + c[:, HR])
    overlap = _masked_mean(cF * cH, mask)

    rho_front = eps_pair_sync - front_sync
    rho_hind  = eps_pair_sync  - hind_sync
    rho_overlap = eps_bound_overlap - overlap

    rho_bound = smooth_min([rho_front / 0.10, rho_hind / 0.10, rho_overlap / 0.10, rho_duty / 0.075], beta=_beta)  # extra: rho_stride / 0.06
    rho_walk  = smooth_min([rho_diag2 / 0.05, rho_stride / 0.05, rho_duty / 0.035, rho_3plus_event, rho_support / 0.08], beta=_beta)
    rho_trot  = smooth_min([rho_diag2 / 0.05, rho_stride / 0.04, rho_duty / 0.31, rho_support / 0.08], beta=_beta)

    rho_gait = jnp.where(mode == MODE_WALK, rho_walk,
                 jnp.where(mode == MODE_TROT, rho_trot, rho_bound))

    rho_gait = jnp.where(gait_enabled, rho_gait, 0.0)

    # effort
    tau_sq_sum = jnp.sum(jnp.square(tau), axis=1)  # (H,)
    tau_effort = _masked_mean(tau_sq_sum, mask)

    # reward: gate gait/tracking on safety
    #safe_ok = (rho_safety > 0.0).astype(jnp.float32)
    r = (
        1.0 * (tanh_norm(rho_safety, _alpha_b))
       # + safe_ok * (
         +   _w_vx * (tanh_norm(rho_v_x, _alpha_vx))
            + w_vy * (tanh_norm(rho_v_y, alpha_vy))
            + w_yaw * (tanh_norm(rho_yaw, alpha_yaw))
           # + _w_gait * (tanh_norm(rho_gait, _alpha_gait))
            - gamma_tau * tau_effort     # _w_tau * (tanh_norm(tau_effort, _alpha_tau))
        )
    

    return (r, tau_effort, rho_safety, rho_torque, rho_comz, rho_roll, rho_pitch, rho_slip, rho_bound, rho_trot, rho_walk,
            rho_v_x, rho_v_y, rho_yaw, rho_nlegs, rho_gait,
            rho_v_x_error, rho_v_y_error, rho_yaw_error, rho_diag2, rho_stride, rho_duty, rho_3plus_event, rho_support)
    

