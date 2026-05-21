import jax
import jax.numpy as jnp

from coeff_config import (
    tau_max,
    beta,
    gamma_tau,
   # alpha_b,
   # w_vx,
   # alpha_vx,
   # w_vy,
   # alpha_vy,
   # w_yaw,
   # alpha_yaw,
   # w_tau,
   # alpha_tau,
   # w_gait,
   # alpha_gait,
    eps_vx_by_mode,
    eps_vy_by_mode,
    eps_yaw_by_mode,
    min_contacts_by_mode,
    com_z_by_mode,
    abs_vz_by_mode,
    roll_abs_by_mode,
    pitch_abs_by_mode,
    slip_speed_by_mode,
    cop_com_xy_dist_by_mode,
    stride_period_by_mode,
    duty_factor_by_mode,
    diag_phase_error_by_mode,
    diag_2contact_fraction_min_by_mode,
    diag_2contact_fraction_max_by_mode,
    contact2_fraction_min_by_mode,
    flight_fraction_min_by_mode,
    front_only_fraction_min_by_mode,
    hind_only_fraction_min_by_mode,
    pair_phase_error_max_by_mode,
    hind_to_front_lag_by_mode,
    K_REQUIRE_3PLUS,
    H_by_mode,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    H_WARMUP_MIN_VALID,
    DT,
    
    # normalization scales
    tau_margin_scale,
    min_contacts_margin_scale,
    com_z_margin_scale,
    abs_vz_margin_scale,
    roll_margin_scale_deg,
    pitch_margin_scale_deg,
    slip_margin_scale,
    support_margin_scale,
    track_axis_weights,
    stride_margin_scale_by_mode,
    duty_margin_scale_by_mode,
    diag_phase_margin_scale_by_mode,
    diag2_margin_scale_by_mode,
    contact2_margin_scale_by_mode,
    pair_phase_margin_scale_by_mode,
    hindfront_margin_scale_by_mode,
    flight_margin_scale_by_mode,
    front_only_margin_scale_by_mode,
    hind_only_margin_scale_by_mode,
    event3plus_margin_scale,
    # group weights / alphas
    w_safe_by_mode,
    w_track_by_mode,
    w_timing_by_mode,
    w_pattern_by_mode,
    alpha_safe_by_mode,
    alpha_track_by_mode,
    alpha_timing_by_mode,
    alpha_pattern_by_mode,
)


def tanh_norm(rho, alpha):
    return jnp.tanh(rho / jnp.maximum(alpha, 1e-6))


def smooth_min(vals, beta=10.0):
    vals = jnp.asarray(vals, dtype=jnp.float32)
    return -jax.nn.logsumexp(-beta * vals) / beta

def smooth_min_sign_preserving(vals, beta=10.0):
    vals = jnp.asarray(vals, dtype=jnp.float32)
    w = jax.nn.softmax(-beta * vals)
    return jnp.sum(w * vals)

def _window_mask(H, valid_len, horizon):
    """Mask the last `active_len=min(valid_len,horizon)` entries of right-aligned history."""
    active_len = jnp.minimum(jnp.minimum(valid_len, horizon), H)
    idx = jnp.arange(H)
    return idx >= (H - active_len)


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

def _safe_div(x, s):
    return x / jnp.maximum(s, 1e-6)


def _touchdown_events(signal, mask):
    # Detect 0->1 contact transitions inside the active window.
    signal = (signal > 0.5)
    prev = jnp.concatenate([jnp.array([False]), signal[:-1]])
    # Treat the first valid step as a potential touchdown if it starts in contact.
    idx = jnp.arange(signal.shape[0])
    first_valid = jnp.argmax(mask.astype(jnp.int32))
    prev = jnp.where(idx == first_valid, False, prev)
    return signal & (~prev) & mask

"""def _touchdown_events(signal, mask):
    signal = (signal > 0.5)
    prev = jnp.concatenate([signal[:1], signal[:-1]])  # keep true previous
    return signal & (~prev) & mask"""

def _event_count(events):
    return jnp.sum(events.astype(jnp.float32))


def _mean_period_seconds(event_counts, valid_steps):
    # valid_steps is scalar count of active steps. event_counts shape (K,)
    valid_steps = jnp.maximum(valid_steps, 1.0)
    has_evt = event_counts > 0.0
    period_steps = valid_steps / jnp.maximum(event_counts, 1.0)
    denom = jnp.maximum(jnp.sum(has_evt.astype(jnp.float32)), 1.0)
    return jnp.sum(jnp.where(has_evt, period_steps, 0.0)) / denom * DT


def _inphase_event_error(events_a, events_b, period_steps):
    """Approximate phase error for two event trains that should be in phase."""
    period_steps = jnp.maximum(period_steps, 1.0)
    H = events_a.shape[0]
    idx = jnp.arange(H)
    ia = idx[:, None]
    ib = idx[None, :]
    dist = jnp.abs(ia - ib).astype(jnp.float32)
    valid = events_a[:, None] & events_b[None, :]
    min_d = jnp.min(jnp.where(valid, dist, jnp.inf), axis=1)
    frac = min_d / period_steps
    frac = jnp.minimum(frac, jnp.abs(1.0 - frac))
    select = events_a & jnp.isfinite(min_d)
    denom = jnp.maximum(jnp.sum(select.astype(jnp.float32)), 1.0)
    err = jnp.sum(jnp.where(select, frac, 0.0)) / denom
    # If there are no usable events, return a large error.
    return jnp.where(jnp.any(select), err, 1.0)


def _forward_lag(events_src, events_dst, period_steps):
    """Normalized mean lag from each source event to the next destination event."""
    period_steps = jnp.maximum(period_steps, 1.0)
    H = events_src.shape[0]
    idx = jnp.arange(H)
    src = idx[:, None]
    dst = idx[None, :]
    d = (dst - src).astype(jnp.float32)
    valid = events_src[:, None] & events_dst[None, :] & (d >= 0.0)
    min_fwd = jnp.min(jnp.where(valid, d, jnp.inf), axis=1)
    select = events_src & jnp.isfinite(min_fwd)
    denom = jnp.maximum(jnp.sum(select.astype(jnp.float32)), 1.0)
    lag = jnp.sum(jnp.where(select, min_fwd / period_steps, 0.0)) / denom
    return jnp.where(jnp.any(select), lag, -1.0)

def _default_group_params(mode):
    mode = jnp.asarray(mode, dtype=jnp.int32)
    return (
        jnp.asarray(w_track_by_mode)[mode],
        jnp.asarray(w_safe_by_mode)[mode],
        jnp.asarray(w_timing_by_mode)[mode],
        jnp.asarray(w_pattern_by_mode)[mode],
        jnp.asarray(alpha_track_by_mode)[mode],
        jnp.asarray(alpha_safe_by_mode)[mode],
        jnp.asarray(alpha_timing_by_mode)[mode],
        jnp.asarray(alpha_pattern_by_mode)[mode],
        jnp.asarray(beta, dtype=jnp.float32),
    )


"""def compute_phase_offset_robustness(c1, c2, expected_lag, stride_period, dt, tolerance=0.1):
    
    Computes robustness for the phase offset between two contact signals.
    
    Args:
        c1: Contact history for pair 1 (e.g., Front) [H, 2]
        c2: Contact history for pair 2 (e.g., Hind) [H, 2]
        expected_lag: Normalized lag (0.0 to 1.0 of stride period)
        stride_period: Current stride period in seconds
        dt: Control step size (0.02)
        tolerance: Allowed deviation in phase
    
    # 1. Convert contact pairs to a single 'pair-in-contact' signal (0.0 to 1.0)
    sig1 = jnp.mean(c1, axis=-1) 
    sig2 = jnp.mean(c2, axis=-1)
    
    # 2. Convert normalized lag to discrete buffer steps
    # e.g., if lag is 0.45 and stride is 0.4s at 50Hz: 0.45 * 0.4 / 0.02 = 9 steps
    lag_steps = jnp.round((expected_lag * stride_period) / dt).astype(jnp.int32)
    
    # 3. Align the signals
    # We compare the current sig1 with sig2 from 'lag_steps' ago
    # We use jnp.roll to shift sig2, though in a real windowed STL you'd just index
    sig2_shifted = jnp.roll(sig2, lag_steps)
    
    # 4. Calculate Robustness
    # 1.0 if they match perfectly, -1.0 if they are perfectly out of sync
    # We use a smooth absolute difference
    diff = jnp.abs(sig1 - sig2_shifted)
    robustness = 1.0 - 2.0 * diff
    
    # 5. Mask out the 'invalid' start of the buffer caused by the roll
    mask = jnp.arange(sig1.shape[0]) >= lag_steps
    return jnp.sum(robustness * mask) / jnp.sum(mask)"""


def reward_step(reward_input, commands, mode, valid_len, weights_override=None):
    # Histories are right-aligned and use foot order [FL, HL, FR, HR].
    tau = reward_input["tau_history"]                  # (H, n_tau)
    c = reward_input["contact_history"].astype(jnp.float32)  # (H,4)
    feet_xy = reward_input["feet_history"]            # (H,4,2)
    com_xy = reward_input["CoM_history"]              # (H,2)
    v_hist = reward_input["lin_velocity_history"]     # (H,3)
    yaw_hist = reward_input["ang_velocity_history"]   # (H,)
    com_z_hist = reward_input["com_z_history"]        # (H,)
    roll_hist = reward_input["roll_history"]          # (H,)
    pitch_hist = reward_input["pitch_history"]        # (H,)
    slip_hist = reward_input["slipmax_history"]       # (H,)

    """if weights_override is None:
        # Backwards-compatible override vector structure used by Barkour.py
        # [w_vx, w_vy, w_yaw, alpha_b, alpha_vx, alpha_vy, alpha_yaw, beta]
        w = jnp.array(
            [w_vx, w_vy, w_yaw, w_gait, alpha_b, alpha_vx, alpha_vy, alpha_yaw, alpha_gait, beta],
            dtype=jnp.float32,
        )
    else:
        w = weights_override

    _w_vx, _w_vy, _w_yaw, _w_gait, _alpha_b, _alpha_vx, _alpha_vy, _alpha_yaw, _alpha_gait, _beta = w """
    
    if weights_override is None:
        _w_track, _w_safe, _w_timing, _w_pattern, _alpha_track, _alpha_safe, _alpha_timing, _alpha_pattern, _beta = _default_group_params(mode)
    else:
        # New grouped override vector:
        # [w_track, w_safe, w_timing, w_pattern, alpha_track, alpha_safe, alpha_timing, alpha_pattern, beta]
        _w_track, _w_safe, _w_timing, _w_pattern, _alpha_track, _alpha_safe, _alpha_timing, _alpha_pattern, _beta = weights_override
        
    H = tau.shape[0]
    valid_len = jnp.minimum(valid_len, H)
    horizon = jnp.asarray(H_by_mode)[mode]
    mask = _window_mask(H, valid_len, horizon)
    active_steps = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)

    # Commands
    v_x_star = commands[0]
    v_y_star = commands[1]
    yaw_star = commands[2]

    # Mode-conditioned tolerances
    eps_vx = jnp.asarray(eps_vx_by_mode)[mode]
    eps_vy = jnp.asarray(eps_vy_by_mode)[mode]
    eps_yaw_m = jnp.asarray(eps_yaw_by_mode)[mode]

    # Foot index map for [FL, HL, FR, HR]
    FL, HL, FR, HR = 0, 1, 2, 3
    # ------------------------------------------------------------------
    # Shared safety terms
    # ------------------------------------------------------------------
    tau_margin = tau_max - jnp.abs(tau)          # (H, n_tau)
    tau_margin_worst_joint = jnp.min(tau_margin, axis=1)  # (H,)
    rho_torque = _masked_mean(tau_margin_worst_joint, mask)

    #rho_torque = _masked_min(tau_max - jnp.abs(tau), mask)

    current_contacts = c[-1]
    
    n_contacts = jnp.sum(c, axis=1)
    min_contacts_req = jnp.asarray(min_contacts_by_mode)[mode]
    rho_nlegs = _masked_min(n_contacts - min_contacts_req, mask)

    zmin = jnp.asarray(com_z_by_mode)[mode]
    rho_comz = _masked_min(com_z_hist - zmin, mask)

    vzmax = jnp.asarray(abs_vz_by_mode)[mode]
    rho_vz = _masked_min(vzmax - jnp.abs(v_hist[:, 2]), mask)

    roll_max = jnp.asarray(roll_abs_by_mode)[mode] * jnp.pi / 180.0
    pitch_max = jnp.asarray(pitch_abs_by_mode)[mode] * jnp.pi / 180.0
    rho_roll = _masked_min(roll_max - jnp.abs(roll_hist), mask)
    rho_pitch = _masked_min(pitch_max - jnp.abs(pitch_hist), mask)

    slipmax = jnp.asarray(slip_speed_by_mode)[mode]
    rho_slip = _masked_min(slipmax - slip_hist, mask)

    # Contact-centroid / CoM distance: only meaningful when at least one foot is in contact.
    denom = jnp.maximum(jnp.sum(c, axis=1, keepdims=True), 1.0)
    centroid = jnp.sum(c[:, :, None] * feet_xy, axis=1) / denom
    support_dist = jnp.linalg.norm(com_xy - centroid, axis=1)
    dmax = jnp.asarray(cop_com_xy_dist_by_mode)[mode]
    support_margin_per_t = jnp.where(n_contacts >= 2.0, dmax - support_dist, jnp.inf)
    
    #rho_support = _masked_min(support_margin_per_t, mask)
    
    valid_support = (n_contacts >= 2.0) & mask
    rho_support = jnp.where(jnp.any(valid_support),
                            _masked_min(support_margin_per_t, mask),
                            0.0)

    """rho_safety = smooth_min(
        [
            rho_torque / 5.0,
            rho_nlegs,
            rho_comz / 0.03,
            rho_vz / 0.10,
            rho_roll / (5.0 * jnp.pi / 180.0),
            rho_pitch / (5.0 * jnp.pi / 180.0),
            rho_slip / 0.20,
            rho_support / 0.06,
        ],
        beta=_beta,
    )"""

    # ------------------------------------------------------------------
    # Tracking terms
    # ------------------------------------------------------------------
    v_x_error_hist = jnp.abs(v_hist[:, 0] - v_x_star)
    v_y_error_hist = jnp.abs(v_hist[:, 1] - v_y_star)
    yaw_error_hist = jnp.abs(yaw_hist - yaw_star)

    rho_v_x = _masked_min(eps_vx - v_x_error_hist, mask)
    rho_v_y = _masked_min(eps_vy - v_y_error_hist, mask)
    rho_yaw = _masked_min(eps_yaw_m - yaw_error_hist, mask)

    rho_v_x_error = _masked_mean(v_x_error_hist, mask)
    rho_v_y_error = _masked_mean(v_y_error_hist, mask)
    rho_yaw_error = _masked_mean(yaw_error_hist, mask)

    # ------------------------------------------------------------------
    # Windowed gait features
    # ------------------------------------------------------------------
    gait_enabled = valid_len >= H_WARMUP_MIN_VALID

    c01 = c.astype(jnp.int32)
    td_FL = _touchdown_events(c01[:, FL], mask)
    td_HL = _touchdown_events(c01[:, HL], mask)
    td_FR = _touchdown_events(c01[:, FR], mask)
    td_HR = _touchdown_events(c01[:, HR], mask)

    td_counts = jnp.array([
        _event_count(td_FL),
        _event_count(td_HL),
        _event_count(td_FR),
        _event_count(td_HR),
    ])
    stride_sec_est = _mean_period_seconds(td_counts, active_steps)

    duty_leg = jnp.sum(c * mask[:, None].astype(jnp.float32), axis=0) / active_steps
    duty_est = jnp.mean(duty_leg)

    # Diagonal phase error (slow / trot)
    period_steps = jnp.maximum(stride_sec_est / DT, 1.0)
    e_diag1 = _inphase_event_error(td_FL, td_HR, period_steps)
    e_diag2 = _inphase_event_error(td_FR, td_HL, period_steps)
    e_diag = 0.5 * (e_diag1 + e_diag2)

    # Bound pair phase errors
    e_front = _inphase_event_error(td_FL, td_FR, period_steps)
    e_hind = _inphase_event_error(td_HL, td_HR, period_steps)

    # Pair-only states for bound priors
    front_only = (c01[:, FL] == 1) & (c01[:, FR] == 1) & (c01[:, HL] == 0) & (c01[:, HR] == 0)
    hind_only = (c01[:, HL] == 1) & (c01[:, HR] == 1) & (c01[:, FL] == 0) & (c01[:, FR] == 0)
    flight = n_contacts == 0.0

    p_front_only = _masked_mean(front_only.astype(jnp.float32), mask)
    p_hind_only = _masked_mean(hind_only.astype(jnp.float32), mask)
    p_flight = _masked_mean(flight.astype(jnp.float32), mask)
    p_2contact = _masked_mean((n_contacts == 2.0).astype(jnp.float32), mask)

    # Diagonal purity among 2-contact states
    mask2 = n_contacts == 2.0
    diag2 = ((c01[:, FL] == 1) & (c01[:, HR] == 1) & (c01[:, HL] == 0) & (c01[:, FR] == 0)) | \
            ((c01[:, FR] == 1) & (c01[:, HL] == 1) & (c01[:, FL] == 0) & (c01[:, HR] == 0))
    two_contact_count = jnp.sum((mask & mask2).astype(jnp.float32))
    diag2_frac = jnp.where(
        two_contact_count > 0.0,
        jnp.sum((mask & mask2 & diag2).astype(jnp.float32)) / two_contact_count,
        0.0,
    )

    # Hind-pair to front-pair lag for bound priors (events on pair-only states)
    td_front_pair = _touchdown_events(front_only.astype(jnp.int32), mask)
    td_hind_pair = _touchdown_events(hind_only.astype(jnp.int32), mask)
    lag_h_to_f = _forward_lag(td_hind_pair, td_front_pair, period_steps)


    # Slow-mode helper: require occasional 3+ support inside the last K steps.
    K = jnp.minimum(valid_len, K_REQUIRE_3PLUS)
    idx = jnp.arange(H)
    lastK_mask = mask & (idx >= (H - K))
    event_vals = jnp.where(lastK_mask, n_contacts - 3.0, -jnp.inf)
    rho_3plus_event = jnp.max(event_vals)

    # ------------------------------------------------------------------
    # Gait robustness per mode
    # ------------------------------------------------------------------
    stride_bounds = jnp.asarray(stride_period_by_mode)
    duty_bounds = jnp.asarray(duty_factor_by_mode)
    diag_err_max = jnp.asarray(diag_phase_error_by_mode)[mode]
    diag2_min = jnp.asarray(diag_2contact_fraction_min_by_mode)[mode]
    diag2_max = jnp.asarray(diag_2contact_fraction_max_by_mode)[mode]
    p2_min = jnp.asarray(contact2_fraction_min_by_mode)[mode]
    flight_min = jnp.asarray(flight_fraction_min_by_mode)[mode]
    front_only_min = jnp.asarray(front_only_fraction_min_by_mode)[mode]
    hind_only_min = jnp.asarray(hind_only_fraction_min_by_mode)[mode]
    pair_phase_max = jnp.asarray(pair_phase_error_max_by_mode)[mode]
    lag_bounds = jnp.asarray(hind_to_front_lag_by_mode)

    """stride_lo = stride_bounds[mode, 0]
    stride_hi = stride_bounds[mode, 1]
    rho_stride = jnp.minimum(stride_sec_est - stride_lo, stride_hi - stride_sec_est)"""

    """duty_lo = duty_bounds[mode, 0]
    duty_hi = duty_bounds[mode, 1]
    rho_duty = jnp.minimum(duty_est - duty_lo, duty_hi - duty_est)"""

    rho_diag_phase = diag_err_max - e_diag
    rho_diag2 = jnp.where(mode == MODE_BOUND, diag2_max - diag2_frac, diag2_frac - diag2_min)

    rho_p2 = p_2contact - p2_min
    rho_front = pair_phase_max - e_front
    rho_hind = pair_phase_max - e_hind

    lag_lo = lag_bounds[mode, 0]
    lag_hi = lag_bounds[mode, 1]
    rho_hindfront_raw = jnp.minimum(lag_h_to_f - lag_lo, lag_hi - lag_h_to_f)
    rho_hindfront = jnp.where(
        (mode == MODE_BOUND) & gait_enabled,
        rho_hindfront_raw,
        0.0,
    )

    rho_flight = p_flight - flight_min
    rho_front_only = p_front_only - front_only_min
    rho_hind_only = p_hind_only - hind_only_min

    # Mode branches
    
    # Normalized grouped robustness
    # ------------------------------------------------------------------
    rho_safety = smooth_min_sign_preserving(
        [
            _safe_div(rho_torque, tau_margin_scale),
            _safe_div(rho_nlegs, min_contacts_margin_scale),
            _safe_div(rho_comz, com_z_margin_scale),
          #  _safe_div(rho_vz, abs_vz_margin_scale),
            _safe_div(rho_roll, roll_margin_scale_deg * jnp.pi / 180.0),
            _safe_div(rho_pitch, pitch_margin_scale_deg * jnp.pi / 180.0),
           # _safe_div(rho_slip, slip_margin_scale),
           # _safe_div(rho_support, support_margin_scale),
        ],
        beta=_beta,
    )

    tvx, tvy, tyaw = track_axis_weights
    rho_tracking = (
        tvx * _safe_div(rho_v_x, eps_vx)
        + tvy * _safe_div(rho_v_y, eps_vy)
        + tyaw * _safe_div(rho_yaw, eps_yaw_m)
    ) / jnp.maximum(tvx + tvy + tyaw, 1e-6)

    stride_scale = jnp.asarray(stride_margin_scale_by_mode)[mode]
    duty_scale = jnp.asarray(duty_margin_scale_by_mode)[mode]
    """rho_timing = smooth_min_sign_preserving(
        [
            _safe_div(rho_stride, stride_scale),
            _safe_div(rho_duty, duty_scale),
        ],
        beta=_beta,
    )"""

    diag_phase_scale = jnp.asarray(diag_phase_margin_scale_by_mode)[mode]
    diag2_scale = jnp.asarray(diag2_margin_scale_by_mode)[mode]
    p2_scale = jnp.asarray(contact2_margin_scale_by_mode)[mode]
    pair_phase_scale = jnp.asarray(pair_phase_margin_scale_by_mode)[mode]
    hindfront_scale = jnp.asarray(hindfront_margin_scale_by_mode)[mode]
    flight_scale = jnp.asarray(flight_margin_scale_by_mode)[mode]
    front_only_scale = jnp.asarray(front_only_margin_scale_by_mode)[mode]
    hind_only_scale = jnp.asarray(hind_only_margin_scale_by_mode)[mode]
    
    rho_walk = smooth_min_sign_preserving(
        [
            _safe_div(rho_diag_phase, diag_phase_scale),            # diagonals not on the same phase - error
            _safe_div(rho_diag2, diag2_scale),                      # number of diagonals / number of 2-contacts
            _safe_div(rho_3plus_event, event3plus_margin_scale),    # number of 3 contacts
        ],
        beta=_beta,
    )

    rho_trot = smooth_min_sign_preserving(
        [
            _safe_div(rho_diag_phase, diag_phase_scale),
            _safe_div(rho_diag2, diag2_scale),    
            _safe_div(rho_p2, p2_scale),                            # number of 2 contacts
        ],
        beta=_beta,
    )

    rho_bound = smooth_min_sign_preserving(
        [
            _safe_div(rho_front, pair_phase_scale),                 # front legs not on the same phase - error
            _safe_div(rho_hind, pair_phase_scale),                  # hind legs not on the same phase - error
            _safe_div(rho_hindfront, hindfront_scale),
            _safe_div(rho_flight, flight_scale),
            _safe_div(rho_front_only, front_only_scale),           # number of 2 contacts
            _safe_div(rho_hind_only, hind_only_scale),             # number of 2 contacts
            _safe_div(rho_diag2, diag2_scale),                     # number of related pairs / number of 2-contacts
        ],
        beta=4.0,
    )
    
    """
    rho_walk = smooth_min(
        [
            rho_stride / 0.05,
            rho_duty / 0.05,
            rho_diag_phase / 0.05,
            rho_diag2 / 0.05,
            rho_3plus_event,
        ],
        beta=_beta,
    )

    rho_trot = smooth_min(
        [
            rho_stride / 0.04,
            rho_duty / 0.03,
            rho_diag_phase / 0.05,
            rho_diag2 / 0.03,
            rho_p2 / 0.10,
        ],
        beta=_beta,
    )

    rho_bound = smooth_min(
        [
            rho_stride / 0.04,
            rho_duty / 0.05,
            rho_front / 0.05,
            rho_hind / 0.05,
            rho_hindfront / 0.10,
            rho_flight / 0.05,
            rho_front_only / 0.05,
            rho_hind_only / 0.05,
            rho_diag2 / 0.05,
        ],
        beta=_beta,
    ) """

    """rho_gait = jnp.where(mode == MODE_WALK, rho_walk,
                 jnp.where(mode == MODE_TROT, rho_trot, rho_bound)) """
                 
    rho_pattern = jnp.where(mode == MODE_WALK, rho_walk,
                    jnp.where(mode == MODE_TROT, rho_trot, rho_bound))
    rho_pattern = jnp.where(gait_enabled, rho_pattern, 0.0)
    
    rho_timing = 0.0

    # Keep legacy aggregate name for logging.
    rho_gait = smooth_min_sign_preserving([rho_timing, rho_pattern], beta=_beta)
    
    rho_gait = jnp.where(gait_enabled, rho_gait, 0.0)

    # ------------------------------------------------------------------
    # Effort and final reward
    # ------------------------------------------------------------------
    tau_sq_sum = jnp.sum(jnp.square(tau), axis=1)
    tau_effort = _masked_mean(tau_sq_sum, mask)

    """r = (
        tanh_norm(rho_safety, _alpha_b)
        + _w_vx * tanh_norm(rho_v_x, _alpha_vx)
        + _w_vy * tanh_norm(rho_v_y, _alpha_vy)
        + _w_yaw * tanh_norm(rho_yaw, _alpha_yaw)
        + _w_gait * tanh_norm(rho_gait, _alpha_gait)
        - gamma_tau * tau_effort
    )"""
    
    r = (
          _w_safe * tanh_norm(rho_safety, _alpha_safe)
        + _w_track * tanh_norm(rho_tracking, _alpha_track)
       # + _w_timing * tanh_norm(rho_timing, _alpha_timing)
       # + jnp.where(mode == MODE_BOUND, 0.2 * tanh_norm(rho_duty, _alpha_timing), 0.0)
        + _w_pattern * tanh_norm(rho_pattern, _alpha_pattern)
        - gamma_tau * tau_effort
    )
    
    rho_stride = 0.0
    rho_duty = 0.0

    return (
        r,
        tau_effort,
        rho_safety,
        rho_torque,
        rho_comz,
        rho_roll,
        rho_pitch,
        rho_slip,
        rho_bound,
        rho_trot,
        rho_walk,
        rho_v_x,
        rho_v_y,
        rho_yaw,
        rho_nlegs,
        rho_gait,
        rho_v_x_error,
        rho_v_y_error,
        rho_yaw_error,
        rho_diag2,
        rho_stride,
        rho_duty,
        rho_3plus_event,
        rho_support,
        rho_diag_phase,
        rho_p2,
        rho_tracking,
        rho_timing,
        rho_pattern,
        rho_front,
        rho_hind,
        rho_hindfront,
        rho_flight,
        rho_front_only,
        rho_hind_only,
        pitch_hist[-1],
        roll_hist[-1],
        current_contacts[0],    # Returns [FL, HL, FR, HR]
        current_contacts[1],
        current_contacts[2],
        current_contacts[3],
       # slip_hist[-1], 
       # denom[-1][0],
       # support_dist[-1],
    )
