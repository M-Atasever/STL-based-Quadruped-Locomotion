import jax
import jax.numpy as jnp

from coeff_config import (
    tau_max,
    beta,
    beta_safe,
    beta_timing,
    beta_pattern,
    gamma_tau,
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
    front_only_fraction_max_by_mode,
    hind_only_fraction_max_by_mode,
    all4_fraction_max_by_mode,
    all4_fraction_min_by_mode,
    pair_front_mismatch_max_by_mode,
    pair_hind_mismatch_max_by_mode,
    hind_to_front_lag_by_mode,
    K_REQUIRE_3PLUS,
    H_by_mode,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    H_WARMUP_MIN_VALID,
    DT,
    clearance_min_by_mode,
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
    pair_mismatch_margin_scale_by_mode,
    hindfront_margin_scale_by_mode,
    flight_margin_scale_by_mode,
    front_only_margin_scale_by_mode,
    hind_only_margin_scale_by_mode,
    all4_margin_scale_by_mode,
    event3plus_margin_scale,
    bound_event_margin_scale,
    clearance_margin_scale_by_mode,
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


def _interval_robustness(x, lo, hi):
    return jnp.minimum(x - lo, hi - x)


def _touchdown_events(signal, mask):
    signal = signal > 0.5
    prev = jnp.concatenate([jnp.array([False]), signal[:-1]])
    idx = jnp.arange(signal.shape[0])
    first_valid = jnp.argmax(mask.astype(jnp.int32))
    prev = jnp.where(idx == first_valid, False, prev)
    return signal & (~prev) & mask


def _event_count(events):
    return jnp.sum(events.astype(jnp.float32))


def _mean_period_seconds(event_counts, valid_steps):
    valid_steps = jnp.maximum(valid_steps, 1.0)
    has_evt = event_counts > 0.0
    period_steps = valid_steps / jnp.maximum(event_counts, 1.0)
    denom = jnp.maximum(jnp.sum(has_evt.astype(jnp.float32)), 1.0)
    return jnp.sum(jnp.where(has_evt, period_steps, 0.0)) / denom * DT


def _inphase_event_error(events_a, events_b, period_steps):
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
    return jnp.where(jnp.any(select), err, 1.0)


def _forward_lag(events_src, events_dst, period_steps):
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


def _mean_pair_mismatch(sig_a, sig_b, mask):
    mismatch = jnp.abs(sig_a.astype(jnp.float32) - sig_b.astype(jnp.float32))
    return _masked_mean(mismatch, mask)


def _optional_clearance_robustness(reward_input, contacts, mask, mode):
    clearance_hist = reward_input.get("clearance_history", None)
    if clearance_hist is None:
        return jnp.array(0.0, dtype=jnp.float32)

    clearance_hist = jnp.asarray(clearance_hist, dtype=jnp.float32)
    swing_mask = mask[:, None] & (contacts < 0.5)
    cmin = jnp.asarray(clearance_min_by_mode)[mode]
    margins = clearance_hist - cmin
    any_swing = jnp.any(swing_mask)
    return jnp.where(any_swing, _masked_min(margins, swing_mask), 0.0)


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


def reward_step(reward_input, commands, mode, valid_len, weights_override=None):
    tau = reward_input["tau_history"]                          # (H, n_tau)
    c = reward_input["contact_history"].astype(jnp.float32)    # (H,4)
    feet_xy = reward_input["feet_history"]                     # (H,4,2)
    com_xy = reward_input["CoM_history"]                       # (H,2)
    v_hist = reward_input["lin_velocity_history"]              # (H,3)
    yaw_hist = reward_input["ang_velocity_history"]            # (H,)
    com_z_hist = reward_input["com_z_history"]                 # (H,)
    roll_hist = reward_input["roll_history"]                   # (H,)
    pitch_hist = reward_input["pitch_history"]                 # (H,)
    slip_hist = reward_input["slipmax_history"]                # (H,)

    if weights_override is None:
        (
            _w_track,
            _w_safe,
            _w_timing,
            _w_pattern,
            _alpha_track,
            _alpha_safe,
            _alpha_timing,
            _alpha_pattern,
            _beta_override,
        ) = _default_group_params(mode)
    else:
        (
            _w_track,
            _w_safe,
            _w_timing,
            _w_pattern,
            _alpha_track,
            _alpha_safe,
            _alpha_timing,
            _alpha_pattern,
            _beta_override,
        ) = weights_override

    H = tau.shape[0]
    valid_len = jnp.minimum(valid_len, H)
    horizon = jnp.asarray(H_by_mode)[mode]
    mask = _window_mask(H, valid_len, horizon)
    active_steps = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
    gait_enabled = valid_len >= H_WARMUP_MIN_VALID

    v_x_star = commands[0]
    v_y_star = commands[1]
    yaw_star = commands[2]

    eps_vx = jnp.asarray(eps_vx_by_mode)[mode]
    eps_vy = jnp.asarray(eps_vy_by_mode)[mode]
    eps_yaw_m = jnp.asarray(eps_yaw_by_mode)[mode]

    FL, HL, FR, HR = 0, 1, 2, 3
    current_contacts = c[-1]

    # ------------------------------------------------------------------
    # Shared safety terms
    # ------------------------------------------------------------------
    tau_margin = tau_max - jnp.abs(tau)
    tau_margin_worst_joint = jnp.min(tau_margin, axis=1)
    rho_torque = _masked_mean(tau_margin_worst_joint, mask)

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
    valid_support = (n_contacts >= 2.0) & mask
    rho_support = jnp.where(jnp.any(valid_support), _masked_min(support_margin_per_t, mask), 0.0)

    rho_safety = smooth_min_sign_preserving(
        [
            _safe_div(rho_torque, tau_margin_scale),
          #  _safe_div(rho_nlegs, min_contacts_margin_scale),
            _safe_div(rho_comz, com_z_margin_scale),
           # _safe_div(rho_vz, abs_vz_margin_scale),
            _safe_div(rho_roll, roll_margin_scale_deg * jnp.pi / 180.0),
            _safe_div(rho_pitch, pitch_margin_scale_deg * jnp.pi / 180.0),
          #  _safe_div(rho_slip, slip_margin_scale),
           # _safe_div(rho_support, support_margin_scale),
        ],
        beta=beta_safe,)

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

    tvx, tvy, tyaw = track_axis_weights
    rho_tracking = (
        tvx * _safe_div(rho_v_x, eps_vx)
        + tvy * _safe_div(rho_v_y, eps_vy)
        + tyaw * _safe_div(rho_yaw, eps_yaw_m)
    ) / jnp.maximum(tvx + tvy + tyaw, 1e-6)

    # ------------------------------------------------------------------
    # Windowed gait features
    # ------------------------------------------------------------------
    c01 = c.astype(jnp.int32)

    td_FL = _touchdown_events(c01[:, FL], mask)
    td_HL = _touchdown_events(c01[:, HL], mask)
    td_FR = _touchdown_events(c01[:, FR], mask)
    td_HR = _touchdown_events(c01[:, HR], mask)

    td_counts = jnp.array([
        _event_count(td_FL),
        _event_count(td_HL),
        _event_count(td_FR),
        _event_count(td_HR),])
    stride_sec_est = _mean_period_seconds(td_counts, active_steps)
    period_steps = jnp.maximum(stride_sec_est / DT, 1.0)

    duty_leg = jnp.sum(c * mask[:, None].astype(jnp.float32), axis=0) / active_steps
    duty_est = jnp.mean(duty_leg)

    e_diag1 = _inphase_event_error(td_FL, td_HR, period_steps)
    e_diag2 = _inphase_event_error(td_FR, td_HL, period_steps)
    e_diag = 0.5 * (e_diag1 + e_diag2)

    front_pair = (c01[:, FL] == 1) & (c01[:, FR] == 1)
    hind_pair = (c01[:, HL] == 1) & (c01[:, HR] == 1)
    front_only = front_pair & (~hind_pair)
    hind_only = hind_pair & (~front_pair)
    all4 = front_pair & hind_pair
    flight = n_contacts == 0.0

    p_front_only = _masked_mean(front_only.astype(jnp.float32), mask)
    p_hind_only = _masked_mean(hind_only.astype(jnp.float32), mask)
    p_flight = _masked_mean(flight.astype(jnp.float32), mask)
    p_all4 = _masked_mean(all4.astype(jnp.float32), mask)
    p_2contact = _masked_mean((n_contacts == 2.0).astype(jnp.float32), mask)

    mask2 = n_contacts == 2.0
    diag2 = (((c01[:, FL] == 1) & (c01[:, HR] == 1) & (c01[:, HL] == 0) & (c01[:, FR] == 0))
        | ((c01[:, FR] == 1) & (c01[:, HL] == 1) & (c01[:, FL] == 0) & (c01[:, HR] == 0)))
    two_contact_count = jnp.sum((mask & mask2).astype(jnp.float32))
    diag2_frac = jnp.where(two_contact_count > 0.0,
        jnp.sum((mask & mask2 & diag2).astype(jnp.float32)) / two_contact_count,
        0.0,)

    front_pair_mismatch = _mean_pair_mismatch(c01[:, FL], c01[:, FR], mask)
    hind_pair_mismatch = _mean_pair_mismatch(c01[:, HL], c01[:, HR], mask)

    td_front_pair = _touchdown_events(front_pair.astype(jnp.int32), mask)
    td_hind_pair = _touchdown_events(hind_pair.astype(jnp.int32), mask)
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
    lag_bounds = jnp.asarray(hind_to_front_lag_by_mode)

    stride_lo = stride_bounds[mode, 0]
    stride_hi = stride_bounds[mode, 1]
    duty_lo = duty_bounds[mode, 0]
    duty_hi = duty_bounds[mode, 1]
    lag_lo = lag_bounds[mode, 0]
    lag_hi = lag_bounds[mode, 1]

    rho_stride = _interval_robustness(stride_sec_est, stride_lo, stride_hi)
    rho_duty = _interval_robustness(duty_est, duty_lo, duty_hi)

    diag_err_max = jnp.asarray(diag_phase_error_by_mode)[mode]
    diag2_min = jnp.asarray(diag_2contact_fraction_min_by_mode)[mode]
    diag2_max = jnp.asarray(diag_2contact_fraction_max_by_mode)[mode]
    p2_min = jnp.asarray(contact2_fraction_min_by_mode)[mode]
    flight_min = jnp.asarray(flight_fraction_min_by_mode)[mode]
    front_only_min = jnp.asarray(front_only_fraction_min_by_mode)[mode]
    hind_only_min = jnp.asarray(hind_only_fraction_min_by_mode)[mode]
    front_only_max = jnp.asarray(front_only_fraction_max_by_mode)[mode]
    hind_only_max = jnp.asarray(hind_only_fraction_max_by_mode)[mode]

    all4_max = jnp.asarray(all4_fraction_max_by_mode)[mode]
    all4_min = jnp.asarray(all4_fraction_min_by_mode)[mode]
    front_mismatch_max = jnp.asarray(pair_front_mismatch_max_by_mode)[mode]
    hind_mismatch_max = jnp.asarray(pair_hind_mismatch_max_by_mode)[mode]

    rho_diag_phase = diag_err_max - e_diag
    rho_diag2 = jnp.where(mode == MODE_BOUND, diag2_max - diag2_frac, diag2_frac - diag2_min)
    rho_p2 = p_2contact - p2_min
    rho_front = front_mismatch_max - front_pair_mismatch
    rho_hind = hind_mismatch_max - hind_pair_mismatch
    rho_hindfront = jnp.where(
        lag_h_to_f >= 0.0,
        _interval_robustness(lag_h_to_f, lag_lo, lag_hi),
        0.0,
    )
    #rho_front_only = jnp.minimum(p_front_only - front_only_min, front_only_max - p_front_only)
    #rho_hind_only  = jnp.minimum(p_hind_only  - hind_only_min,  hind_only_max  - p_hind_only)

    rho_flight = p_flight - flight_min
    rho_front_only = p_front_only - front_only_min
    rho_hind_only = p_hind_only - hind_only_min
    #rho_all4 = all4_max - p_all4
    rho_all4 = jnp.minimum(p_all4 - all4_min, all4_max - p_all4)
    rho_bound_event = jnp.minimum(_event_count(td_front_pair), _event_count(td_hind_pair)) - 1.0
    rho_clearance = _optional_clearance_robustness(reward_input, c, mask, mode)

    stride_scale = jnp.asarray(stride_margin_scale_by_mode)[mode]
    duty_scale = jnp.asarray(duty_margin_scale_by_mode)[mode]
    rho_timing = smooth_min_sign_preserving(
        [
            _safe_div(rho_stride, stride_scale),
            _safe_div(rho_duty, duty_scale),
        ],
        beta=beta_timing,
    )

    diag_phase_scale = jnp.asarray(diag_phase_margin_scale_by_mode)[mode]
    diag2_scale = jnp.asarray(diag2_margin_scale_by_mode)[mode]
    p2_scale = jnp.asarray(contact2_margin_scale_by_mode)[mode]
    pair_mismatch_scale = jnp.asarray(pair_mismatch_margin_scale_by_mode)[mode]
    hindfront_scale = jnp.asarray(hindfront_margin_scale_by_mode)[mode]
    flight_scale = jnp.asarray(flight_margin_scale_by_mode)[mode]
    front_only_scale = jnp.asarray(front_only_margin_scale_by_mode)[mode]
    hind_only_scale = jnp.asarray(hind_only_margin_scale_by_mode)[mode]
    all4_scale = jnp.asarray(all4_margin_scale_by_mode)[mode]
    clearance_scale = jnp.asarray(clearance_margin_scale_by_mode)[mode]

    rho_walk = smooth_min_sign_preserving(
        [
            _safe_div(rho_diag_phase, diag_phase_scale),             # diagonals not on the same phase - error
            _safe_div(rho_diag2, diag2_scale),                       # number of diagonals / number of 2-contacts
            _safe_div(rho_3plus_event, event3plus_margin_scale),     # number of 3 contacts
         #   _safe_div(rho_clearance, clearance_scale),
        ],
        beta=beta_pattern,
    )

    rho_trot = smooth_min_sign_preserving(
        [
            _safe_div(rho_diag_phase, diag_phase_scale),
            _safe_div(rho_diag2, diag2_scale),
            _safe_div(rho_p2, p2_scale),                             # number of 2 contacts
         #   _safe_div(rho_clearance, clearance_scale),
        ],
        beta=beta_pattern,
    )

    rho_bound = smooth_min_sign_preserving(
        [
          #  _safe_div(rho_front, pair_mismatch_scale),             # front legs not on the same phase - error
          #  _safe_div(rho_hind, pair_mismatch_scale),              # hind legs not on the same phase - error
          #  _safe_div(rho_hindfront, hindfront_scale),
            _safe_div(rho_flight, flight_scale),
            _safe_div(rho_front_only, front_only_scale),            # number of 2 contacts
            _safe_div(rho_hind_only, hind_only_scale),              # number of 2 contacts
            _safe_div(rho_all4, all4_scale),
            _safe_div(rho_diag2, diag2_scale),                      # number of related pairs / number of 2-contacts
            _safe_div(rho_bound_event, bound_event_margin_scale),
         #   _safe_div(rho_clearance, clearance_scale),
        ],
        beta=0.1,
    )

    rho_pattern = jnp.where(
        mode == MODE_WALK,
        rho_walk,
        jnp.where(mode == MODE_TROT, rho_trot, rho_bound),
    )
    rho_pattern = jnp.where(gait_enabled, rho_pattern, 0.0)
    rho_timing = jnp.where(gait_enabled, rho_timing, 0.0)

    rho_gait = smooth_min_sign_preserving([rho_timing, rho_pattern], beta=_beta_override)
    rho_gait = jnp.where(gait_enabled, rho_gait, 0.0)

    tau_sq_sum = jnp.sum(jnp.square(tau), axis=1)
    tau_effort = _masked_mean(tau_sq_sum, mask)

    r = ( _w_safe * tanh_norm(rho_safety, _alpha_safe)
        + _w_track * tanh_norm(rho_tracking, _alpha_track)
       #+ _w_timing * tanh_norm(rho_timing, _alpha_timing)
        + _w_pattern * tanh_norm(rho_pattern, _alpha_pattern)
        - gamma_tau * tau_effort
        )

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
        rho_all4,
        rho_bound_event,
        pitch_hist[-1],
        roll_hist[-1],
        current_contacts[0],
        current_contacts[1],
        current_contacts[2],
        current_contacts[3],
    )
