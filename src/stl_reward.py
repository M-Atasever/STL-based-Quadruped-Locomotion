import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import mujoco
from coeff_config import tau_max, Fz_min, eps_v, eps_yaw, mu, beta
from coeff_config import alpha_b, w_vx, alpha_vx, w_vy, alpha_vy, w_yaw, alpha_yaw, alpha_torque
from coeff_config import w_nlegs, alpha_nlegs, w_torque, gamma_tau, w_tau, alpha_tau


def smooth_min(vals, beta=10.0):
    vals = jnp.asarray(vals)                      
    return -jnp.log(jnp.sum(jnp.exp(-beta * vals))) / beta

def tanh_norm(rho, alpha):  # bound to [-1,1]
    return jnp.tanh(rho / alpha)

def softsign_p(x, p=2.0):
    return x / jnp.power(1.0 + jnp.abs(x)**p, 1.0/p)

def dist_in_poly(y, poly, valid_count):  # signed; positive inside
    # poly: list of 2D vertices counterclockwise
    # poly: (N,2) but only the first valid_count rows are real vertices.
    
    """if len(poly) < 3:
        return -1e6
    dists = []
    n = len(poly)
    for i in range(n):
        a = np.array(poly[i])
        b = np.array(poly[(i+1)%n])
        edge = b - a
        n_in = np.array([ -edge[1], edge[0] ])  # CCW inward normal
        n_in = n_in / (np.linalg.norm(n_in)+1e-12)
        dists.append( np.dot(n_in, (np.array(y)-a)) )
    return min(dists)"""

    N = poly.shape[0]

    def compute(_):
        idx = jnp.arange(N, dtype=jnp.int32)
        idx_next = jnp.where(idx + 1 < valid_count, idx + 1, jnp.int32(0))

        a = poly[idx]        # (N,2)
        b = poly[idx_next]   # (N,2)
        edge = b - a         # (N,2)

        # CCW inward normal [-ey, ex]
        n_in = jnp.stack([-edge[:, 1], edge[:, 0]], axis=1)                 # (N,2)
        n_in = n_in / (jnp.linalg.norm(n_in, axis=1, keepdims=True) + 1e-12)

        # Distance of point y to each supporting line (inward normal)
        d = jnp.einsum('ij,ij->i', n_in, (y[None, :] - a))                  # (N,)

        # Ignore padded edges by setting them to +inf before min
        d = jnp.where(idx < valid_count, d, jnp.inf)
        return jnp.min(d)  

    # If we don't have at least a line, return a large negative sentinel
    return lax.cond(valid_count >= 2, compute,
        lambda _: jnp.array(-1e6, dtype=poly.dtype),operand=None,)
    

def find_first_real_vertex(polygon, max_vertices):
    # Your loop's initial state. Here, just the index 'count'.
    init_count = 0

    # 1. Condition Function:
    #    Takes the loop state (count) and returns True to continue looping.
    def cond_fun(count):
        # Check 1: Are we still a zero vertex?
        # (jnp.all is a cleaner way to write your check)
        is_zero_vertex = jnp.all(polygon[count] == 0)
        
        # Check 2: Are we still in bounds? (IMPORTANT!)
        is_in_bounds = count < max_vertices
        
        # Continue WHILE both are true
        return jnp.logical_and(is_zero_vertex, is_in_bounds)

    # 2. Body Function:
    #    Takes the loop state (count) and returns the *new* state.
    def body_fun(count):
        # Just increment the count
        return count + 1

    # 3. Run the JAX-native while loop
    #    This will run `body_fun` as long as `cond_fun` returns True.
    first_real_index = jax.lax.while_loop(cond_fun, body_fun, init_count)

    return first_real_index


def point_in_polygon_not_used(pt, poly):
    # https://www.geeksforgeeks.org/dsa/how-to-check-if-a-given-point-lies-inside-a-polygon/
    # Ray casting algorithm
    
    polygon = poly
    num_vertices = 4  # polygon.shape[0]
    x, y = pt
    inside = False

    count = find_first_real_vertex(polygon, num_vertices)
    p1 = polygon[count]
    
    polygon = jnp.vstack((polygon, p1))

    count_host = jax.device_get(count) 
    i = count_host.item()
    while i < num_vertices:
        count = find_first_real_vertex(polygon[i+1:,:], num_vertices-i)
        count_host = jax.device_get(count)
        i = i + count_host.item() + 1
        p2 = polygon[i]
            
        # Check if the point is above the minimum y coordinate of the edge
        if y > min(p1[1], p2[1]):
            # Check if the point is below the maximum y coordinate of the edge
            if y <= max(p1[1], p2[1]):
                # Check if the point is to the left of the maximum x coordinate of the edge
                if x <= max(p1[0], p2[0]):
                    # Calculate the x-intersection of the line connecting the point to the edge
                    x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]

                    # Check if the point is on the same line as the edge or to the left of the x-intersection
                    if p1[0] == p2[0] or x <= x_intersection:
                        # Flip the inside flag
                        inside = not inside

        p1 = p2

    return inside

def _is_zero_rows(arr, tol=1e-8):
    # Row is treated as zero if all coordinates are ~0
    return jnp.all(jnp.isclose(arr, 0.0, atol=tol), axis=1)

def _first_true(mask):
    # index of first True; 0 if none True
    n = mask.shape[0]
    idx = jnp.arange(n)
    ranked = jnp.where(mask, idx, idx + n)
    return jnp.argmin(ranked)

def _next_true(mask, i):
    # next index > i (cyclic) where mask[j] is True
    n = mask.shape[0]
    steps = jnp.arange(1, n + 1)
    js = (i + steps) % n
    ok = mask[js]
    ranked = jnp.where(ok, steps, steps + n)
    step = jnp.argmin(ranked) + 1
    return (i + step) % n

def _ordered_indices_of_valid_vertices(valid):
    # Return up to 4 indices in cyclic order starting at first True.
    start = _first_true(valid)
    def body(i, _):
        j = _next_true(valid, i)
        return j, j
    # Collect 3 subsequent indices (max polygon size = 4)
    _, rest = lax.scan(body, start, xs=None, length=3)  # (3,)
    return jnp.concatenate([jnp.array([start]), rest], axis=0)  # (4,)

def point_in_polygon(pt, poly, tol_zero=1e-8, tol_edge=1e-8):
    """
    Tri-valued point-in-polygon for a (4,3) array with zero-row masking.

    Returns:
      +1.0 if pt is inside OR on an edge,
      -1.0 if pt is outside,
       0.0 if fewer than 3 valid vertices (2+ rows are zero).
    """
    xy = poly[:, :2]                    # use x,y only
    valid = ~_is_zero_rows(poly, tol_zero)   # (4,)
    K = jnp.sum(valid)                        # number of real vertices

    def compute(_):
        idxs = _ordered_indices_of_valid_vertices(valid)   # (4,)
        verts = xy[idxs]                                   # (4,2)
        # Edge list uses wrap-around
        v1 = verts
        v2 = jnp.roll(verts, -1, axis=0)

        # Only first K edges are real (others are placeholders)
        edge_mask = jnp.arange(4) < K

        x, y = pt[0], pt[1]
        x1, y1 = v1[:, 0], v1[:, 1]
        x2, y2 = v2[:, 0], v2[:, 1]
        dx, dy = x2 - x1, y2 - y1

        # --- 1) Explicit on-edge test (robust to floating error) ---
        cross = (y - y1) * dx - (x - x1) * dy     # 2D cross product
        colinear = jnp.abs(cross) <= tol_edge
        dot = (x - x1) * dx + (y - y1) * dy
        seg_len2 = dx * dx + dy * dy
        within = (dot >= -tol_edge) & (dot <= seg_len2 + tol_edge)
        on_edge = edge_mask & colinear & within

        any_on_edge = jnp.any(on_edge)
        def on_edge_branch(_):
            return jnp.array(1.0, xy.dtype)

        # --- 2) Even–odd (ray casting) with strict '<' (boundary handled above) ---
        def inside_outside(_):
            straddles = jnp.logical_xor(y1 > y, y2 > y)
            # denom nonzero when straddles; safe to divide
            denom = y2 - y1
            x_intersect = (x2 - x1) * (y - y1) / denom + x1
            crosses = edge_mask & straddles & (x < x_intersect)
            inside = jnp.logical_xor.reduce(crosses)
            return jnp.where(inside, jnp.array(1.0, xy.dtype),
                                     jnp.array(-1.0, xy.dtype))

        return lax.cond(any_on_edge, on_edge_branch, inside_outside, operand=None)

    # Not enough vertices → 0.0
    return lax.cond(K >= 3,
                    compute,
                    lambda _: jnp.array(0.0, xy.dtype),
                    operand=None)


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

def friction_cone_margin(model, data, Fz_min=Fz_min, delta=0.0):
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
    return worst if worst < np.inf else -1e6


def reward_step(model, data, input, commands):
   
    # 1-) Torque Limits 
    # 2-) Center of Mass within the Support Polygon for Static Stability
    # 3-) Increased Size of Support Polygon (more legs on the ground)
    # 4-) Zero Moment Point (ZMP) within the Support Polygon for Dynamic Stability
    # 5-) Velocity tracking
    # 6-) Heading tracking
    # 7-) Contact force limits due to friction cone constraints
    # NOT USED! Equal to 4 8-) Center of pressure (CoP) remaining in the support polygon constraints  (SAME WITH #4 ??)

    # mujoco already considers friction cone constraints https://mujoco.readthedocs.io/en/stable/computation/index.html

                        
    tau = input["tau_history"]              
    com_xy = input["CoM_history"]       
    
    c = input["contact_history"]           
    poly = input["feet_history"]       
    
    v_hist = input["lin_velocity_history"] 
    yaw = input["ang_velocity_history"] 
    
    v_x_star = commands[0] 
    v_y_star = commands[1] 
    heading_star = commands[2] 
    
    # F --> (fx,fy,fz)
    cop_xy, Fz, F, _ = cop_from_cfrc_ext(model, data)
    
    
    # R1: Torque limits
    rho_torque = jnp.min(tau_max - jnp.abs(tau))
    
    # R2: CoM in S
    #flag = point_in_polygon(com_xy, poly) # -1, 0, +1
    #rho_com = tanh_norm(flag, 0.5)
    
    # R3: More legs on the ground
    rho_nlegs = jnp.sum(c.astype(jnp.float32), axis=1) - 2
    rho_nlegs = jnp.min(rho_nlegs)

    # R4: ZMP/CoP in S
    #flag2 = point_in_polygon(cop_xy, poly)
    #rho_cop = tanh_norm(flag2, 0.5)
    #rho_zmp = min(Fz - Fz_min, rho_cop)
    #rho_zmp = jnp.minimum(Fz - Fz_min, rho_cop)

    # R5: Velocity tracking
    v_x_error_hist = jnp.abs(v_hist[:, 0] - v_x_star)
    rho_v_x = jnp.min(eps_v - v_x_error_hist)
    v_y_error_hist = jnp.abs(v_hist[:, 1] - v_y_star)
    rho_v_y = jnp.min(eps_v - v_y_error_hist)

    # R6: Heading tracking
    yaw_error_hist = jnp.abs(yaw - heading_star)
    rho_yaw = jnp.min(eps_yaw - yaw_error_hist)
    
    # Error Tracking During Training
    rho_v_x_error = jnp.min(v_x_error_hist)
    rho_v_y_error = jnp.min(v_y_error_hist)
    rho_yaw_error = jnp.min(yaw_error_hist)

    # R7: Friction cones (optinal)  
    #rho_cone = friction_cone_margin(model, data, Fz_min, delta=.5)

    
    # Safety 
    rho_safety = smooth_min(vals=[rho_torque]) # rho_torque

    # Penalty for effort/energy
    tau_joint = jnp.sum(jnp.square(input["tau_history"]), axis=1)
    tau_dot = jnp.mean(tau_joint)  # mean over the temporal horizon (if you wanna be strict, use max)

    # Rewards
    
    #r = (0.1*rho_safety + rho_v_x + rho_v_y + rho_yaw + 0.5*rho_nlegs + 0.1*rho_torque - gamma_tau*jnp.dot(tau, tau))
    
    
    r = (
        1.0*(tanh_norm(rho_safety, alpha_b)+1)
        + w_vx*(tanh_norm(rho_v_x, alpha_vx)+1)
        + w_vy*(tanh_norm(rho_v_y, alpha_vy)+1)
        + w_yaw*(tanh_norm(rho_yaw, alpha_yaw)+1)
        #+ w_torque*tanh_norm(rho_torque, alpha_torque)
        + w_nlegs*(tanh_norm(rho_nlegs, alpha_nlegs)+1)
        #- gamma_tau*tau_dot
        - w_tau*(tanh_norm(tau_dot, alpha_tau)+1)
    )
    
    
    """r = (
        1.0*tanh_norm(rho_safety, alpha_b)
        + w_vx*tanh_norm(rho_v_x, alpha_vx)
        + w_vy*tanh_norm(rho_v_y, alpha_vy)
        + w_yaw*tanh_norm(rho_yaw, alpha_yaw)
        #+ w_torque*tanh_norm(rho_torque, alpha_torque)
        + w_nlegs*tanh_norm(rho_nlegs, alpha_nlegs)
        #- gamma_tau*tau_dot
        - w_tau*tanh_norm(tau_dot, alpha_tau)
    )"""
    
    """r = (
        1.0*softsign_p(rho_safety)
        + w_vx*softsign_p(rho_v_x)
        + w_vy*softsign_p(rho_v_y)
        + w_yaw*softsign_p(rho_yaw)
        + w_torque*softsign_p(rho_torque)
        + w_nlegs*softsign_p(rho_nlegs)
        - gamma_tau*jnp.dot(tau, tau)
    )"""
    
    return r, tau_dot, rho_safety, rho_torque, rho_v_x, rho_v_y, rho_yaw, \
           rho_nlegs, rho_v_x_error, rho_v_y_error, rho_yaw_error

