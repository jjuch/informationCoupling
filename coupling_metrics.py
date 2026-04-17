"""
Numerical structural and informational coupling metrics for the Furuta pendulum benchmark.

This module is intentionally *numerical* and *trajectory-local*:
- It avoids equilibrium-only linearization.
- Structural metrics are computed from local sensitivities (Jacobian blocks) and from finite perturbation gains.
- Informational metrics are computed from EKF covariance reduction (entropy reduction proxy) and optionally TE on EKF state-corrections.

Assumptions about project structure:
- furuta_model.py provides rhs_continuous(...) and rk4_step(...)
- ekf.py provides EKF class
- info_metrics.py provides te_logdet(...) if TE is requested

All angles should be treated modulo 2π; binning by theta uses wrapped theta in (-π, π] by default.
"""

from __future__ import annotations

import numpy as np
from furuta_model import wrap_angle


def bin_by_theta(theta: np.ndarray, nbins: int = 31, theta_min: float = -np.pi, theta_max: float = np.pi):
    """Return bin indices and bin centers for theta values."""
    theta = np.asarray(theta, dtype=float)
    edges = np.linspace(theta_min, theta_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(theta, edges) - 1
    idx = np.clip(idx, 0, nbins - 1)
    return idx, centers, edges


def robust_stat_per_bin(values: np.ndarray, bin_idx: np.ndarray, nbins: int, stat: str = 'median'):
    """Compute robust statistic per bin."""
    values = np.asarray(values, dtype=float)
    out = np.full(nbins, np.nan)
    lo = np.full(nbins, np.nan)
    hi = np.full(nbins, np.nan)
    for b in range(nbins):
        v = values[bin_idx == b]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        if stat == 'median':
            out[b] = np.median(v)
            lo[b] = np.percentile(v, 25)
            hi[b] = np.percentile(v, 75)
        elif stat == 'mean':
            out[b] = np.mean(v)
            lo[b] = np.percentile(v, 25)
            hi[b] = np.percentile(v, 75)

        else:
            raise ValueError('stat must be median or mean')
    return out, lo, hi


def plant_step_substeps(x, u, dt, n_sub, p, kappa, rhs_continuous, rk4_step, b_theta_true=None, G_shape='const'):
    """Integrate plant for dt using n_sub RK4 substeps."""
    x = np.asarray(x, dtype=float)
    dt_int = dt / int(n_sub)
    for _ in range(int(n_sub)):
        x = rk4_step(rhs_continuous, x, u, dt_int, p, kappa=float(kappa), G_shape=G_shape, b_theta_true=b_theta_true)
        if not np.all(np.isfinite(x)):
            break
    return x


def jacobian_discrete_step(x, u, dt, n_sub, p, kappa, rhs_continuous, rk4_step, eps=1e-6, G_shape='const'):
    """Numerical Jacobian of one-step map Phi(x,u) w.r.t. x using central differences."""
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((n, n), dtype=float)

    def Phi(xx):
        return plant_step_substeps(xx, u, dt, n_sub, p, kappa, rhs_continuous, rk4_step, G_shape=G_shape)

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        xp = Phi(x + dx)
        xm = Phi(x - dx)
        J[:, i] = (xp - xm) / (2.0 * eps)
    return J


def structural_coupling_metrics(x, u, dt, n_sub, p, kappa, rhs_continuous, rk4_step, eps=1e-6, norm='sv', perturb='basis', G_shape='const'):
    """Compute structural coupling at (x,u).

    Returns a dict with:
      - S_lin: norm of discrete Jacobian block d x1_next / d x2
      - S_nonlin: finite perturbation gain from x2 perturbations into x1_next

    Partition:
      x1 = (phi, phidot) indices [0,2]
      x2 = (theta, thetadot) indices [1,3]
    """
    x = np.asarray(x, dtype=float)

    J = jacobian_discrete_step(x, u, dt, n_sub, p, kappa, rhs_continuous, rk4_step, eps=eps, G_shape=G_shape)
    x1_idx = [0, 2]
    x2_idx = [1, 3]
    J12 = J[np.ix_(x1_idx, x2_idx)]

    if norm == 'fro':
        S_lin = float(np.linalg.norm(J12, ord='fro'))
    elif norm == 'sv':
        S_lin = float(np.linalg.svd(J12, compute_uv=False)[0])
    else:
        raise ValueError('norm must be fro or sv')

    # Nonlinear perturbation gain
    def Phi(xx):
        return plant_step_substeps(xx, u, dt, n_sub, p, kappa, rhs_continuous, rk4_step, G_shape=G_shape)

    x_next = Phi(x)
    x1_next = x_next[x1_idx]

    eps_p = eps
    gains = []

    if perturb == 'basis':
        # perturb theta and thetadot directions
        for j, idx in enumerate(x2_idx):
            d = np.zeros_like(x)
            d[idx] = eps_p
            xp = Phi(x + d)[x1_idx]
            xm = Phi(x - d)[x1_idx]
            g = np.linalg.norm(xp - xm) / (2.0 * eps_p)
            gains.append(g)
        S_nonlin = float(np.max(gains))
    elif perturb == 'random':
        # random direction in x2
        v = np.random.default_rng(0).normal(size=2)
        v = v / np.linalg.norm(v)
        d = np.zeros_like(x)
        d[x2_idx] = eps_p * v
        xp = Phi(x + d)[x1_idx]
        xm = Phi(x - d)[x1_idx]
        S_nonlin = float(np.linalg.norm(xp - xm) / (2.0 * eps_p))
    else:
        raise ValueError('perturb must be basis or random')

    return {'S_lin': S_lin, 'S_nonlin': S_nonlin}

def ekf_information_gain_step(P_pred, P_upd, idx_theta=(1, 3), jitter=1e-12):
    """Gaussian entropy reduction proxy (theta-block): 0.5 log det(P_pred)/det(P_upd)."""
    P_pred = np.asarray(P_pred, dtype=float)
    P_upd = np.asarray(P_upd, dtype=float)
    blk_pred = P_pred[np.ix_(idx_theta, idx_theta)]
    blk_upd = P_upd[np.ix_(idx_theta, idx_theta)]

    def logdet(M):
        M = 0.5 * (M + M.T)
        n = M.shape[0]
        sign, ld = np.linalg.slogdet(M + jitter * np.eye(n))
        if sign <= 0 or not np.isfinite(ld):
            sign, ld = np.linalg.slogdet(M + 1e-8 * np.eye(n))
        return ld if sign > 0 else np.nan

    ld_pred = logdet(blk_pred)
    ld_upd = logdet(blk_upd)
    return 0.5 * (ld_pred - ld_upd)

def compute_theta_binned_maps(time, X_true, U, S_lin, S_nonlin, info_gain, nbins=31, theta_wrap=True, theta_min=-np.pi, theta_max=np.pi, t_min=None, t_max=None):
    """Compute binned maps vs theta for given time series."""
    
    time = np.asarray(time, float)
    if t_min is not None or t_max is not None:
        m = np.ones_like(time, dtype=bool)
        if t_min is not None:
            m &= (time >= float(t_min))
        if t_max is not None:
            m &= (time <= float(t_max))
        # apply mask consistently
        time = time[m]
        X_true = np.asarray(X_true)[m]
        U = np.asarray(U)[m]
        S_lin = np.asarray(S_lin)[m]
        S_nonlin = np.asarray(S_nonlin)[m]
        info_gain = np.asarray(info_gain)[m]

    theta = X_true[:, 1]
    if theta_wrap:
        theta = wrap_angle(theta)

    bin_idx, centers, edges = bin_by_theta(theta, nbins=nbins, theta_min=theta_min, theta_max=theta_max)

    Slin_med, Slin_q1, Slin_q3 = robust_stat_per_bin(S_lin, bin_idx, nbins, stat='median')
    Snl_med,  Snl_q1,  Snl_q3  = robust_stat_per_bin(S_nonlin, bin_idx, nbins, stat='median')
    Ig_med,   Ig_q1,   Ig_q3   = robust_stat_per_bin(info_gain, bin_idx, nbins, stat='median')

    counts = np.array([(bin_idx == b).sum() for b in range(nbins)], dtype=int)

    return {
        'theta_centers': centers,
        'theta_edges': edges,
        'counts': counts,
        'S_lin_med': Slin_med,
        'S_lin_q1': Slin_q1,
        'S_lin_q3': Slin_q3,
        'S_nonlin_med': Snl_med,
        'S_nonlin_q1': Snl_q1,
        'S_nonlin_q3': Snl_q3,
        'I_med': Ig_med,
        'I_q1': Ig_q1,
        'I_q3': Ig_q3,
    }

def extract_theta_segments(theta, mask, min_len):
    """
    Extract contiguous index segments where mask is True.
    Returns a list of (start, end) inclusive indices.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    segments = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            if (prev - start + 1) >= min_len:
                segments.append((start, prev))
            start = i
            prev = i
    if (prev - start + 1) >= min_len:
        segments.append((start, prev))
    return segments

def concat_segments(X, segments):
    """
    Concatenate contiguous time segments from X (T,d) into one array (Tcat,d).
    """
    parts = []
    for a, b in segments:
        parts.append(X[a:b+1])
    if len(parts) == 0:
        return None
    return np.vstack(parts)


def theta_binned_te(dx_series, theta, nbins=31, theta_min=-np.pi, theta_max=np.pi, te_lag=3, min_count=20, min_seg_len=None, te_func=None, t=None, t_min=None, t_max=None):
    """
    Compute theta-binned TE on EKF state corrections.

    dx_series: (T,4) EKF corrections
    theta: (T,) angle values (should be wrapped consistently)
    te_func: callable(nu1,nu2,lag)->float, e.g. info_metrics.te_value or te_logdet(...)[0]

    We compute TE_{2->1} where:
      nu1 = dx[:, [0,2]] (phi, phidot corrections)
      nu2 = dx[:, [1,3]] (theta, thetadot corrections)

    For each theta bin:
      - build a mask of points in that bin
      - extract contiguous segments with sufficient length
      - concatenate segments
      - compute TE if total concatenated samples >= min_count

    Returns:
      centers, te_vals, counts
    """
    if te_func is None:
        raise ValueError("Provide te_func to compute TE (e.g., lambda a,b,k: te_logdet(a,b,k)[0]).")
    
    
    if t is not None and (t_min is not None or t_max is not None):
        t = np.asarray(t, float)
        m = np.ones_like(t, dtype=bool)
        if t_min is not None:
            m &= (t >= float(t_min))
        if t_max is not None:
            m &= (t <= float(t_max))
        dx_series = np.asarray(dx_series)[m]
        theta = np.asarray(theta)[m]

    theta = np.asarray(theta, float)
    dx = np.asarray(dx_series, float)
    if dx.shape[0] != theta.shape[0]: 
        raise ValueError("dx_series and theta length mismatch")

    # choose min_seg_len
    if min_seg_len is None:
        min_seg_len = max(10, te_lag + 5)

    # bins
    edges = np.linspace(theta_min, theta_max, nbins + 1)
    centers = 0.5*(edges[:-1] + edges[1:])
    bin_idx = np.digitize(theta, edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    te21 = np.full(nbins, np.nan)
    te12 = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)

    nu1_full = dx[:, [0,2]]
    nu2_full = dx[:, [1,3]]

    for b in range(nbins):
        mask = (bin_idx == b) & np.all(np.isfinite(dx), axis=1) & np.isfinite(theta)
        segments = extract_theta_segments(theta, mask, min_len=min_seg_len)
        nu1 = concat_segments(nu1_full, segments)
        nu2 = concat_segments(nu2_full, segments)
        if nu1 is None or nu2 is None:
            continue
        counts[b] = nu1.shape[0]

        if counts[b] >= min_count:
            try:
                te21[b] = float(te_func(nu1, nu2, te_lag))  # 2->1
                te12[b] = float(te_func(nu2, nu1, te_lag))  # 1->2

            except Exception:
                te21[b] = np.nan
                te12[b] = np.nan

    return centers, te21, te12, counts
