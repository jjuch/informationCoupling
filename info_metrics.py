# info_metrics.py
import numpy as np

def _safe_logdet(S, jitter=1e-12):
    """
    Robust log(det(S)) for (near) SPD matrices. Adds jitter if needed.
    """
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)
    n = S.shape[0]
    for j in [jitter, 1e-10, 1e-8, 1e-6]:
        Sj = S + j * np.eye(n)
        sign, ld = np.linalg.slogdet(Sj)
        if sign > 0 and np.isfinite(ld):
            return ld
    # If still not positive, return nan
    return np.nan

def logdet_theta_block(P, idx_theta=(1, 3), jitter=1e-12):
    """
    logdet of EKF covariance block for (theta, thetadot).
    """
    P = np.asarray(P, dtype=float)
    block = P[np.ix_(idx_theta, idx_theta)]
    return _safe_logdet(block, jitter=jitter)

def build_lag_matrix(X, k):
    """
    X: (T, d) time series. Return targets Y (T-k, d) and regressor lags Z (T-k, d*k).
    """
    X = np.asarray(X, dtype=float)
    T, d = X.shape
    if T <= k:
        raise ValueError("Time series too short for lag order.")
    Y = X[k:]
    Z = np.hstack([X[k - i - 1: T - i - 1] for i in range(k)])
    return Y, Z

def residual_cov(Y, Z):
    """
    Multivariate OLS residual covariance.
    """
    B_hat, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    E = Y - Z @ B_hat
    return (E.T @ E) / E.shape[0]

def te_logdet(nu1, nu2, k, ridge=1e-10):
    """
    Vector TE_{2->1} via nested VAR on nu1 with and without nu2 lags.
    Returns (te, Sigma_red, Sigma_full).
    Adds 'ridge' to covariance matrices for numerical robustness

    """
    nu1 = np.asarray(nu1, dtype=float)
    nu2 = np.asarray(nu2, dtype=float)
    if nu1.shape[0] != nu2.shape[0]:
        raise ValueError("nu1 and nu2 must have same length.")

    Y1, Z1 = build_lag_matrix(nu1, k)
    _,  Z2 = build_lag_matrix(nu2, k)

    Sig_red  = residual_cov(Y1, Z1)
    Sig_full = residual_cov(Y1, np.hstack([Z1, Z2]))
    
    # ridge for PD logdet stability
    Sig_red  = Sig_red  + ridge * np.eye(Sig_red.shape[0])
    Sig_full = Sig_full + ridge * np.eye(Sig_full.shape[0])

    te = 0.5 * (_safe_logdet(Sig_red) - _safe_logdet(Sig_full))
    return te, Sig_red, Sig_full

def te_value(nu1, nu2, k, ridge=1e-10):
    te, _, _ = te_logdet(nu1, nu2, k, ridge=ridge)
    return float(te)

def time_to_steady_fraction(logdet_series, dt, frac=0.05, tail_seconds=2.0):
    """
    Time to reach within 'frac' of the steady-state level, per trial.

    - logdet_series: array (N,)
    - frac: 0.05 means 95% of the way from initial to steady
    - tail_seconds: how much of the tail to compute steady median

    Returns: time in seconds or nan if never reached.
    """
    logdet_series = np.asarray(logdet_series, dtype=float)
    N = len(logdet_series)
    tail_N = max(1, int(tail_seconds / dt))
    tail = logdet_series[-tail_N:]
    steady = np.nanmedian(tail)

    l0 = logdet_series[0]
    if not np.isfinite(steady) or not np.isfinite(l0):
        return np.nan

    target = steady + frac * (l0 - steady)
    idx = np.where(logdet_series <= target)[0]
    return (idx[0] * dt) if len(idx) else np.nan


def auc_logdet(logdet_series, dt):
    """
    Area under curve of logdet over time. Lower is better.
    Uses trapezoidal rule.
    """
    logdet_series = np.asarray(logdet_series, dtype=float)
    return float(np.trapezoid(logdet_series, dx=dt))


def windowed_te_series(nu1, nu2, k, dt, window_seconds=2.0, step_seconds=0.2, ridge=1e-10, start_idx=0):
    """
    Compute a time series of TE_{2->1} using sliding windows.

    nu1, nu2 : arrays shape (T, d1), (T, d2)
      Destination and source multivariate time series.
    k : int
      VAR order (lag).
    dt : float
      Sampling time of nu1/nu2 series (use the update sampling if you only log updates).
    window_seconds : float
      Window length in seconds.
    step_seconds : float
      Step between windows in seconds.
    ridge : float
      Ridge for covariance stability in te_logdet.
    start_idx : int
      Index to start computing TE (e.g. after transient warm-up).

    Returns:
      t_mid : (M,) window center times
      te    : (M,) TE values (nan if window too short)
    """
    nu1 = np.asarray(nu1, float)
    nu2 = np.asarray(nu2, float)

    T = min(nu1.shape[0], nu2.shape[0])
    nu1 = nu1[:T]
    nu2 = nu2[:T]

    w = max(1, int(round(window_seconds / dt)))
    s = max(1, int(round(step_seconds / dt)))


    t_mid = []
    te_vals = []

    for a in range(start_idx, T - w + 1, s):
        b = a + w
        seg1 = nu1[a:b]
        seg2 = nu2[a:b]
        if seg1.shape[0] <= k + 5:
            t_mid.append((a + b - 1) * 0.5 * dt)
            te_vals.append(np.nan)
            continue
        try:
            te, _, _ = te_logdet(seg1, seg2, k, ridge=ridge)
            te_vals.append(float(te))
        except Exception:
            te_vals.append(np.nan)
        t_mid.append((a + b - 1) * 0.5 * dt)

    return np.asarray(t_mid), np.asarray(te_vals)
