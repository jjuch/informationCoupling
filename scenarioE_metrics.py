import numpy as np
from furuta_model import rhs_continuous, rk4_step, b_theta_true
from coupling_metrics import structural_coupling_metrics, compute_theta_binned_maps



def compute_structural_series(base, cfg, p):
    """
    Compute S_lin(t), S_nonlin(t) along trajectory using structural_coupling_metrics().
    """
    t = base["t"]
    X = base["X"]
    U = base["U"]

    dt = float(cfg["dt"])
    n_sub = int(cfg["n_sub"])
    eps = float(cfg["coupling"]["eps"])
    norm = cfg["coupling"]["norm"]
    perturb = cfg["coupling"]["perturb"]

    kappa = float(cfg["_case_key"]["kappa"])
    G_shape = cfg["_case_key"]["G_shape"]

    N = len(t)
    S_lin = np.full(N, np.nan, float)
    S_non = np.full(N, np.nan, float)

    b0_true, b1_true = p.b0_nom, p.b1_nom
    b_th = lambda th: b_theta_true(th, b0_true, b1_true)

    for i in range(N):
        out = structural_coupling_metrics(
            X[i], U[i], dt, n_sub, p, kappa,
            rhs_continuous, rk4_step,
            eps=eps, norm=norm, perturb=perturb, G_shape=G_shape, b_theta_true=b_th
        )
        S_lin[i] = out["S_lin"]
        S_non[i] = out["S_nonlin"]

    return S_lin, S_non

def compute_theta_binned_structural_maps(base, cfg, S_lin, S_non):
    nbins = int(cfg["coupling"]["nbins"])
    # no EKF info_gain in scenario E, pass zeros
    info_gain = np.zeros_like(base["t"], dtype=float)

    maps = compute_theta_binned_maps(
        base["t"], base["X"], base["U"], S_lin, S_non, info_gain,
        nbins=nbins, theta_wrap=True, theta_min=-np.pi, theta_max=np.pi,
        t_min=cfg.get("T_burn", None), t_max=None
    )
    return maps
