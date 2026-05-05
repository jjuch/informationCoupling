# scenarioF_debug_plots.py
import numpy as np
import matplotlib.pyplot as plt

from config import FurutaParams
from furuta_model import wrap_center, b_theta_true

def theta_dev(theta):
    """eta = wrap(theta - pi) in (-pi, pi]."""
    return wrap_center(theta, np.pi) - np.pi

def theta_energy_about_pi(eta, thetadot, beta, delta):
    """Energy-like scalar for rocking about theta=pi."""
    return 0.5 * beta * thetadot**2 + delta * (1.0 - np.cos(eta))

def _get(base, key, default=None):
    return base[key] if (key in base and base[key] is not None) else default


def _safe_corr(a, b):
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if a.size != b.size or a.size < 3:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def moving_average(x, win):
    x = np.asarray(x, float)
    win = int(max(1, win))
    kern = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kern, mode="same")


def compute_debug_signals(base, cfg, p: FurutaParams):
    """
    Ensures a consistent set of debug series are available.
    Works whether you stored them in `base` or not.
    """
    t = np.asarray(_get(base, "t"))
    X = np.asarray(_get(base, "X"))  # (N,4)
    U = np.asarray(_get(base, "U"))  # (N,)
    V = np.asarray(_get(base, "V", np.zeros_like(U)))  # (N,)

    phi = X[:, 0]
    theta = X[:, 1]
    phidot = X[:, 2]
    thetadot = X[:, 3]

    # eta
    eta = _get(base, "eta", None)
    if eta is None:
        eta = theta_dev(theta)
    eta = np.asarray(eta, float)

    # Energy target from amplitude A
    Aamp = float(cfg.get("theta", {}).get("A", 0.0))
    E_star_scalar = float(p.delta * (1.0 - np.cos(Aamp)))
    Etheta_star = _get(base, "Etheta_star", None)
    if Etheta_star is None:

        Etheta_star = np.full_like(thetadot, E_star_scalar, dtype=float)
    Etheta_star = np.asarray(Etheta_star, float)

    # Etheta and Eerr
    Etheta = _get(base, "Etheta", None)
    if Etheta is None:
        Etheta = theta_energy_about_pi(eta, thetadot, p.beta, p.delta)
    Etheta = np.asarray(Etheta, float)

    Eerr = _get(base, "Eerr", None)
    if Eerr is None:
        Eerr = Etheta - E_star_scalar
    Eerr = np.asarray(Eerr, float)

    # v decomposition: v_s, v_energy, v_align, gate, headroom
    v_energy = _get(base, "v_energy", None)
    if v_energy is None:
        # best-effort compute if kE exists
        kE = float(cfg.get("orbit", {}).get("kE", 0.0))
        v_energy = -kE * Eerr * thetadot
    v_energy = np.asarray(v_energy, float)

    v_align = _get(base, "v_align", None)
    if v_align is None:
        # might not be logged; set zeros
        v_align = np.zeros_like(V)
    v_align = np.asarray(v_align, float)

    v_s = _get(base, "v_s", None)
    if v_s is None:
        # infer residual (not perfect if v_extra was saturated)
        v_s = V - v_energy - v_align
    v_s = np.asarray(v_s, float)

    gate = _get(base, "gate", None)
    if gate is None:
        gate = np.zeros_like(V)
    gate = np.asarray(gate, float)

    headroom = _get(base, "headroom", None)
    if headroom is None:
        headroom = np.full_like(V, np.nan)
    headroom = np.asarray(headroom, float)

    # u_raw and u_sat
    u_raw = _get(base, "u_raw", None)
    if u_raw is None:
        u_raw = np.full_like(U, np.nan)
    u_raw = np.asarray(u_raw, float)

    u_sat = _get(base, "u_sat", None)
    if u_sat is None:
        u_sat = U
    u_sat = np.asarray(u_sat, float)

    # Power accounting
    P_in = _get(base, "P_in", None)
    if P_in is None:
        P_in = u_sat * phidot
    P_in = np.asarray(P_in, float)

    b0_true, b1_true = p.b0_nom, p.b1_nom
    bth = b_theta_true(theta, b0_true, b1_true)  # vectorized
    P_d_theta = _get(base, "P_d_theta", None)
    if P_d_theta is None:
        P_d_theta = bth * (thetadot ** 2)
    P_d_theta = np.asarray(P_d_theta, float)

    P_d_phi = _get(base, "P_d_phi", None)

    if P_d_phi is None:
        P_d_phi = p.b_phi * (phidot ** 2)
    P_d_phi = np.asarray(P_d_phi, float)

    # cumulative works
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    W_in = np.cumsum(P_in) * dt
    W_d_theta = np.cumsum(P_d_theta) * dt
    W_d_phi = np.cumsum(P_d_phi) * dt

    return dict(
        t=t, dt=dt,
        phi=phi, theta=theta, phidot=phidot, thetadot=thetadot, eta=eta,
        U=u_sat, u_raw=u_raw, u_sat=u_sat, V=V,
        Etheta=Etheta, Etheta_star=Etheta_star, Eerr=Eerr,
        v_s=v_s, v_energy=v_energy, v_align=v_align, gate=gate, headroom=headroom,
        P_in=P_in, P_d_theta=P_d_theta, P_d_phi=P_d_phi,
        W_in=W_in, W_d_theta=W_d_theta, W_d_phi=W_d_phi,
    )


# -----------------------------
# Debug summary printer
# -----------------------------

def print_debug_summary(base, cfg, label=""):
    """
    Prints an updated debug summary that includes:
    - u_max, saturation ratio
    - mean injected power
    - cumulative energy balance
    - corr(u, phidot)
    - corr(v_align, phidot) and gate stats
    """
    p = FurutaParams()
    dbg = compute_debug_signals(base, cfg, p)

    u_max = float(cfg.get("limits", {}).get("u_max", np.nan))

    u_raw = dbg["u_raw"]
    u_sat = dbg["u_sat"]
    sat_frac = float(np.mean(np.isfinite(u_raw) & (np.abs(u_sat - u_raw) > 1e-9)))

    P_in = dbg["P_in"]
    mean_Pin = float(np.mean(P_in))
    mean_Pin_last = float(np.mean(P_in[int(0.5 * len(P_in)):]))

    W_in = dbg["W_in"]
    W_d_theta = dbg["W_d_theta"]
    W_d_phi = dbg["W_d_phi"]

    corr_uphi = _safe_corr(dbg["U"], dbg["phidot"])
    corr_ualign = _safe_corr(dbg["v_align"], dbg["phidot"])  # proxy alignment relation

    gate = dbg["gate"]
    gate_mean = float(np.nanmean(gate))
    gate_max = float(np.nanmax(gate))

    print("\n[DEBUG SUMMARY]" + (f" {label}" if label else ""))
    print(f"u_max = {u_max}")
    print(f"fraction steps where u_sat != u_raw (approx sat) = {sat_frac:.3f}")
    print(f"mean(P_in) overall = {mean_Pin:.6f}")
    print(f"mean(P_in) second half = {mean_Pin_last:.6f}")
    print(f"W_in(end) = {W_in[-1]:.6f}, W_d_theta(end) = {W_d_theta[-1]:.6f}, W_d_phi(end) = {W_d_phi[-1]:.6f}")
    print(f"corr(u, phidot) = {corr_uphi:.6f}")
    print(f"corr(v_align, phidot) = {corr_ualign:.6f}")
    print(f"gate mean = {gate_mean:.6f}, gate max = {gate_max:.6f}\n")
    theta_share = float(dbg["W_d_theta"][-1] / max(1e-12, dbg["W_in"][-1]))
    phi_share   = float(dbg["W_d_phi"][-1]   / max(1e-12, dbg["W_in"][-1]))
    print(f"W_d_theta/W_in = {theta_share:.3f}, W_d_phi/W_in = {phi_share:.3f}")
    
    print(f"gate max = {np.nanmax(dbg['gate']):.6f}, gate mean = {np.nanmean(dbg['gate']):.6f}")
    print(f"headroom min = {np.nanmin(dbg['headroom']):.6f}, headroom mean = {np.nanmean(dbg['headroom']):.6f}")
    print(f"max|v_align| = {np.nanmax(np.abs(dbg['v_align'])):.6f}, max|v_energy| = {np.nanmax(np.abs(dbg['v_energy'])):.6f}")



# -----------------------------
# Plots
# -----------------------------


def plot_debug_bundle(base, cfg, meta_title="Scenario F debug", save_path=None, show=True):
    """
    Produce a 4-panel debug figure and optionally save to disk.
    
    4-panel bundle:
      (1) eta and thetadot
      (2) Etheta, E*, Eerr
      (3) v terms: v, v_s, v_energy, v_align + gate (twin axis)
      (4) power terms

    """
 
    p = FurutaParams()
    dbg = compute_debug_signals(base, cfg, p)
    t = dbg["t"]

    fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # 1) eta and thetadot
    axs[0].plot(t, dbg["eta"], lw=1.3, label="eta = wrap(theta-pi)")
    axs[0].plot(t, dbg["thetadot"], lw=1.0, label="thetadot")
    axs[0].axhline(0.0, color="k", lw=0.8, alpha=0.3)
    axs[0].set_ylabel("eta, thetadot")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="upper right")

    # 2) Energy
    axs[1].plot(t, dbg["Etheta"], lw=1.3, label="Etheta")
    axs[1].plot(t, dbg["Etheta_star"], lw=1.0, ls="--", label="E*")
    axs[1].plot(t, dbg["Eerr"], lw=1.0, ls=":", label="Etheta - E*")
    axs[1].axhline(0.0, color="k", lw=0.8, alpha=0.3)
    axs[1].set_ylabel("Energy")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="upper right")


    # 3) v decomposition + gate
    axs[2].plot(t, dbg["V"], lw=1.2, label="v (total)")
    axs[2].plot(t, dbg["v_s"], lw=1.0, ls="--", label="v_s (s-stab)")
    axs[2].plot(t, dbg["v_energy"], lw=1.0, label="v_energy")
    axs[2].plot(t, dbg["v_align"], lw=1.0, label="v_align")
    axs[2].axhline(0.0, color="k", lw=0.8, alpha=0.3)
    axs[2].set_ylabel("v terms")
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(loc="upper right")

    ax2b = axs[2].twinx()
    ax2b.plot(t, dbg["gate"], lw=0.8, color="gray", alpha=0.7, label="gate")
    ax2b.set_ylabel("gate", color="gray")
    ax2b.tick_params(axis="y", labelcolor="gray")

    # 4) Power
    axs[3].plot(t, dbg["P_in"], lw=1.2, label="P_in = u*phidot")
    axs[3].plot(t, dbg["P_d_theta"], lw=1.0, label="P_d_theta = b(theta)*thetadot^2")
    axs[3].plot(t, dbg["P_d_phi"], lw=1.0, ls="--", label="P_d_phi = b_phi*phidot^2")
    axs[3].axhline(0.0, color="k", lw=0.8, alpha=0.3)
    axs[3].set_ylabel("Power")
    axs[3].set_xlabel("time [s]")
    axs[3].grid(True, alpha=0.25)
    axs[3].legend(loc="upper right")

    fig.suptitle(meta_title, y=0.995)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[SAVE DEBUG PLOT] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def save_debug_bundle_png(base, cfg, out_dir, tag, meta_title):
    """
    Convenience function: saves debug plot as PNG with a standard filename.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"F_debug_{tag}.png"
    path = os.path.join(out_dir, fname)
    plot_debug_bundle(base, cfg, meta_title=meta_title, save_path=path, show=False)
    return path

def plot_debug_actuation_and_energy_integrals(base, cfg, meta_title="Scenario F energy balance debug",
                                             save_path=None, show=True):
    """
    Extra debug figure:
    (1) u_raw vs u_sat and u_max
    (2) instantaneous power terms
    (3) cumulative injected/dissipated work
    (4) moving-average injected power
    """
    p = FurutaParams()
    dbg = compute_debug_signals(base, cfg, p)
    t = dbg["t"]
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    U = dbg["U"]
    phidot = dbg["phidot"]
    thetadot = dbg["thetadot"]
    theta = dbg["theta"]

    # u_raw / u_sat
    u_raw = _get(base, "u_raw", None)
    if u_raw is None:
        u_raw = np.full_like(U, np.nan)
    else:
        u_raw = np.asarray(u_raw, dtype=float)

    u_sat = _get(base, "u_sat", None)
    if u_sat is None:
        # if not explicitly logged, assume U is u_sat
        u_sat = U
    else:
        u_sat = np.asarray(u_sat, dtype=float)

    u_max = float(cfg.get("limits", {}).get("u_max", np.nan))
    u_max_line = np.full_like(U, u_max)

    # instantaneous powers
    P_in = dbg["P_in"]
    P_d_theta = dbg["P_d_theta"]
    P_d_phi = dbg["P_d_phi"]

    # cumulative integrals
    W_in = np.cumsum(P_in) * dt
    W_d_theta = np.cumsum(P_d_theta) * dt
    W_d_phi = np.cumsum(P_d_phi) * dt

    # moving average of P_in
    win_sec = 0.5
    win = max(1, int(win_sec / dt))
    P_in_ma = np.convolve(P_in, np.ones(win)/win, mode="same")

    # summary stats
    frac_sat = np.mean(np.isfinite(u_raw) & (np.abs(u_sat - u_raw) > 1e-9))
    mean_Pin = float(np.mean(P_in))
    mean_Pin_last = float(np.mean(P_in[int(0.5*len(P_in)):]))

    
    # correlation diagnostic (robust to scale)
    if np.std(U) > 1e-12 and np.std(phidot) > 1e-12:
        corr_uphi = float(np.corrcoef(U, phidot)[0, 1])
    else:
        corr_uphi = float("nan")


    # print("\n[DEBUG SUMMARY]")
    # print(f"u_max = {u_max}")
    # print(f"fraction steps where u_sat != u_raw (approx sat) = {frac_sat:.3f}")
    # print(f"mean(P_in) overall = {mean_Pin:.6f}")
    # print(f"mean(P_in) second half = {mean_Pin_last:.6f}")
    # print(f"W_in(end) = {W_in[-1]:.6f}, W_d_theta(end) = {W_d_theta[-1]:.6f}, W_d_phi(end) = {W_d_phi[-1]:.6f}\n")
    # print(f"corr(u, phidot) = {corr_uphi:.6f}")
    print_debug_summary(base, cfg)

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # (1) u raw/sat
    axs[0].plot(t, u_raw, lw=1.0, label="u_raw")
    axs[0].plot(t, u_sat, lw=1.2, label="u_sat (applied)")
    axs[0].plot(t, u_max_line, lw=0.9, ls="--", color="k", alpha=0.5, label="u_max")
    axs[0].plot(t, -u_max_line, lw=0.9, ls="--", color="k", alpha=0.5)
    axs[0].set_ylabel("u")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="upper right")

    # (2) powers
    axs[1].plot(t, P_in, lw=1.2, label="P_in = u*phidot")
    axs[1].plot(t, P_d_theta, lw=1.0, label="P_d_theta")
    axs[1].plot(t, P_d_phi, lw=1.0, ls="--", label="P_d_phi")
    axs[1].axhline(0.0, color="k", lw=0.8, alpha=0.3)
    axs[1].set_ylabel("Power")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="upper right")

    # (3) cumulative works
    axs[2].plot(t, W_in, lw=1.2, label="W_in = ∫P_in dt")
    axs[2].plot(t, W_d_theta, lw=1.0, label="W_d_theta = ∫P_d_theta dt")
    axs[2].plot(t, W_d_phi, lw=1.0, ls="--", label="W_d_phi = ∫P_d_phi dt")
    axs[2].axhline(0.0, color="k", lw=0.8, alpha=0.3)
    axs[2].set_ylabel("Cumulative work")
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(loc="upper right")

    # (4) moving average of P_in
    axs[3].plot(t, P_in, lw=0.8, alpha=0.35, label="P_in")
    axs[3].plot(t, P_in_ma, lw=1.4, label=f"P_in moving avg ({win_sec}s)")
    axs[3].axhline(0.0, color="k", lw=0.8, alpha=0.3)
    axs[3].set_ylabel("P_in")
    axs[3].set_xlabel("time [s]")
    axs[3].grid(True, alpha=0.25)
    axs[3].legend(loc="upper right")

    fig.suptitle(meta_title, y=0.995)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[SAVE DEBUG ENERGY] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
