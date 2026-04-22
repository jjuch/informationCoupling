"""
run_scenario_D.py

Scenario D: Compare kappa > 0 against stored kappa=0 baseline (Scenario C)
using G_kappa(q) = kappa * sin(theta)^2 * S (shape="sin2").

- Loads baseline .npz/.json from furuta/data (ScenarioC_baseline_kappa0_seed{seed}.npz)
- Reuses baseline probe input u(t) for fair comparisons
- Runs kappas = {0.05, 0.10, 0.20} with identical settings (dt, update_period, noise flags, friction_uncertainty)
- Computes the same metrics as Scenario C:
  structural coupling maps, per-update info gain maps, theta-binned TE maps
- Saves per-kappa results and produces clean comparison plots:
  overlays and delta (kappa - baseline) small multiples.
- transient-aware metrics: evaluate θ-binned maps and TE on [t0, t0+Ttrans]
- windowed TE vs time: TE(t) computed on sliding windows (transient shows up)
- caching: if ScenarioD npz/json exists and metadata matches, load instead of rerun


Usage:
  python run_scenario_D.py
Optional flags:
  --baseline_npz path/to/ScenarioC_baseline_kappa0_seed1.npz
  --baseline_json path/to/ScenarioC_baseline_kappa0_seed1.json
"""

import os
import json
import hashlib
from pathlib import Path
import argparse
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FurutaParams, ExperimentConfig
from run_scenario_C import run_probe_rollout, compute_theta_binned_maps, plot_state_estimates_time, generate_multisine_u, make_damping_gif, plot_coeff_uncertainty_and_mean, plot_damping_function
from furuta_model import wrap_angle
from coupling_metrics import theta_binned_te
from info_metrics import te_logdet

data_subfolder = Path("nonlinear_50ms_sin_1plusCos")


# --------------------
# Utility: stable hash of probe input
# --------------------
def hash_array(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=np.float64)
    h = hashlib.sha1(a.tobytes()).hexdigest()
    return h

def _default_baseline_paths(cfg: ExperimentConfig):
    data_dir = Path("data") / data_subfolder
    npz = data_dir / f"ScenarioD_sin_1plusCos_kappa0_00_seed1.npz"
    js  = data_dir / f"ScenarioD_sin_1plusCos_kappa0_00_seed1.json"
    return npz, js


def meta_matches(meta: dict, cfg: ExperimentConfig, kappa: float, G_shape: str, measurement_dim: int, measurement_noise: bool, friction_uncertainty: bool,
                 u_hash: str) -> bool:
    """Return True if stored metadata is consistent with current requested run."""
    if not meta:
        return False
    checks = [
        ("kappa", float(kappa)),
        ("G_shape", G_shape),
        ("seed", int(cfg.seed)),
        ("dt", float(cfg.dt)),
        ("update_period", float(cfg.update_period)),
        ("T", float(cfg.T)),
        ("sigma_phi", float(cfg.sigma_phi)),
        ("te_lag", int(cfg.te_lag)),
        ("te_start_time", float(cfg.te_start_time)),
        ("measurement_dim", int(measurement_dim)),
        ("measurement_noise", bool(measurement_noise)),
        ("friction_uncertainty", bool(friction_uncertainty)),
        ("probe_u_hash", u_hash),
    ]
    for k, v in checks:
        if k not in meta:
            return False
        if meta[k] != v:
            return False
    return True


def _load_data(npz_path: Path, json_path: Path):
    base = np.load(npz_path, allow_pickle=True)
    base = {k: base[k] for k in base.files}
    meta = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    return base, meta


def safe_get(d, key):
    return d[key] if key in d else None

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def info_gain_coeff_series(Pcc_pred_hist, Pcc_upd_hist, eps=1e-12):
    K = Pcc_pred_hist.shape[0]
    out = np.full(K, np.nan)
    I = np.eye(Pcc_pred_hist.shape[1])
    for k in range(K):
        A = Pcc_pred_hist[k] + eps*I
        B = Pcc_upd_hist[k] + eps*I
        sa, la = np.linalg.slogdet(A)
        sb, lb = np.linalg.slogdet(B)
        if sa > 0 and sb > 0:
            out[k] = 0.5*(la - lb)
    return out


def _mask_by_count(x, y, q1, q3, counts, min_count):
    """Return masked arrays (finite and counts>=min_count)."""
    m = (counts >= min_count) & np.isfinite(y) & np.isfinite(q1) & np.isfinite(q3)
    return x[m], y[m], q1[m], q3[m], m


def _plot_overlay(theta, base_med, base_q1, base_q3,
                  curves, ylabel, title, fname, min_count, base_counts, curve_counts):
    """
    Overlay baseline + kappa curves.
    curves: list of dicts with keys: 'label', 'med','q1','q3','counts'
    """
    plt.figure(figsize=(10, 4.6))

    # Baseline (masked)
    xb, yb, lb, ub, mb = _mask_by_count(theta, base_med, base_q1, base_q3, base_counts, min_count)
    plt.plot(xb, yb, color="black", linewidth=2.0, label="baseline κ=0")
    plt.fill_between(xb, lb, ub, color="black", alpha=0.12)

    # Kappa curves (masked)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, c in enumerate(curves):
        col = colors[i % len(colors)]
        xk, yk, l, u, mk = _mask_by_count(theta, c["med"], c["q1"], c["q3"], c["counts"], min_count)
        plt.plot(xk, yk, color=col, linewidth=1.6, label=c["label"])
        plt.fill_between(xk, l, u, color=col, alpha=0.15)

    plt.xlabel("theta [rad] (wrapped)")
    plt.ylabel(ylabel)
    plt.title(title + f" (masked bins: count<{min_count})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def _plot_delta_small_multiples(theta, base_med, base_counts,
                               curves, ylabel, title, fname, min_count):
    """
    Plot Δ = curve - baseline for each kappa in separate subplots (1 row, K columns).
    """
    K = len(curves)
    fig, axs = plt.subplots(1, K, figsize=(4.2*K, 3.8), sharey=True, sharex=True)

    if K == 1:
        axs = [axs]

    # baseline mask
    base_mask = (base_counts >= min_count) & np.isfinite(base_med)

    for j, c in enumerate(curves):
        ax = axs[j]
        cur_mask = (c["counts"] >= min_count) & np.isfinite(c["med"])
        m = base_mask & cur_mask

        ax.plot(theta[m], (c["med"][m] - base_med[m]), linewidth=1.8)
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
        ax.set_title(c["label"])
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("theta [rad]")

    axs[0].set_ylabel(ylabel)
    fig.suptitle(title + f" (Δ vs baseline; masked bins: count<{min_count})", y=1.02)
    fig.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

def _plot_te_overlay(theta, base_te21, base_te12, base_counts,
                     curves, fname, min_count):
    """
    Two-panel plot: TE 2->1 and TE 1->2 overlays.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)

    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Panel 1: TE 2->1
    m0 = (base_counts >= min_count) & np.isfinite(base_te21)
    axs[0].plot(theta[m0], base_te21[m0], color="black", linewidth=2.0, label="baseline κ=0")
    for i, c in enumerate(curves):
        col = colors[i % len(colors)]
        mk = (c["te_counts"] >= min_count) & np.isfinite(c["te_2to1"])
        axs[0].plot(theta[mk], c["te_2to1"][mk], color=col, linewidth=1.6, label=c["label"])
    axs[0].set_title("TE 2→1 (θ-block → φ-block)")
    axs[0].set_ylabel("TE (log-det VAR score)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Panel 2: TE 1->2
    m1 = (base_counts >= min_count) & np.isfinite(base_te12)
    axs[1].plot(theta[m1], base_te12[m1], color="black", linewidth=2.0, label="baseline κ=0")
    for i, c in enumerate(curves):
        col = colors[i % len(colors)]
        mk = (c["te_counts"] >= min_count) & np.isfinite(c["te_1to2"])
        axs[1].plot(theta[mk], c["te_1to2"][mk], color=col, linewidth=1.6, label=c["label"])
    axs[1].set_title("TE 1→2 (φ-block → θ-block)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    for ax in axs:
        ax.set_xlabel("theta [rad] (wrapped)")

    fig.suptitle(f"Theta-binned directional TE overlays (masked bins: count<{min_count})", y=1.02)
    fig.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def plot_coeff_info_gain(t_updates, Pcc_pred_hist, Pcc_upd_hist, fname: str="D_coefficients_info_gain.png", label=""):
    if t_updates is None or Pcc_pred_hist is None or Pcc_upd_hist is None:
        print("[WARN] Missing fields for info gain:", fname)
        return

    dI = info_gain_coeff_series(Pcc_pred_hist, Pcc_upd_hist)
    cum = np.nancumsum(np.where(np.isfinite(dI), dI, 0.0))

    fig, axs = plt.subplots(2,1,figsize=(10,6.5),sharex=True)

    axs[0].plot(t_updates, dI)
    axs[0].set_ylabel("ΔI_c per update")
    axs[0].set_title(f"Parameter information gain {label}")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_updates, cum)
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("cumulative ΔI_c")
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("Saved:", fname)


def info_gain_coeff_series(Pcc_pred_hist, Pcc_upd_hist, eps=1e-12):
    Pcc_pred_hist = np.asarray(Pcc_pred_hist, float)
    Pcc_upd_hist  = np.asarray(Pcc_upd_hist, float)
    K = Pcc_pred_hist.shape[0]
    out = np.full(K, np.nan)
    I = np.eye(Pcc_pred_hist.shape[1])
    for k in range(K):
        A = Pcc_pred_hist[k] + eps*I
        B = Pcc_upd_hist[k] + eps*I
        sa, la = np.linalg.slogdet(A)
        sb, lb = np.linalg.slogdet(B)
        if sa > 0 and sb > 0:
            out[k] = 0.5*(la - lb)
    return out


def plot_dIc_vs_theta_updates(base: dict, fname: str="D_dIc_theta.png"):
    """
    Scatter ΔI_c,k vs θ_true at update instants.
    Requires: t, X_true, t_updates, Pcc_pred_hist, Pcc_upd_hist
    """
    t = base.get("t", None)
    X_true = base.get("X_true", None)
    t_updates = base.get("t_updates", None)
    Ppred = base.get("Pcc_pred_hist", None)
    Pupd  = base.get("Pcc_upd_hist", None)

    if any(v is None for v in (t, X_true, t_updates, Ppred, Pupd)):
        print("[WARN] Missing fields for ΔI_c vs θ plot")
        return

    t = np.asarray(t, float)
    X_true = np.asarray(X_true, float)
    t_updates = np.asarray(t_updates, float)

    dI = info_gain_coeff_series(Ppred, Pupd)

    # map update times to nearest indices
    idx = np.searchsorted(t, t_updates)
    idx = np.clip(idx, 0, len(t)-1)

    theta = X_true[idx, 1]
    theta = wrap_angle(theta)

    plt.figure(figsize=(8,4.8))
    plt.scatter(theta, dI, s=12, alpha=0.6)
    plt.xlabel("θ_true at update [rad] (wrapped)")
    plt.ylabel("ΔI_c per update")
    plt.title("Parameter information gain bursts vs configuration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved:", fname)


def plot_nis_and_Qcc(t_updates, nis_hist, Qcc_hist, fname=str, label=""):
    # NIS: Normalized Innovation Squared
    # innovation: v_k = y_k - y_k_hat^-
    # innovation covariance: S_k = H_kP_k^{-}H_k^T + R
    # NIS_k = nu_k^T S_k^-1 nu_k
    if t_updates is None or (nis_hist is None and Qcc_hist is None):
        print("[WARN] Missing NIS/Qcc for:", fname)
        return

    fig, axs = plt.subplots(2,1,figsize=(10,6),sharex=True)

    if nis_hist is not None:
        axs[0].plot(t_updates[:len(nis_hist)], nis_hist)
        axs[0].axhline(1.0, color="gray", lw=1, alpha=0.6)
        axs[0].set_ylabel("NIS")
        axs[0].set_title(f"NIS vs time {label}")
        axs[0].grid(True, alpha=0.3)
    else:
        axs[0].text(0.5,0.5,"nis_hist missing",ha="center",va="center")
        axs[0].set_axis_off()

    if Qcc_hist is not None:
        axs[1].plot(t_updates[:len(Qcc_hist)], Qcc_hist)
        axs[1].set_yscale("log")
        axs[1].set_ylabel("Qcc")
        axs[1].set_xlabel("time [s]")
        axs[1].set_title("Adaptive coefficient process noise")
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5,0.5,"Qcc_hist missing",ha="center",va="center")
        axs[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)

def load_or_run_case(
        *,
        p, cfg, data_dir: Path,
        kappa: float,
        G_shape: str,
        u_used: np.ndarray,
        seed: int,
        measurement_dim: int=1,
        measurement_noise: bool=True,
        friction_uncertainty: bool=True,
):
    """
    Load cached Scenario D run if metadata matches, otherwise run run_probe_rollout.
    Returns (base_dict, meta_dict).
    """
    # ---------- file naming ----------
    kappa_str = f"{kappa:.2f}".replace('.', '_')
    tag = f"{G_shape}_{kappa_str}"
    npz_path = data_dir / f"data_{tag}.npz"
    json_path = data_dir / f"data_{tag}.json"

    u_hash = hash_array(u_used)

    # ---------- try cache ----------
    if npz_path.exists() and json_path.exists():
        base, meta = _load_data(npz_path, json_path)
        if meta_matches(
            meta, cfg, kappa, G_shape,
            measurement_dim, measurement_noise,
            friction_uncertainty, u_hash
        ):
            print(f"[CACHE] Loaded {npz_path.name}")
            return base, meta
        else:
            print(f"[CACHE] Metadata mismatch for {npz_path.name}, rerunning.")

    # ---------- otherwise run ----------
    X, Xhat, u_out, b0_true, b1_true, S_lin, S_nonlin, info_gain, logdetP, dx, TE_global, hist_data = run_probe_rollout(
            p, cfg, u_used,
            kappa=float(kappa),
            n_sub=20,
            measurement_noise=measurement_noise,
            friction_uncertainty=friction_uncertainty,
            seed=cfg.seed,
            measurement_dim=1,
            G_shape=G_shape,
            make_gif=False
        )
    t_updates, c_hat_hist, Pcc_pred_hist, Pcc_upd_hist, nis_hist, Qcc_hist = hist_data
    
    # ---------- SAVE ----------
    np.savez_compressed(
        npz_path,
        t=np.asarray(cfg.dt * np.arange(len(X)), float),
        u=u_out,
        X_true=X,
        X_hat=Xhat,
        b0_true=b0_true,
        b1_true=b1_true,
        S_lin=S_lin,
        S_nonlin=S_nonlin,
        info_gain=info_gain,
        logdetP=logdetP,
        dx_series=dx,
        t_updates=t_updates,
        c_hat_hist=c_hat_hist,
        Pcc_pred_hist=Pcc_pred_hist,
        Pcc_upd_hist=Pcc_upd_hist,
        nis_hist=nis_hist,
        Qcc_hist=Qcc_hist,
    )

    meta = {
        "kappa": float(kappa),
        "G_shape": G_shape,
        "seed": int(seed),
        "dt": float(cfg.dt),
        "update_period": float(cfg.update_period),
        "T": float(cfg.T),
        "sigma_phi": float(cfg.sigma_phi),
        "te_lag": int(cfg.te_lag),
        "te_start_time": float(cfg.te_start_time),
        "measurement_dim": int(measurement_dim),
        "measurement_noise": bool(measurement_noise),
        "friction_uncertainty": bool(friction_uncertainty),
        "probe_u_hash": u_hash,
        "Qcc_mode": "adaptive_v1",
        "Qcc_qmin": cfg.q_min,
        "Qcc_qmax": cfg.q_max,
        "Qcc_gamma": cfg.q_gamma,
        "Qcc_tau": cfg.q_tau,
    }

    json_path.write_text(json.dumps(meta, indent=2))
    print(f"[SAVE] {npz_path.name}")

    base, meta2 = _load_data(npz_path, json_path)
    return base, meta2


def base_to_legacy_rollout_outputs(base: dict):
    """
    Convert cached base dict (loaded from npz) into the legacy return tuple:
      Xtrue, Xhat, u_out, S_lin, S_nonlin, info_gain, logdetP, dx, TE_global, hist_data

    hist_data is:
      (t_updates, c_hat_hist, Pcc_pred_hist, Pcc_upd_hist, nis_hist, Qcc_hist)
    """

    Xtrue     = base.get("X_true")
    Xhat      = base.get("X_hat")
    u_out     = base.get("u")
    b0_true   = base.get("b0_true")
    b1_true   = base.get("b1_true")
    S_lin     = base.get("S_lin")
    S_nonlin  = base.get("S_nonlin")
    info_gain = base.get("info_gain")
    logdetP   = base.get("logdetP")
    dx        = base.get("dx_series")

    # Optional legacy: TE_global (not always stored); keep None if missing
    TE_global = base.get("te_global", None)

    # History pack (new fields)
    t_updates      = base.get("t_updates", None)
    c_hat_hist     = base.get("c_hat_hist", None)
    Pcc_pred_hist  = base.get("Pcc_pred_hist", None)
    Pcc_upd_hist   = base.get("Pcc_upd_hist", None)
    nis_hist       = base.get("nis_hist", None)
    Qcc_hist       = base.get("Qcc_hist", None)

    hist_data = (t_updates, c_hat_hist, Pcc_pred_hist, Pcc_upd_hist, nis_hist, Qcc_hist)

    return Xtrue, Xhat, u_out, b0_true, b1_true, S_lin, S_nonlin, info_gain, logdetP, dx, TE_global, hist_data




# --------------------------
# Main driver
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_npz", type=str, default=None)
    parser.add_argument("--baseline_json", type=str, default=None)
    args = parser.parse_args()

    p = FurutaParams()
    cfg = ExperimentConfig()

    # Baseline paths
    default_npz, default_json = _default_baseline_paths(cfg)
    baseline_npz = Path(args.baseline_npz) if args.baseline_npz else default_npz
    baseline_json = Path(args.baseline_json) if args.baseline_json else default_json

    if not baseline_npz.exists():
        raise FileNotFoundError(f"Baseline npz not found: {baseline_npz}")

    base, meta = _load_data(baseline_npz, baseline_json)
    print("Loaded baseline:", baseline_npz)

    # Use baseline u(t) for fair comparison
    u_used = np.array(base["u"], dtype=float)
    t_used = np.array(base["t"], dtype=float)
    # probing input
    # t_used, u_used = generate_multisine_u(cfg.dt, cfg.T, amp=1.0, freqs=(0.2, 0.35, 0.6, 0.9, 1.2), seed=0, plot=False)

    if "mearurement_noise" in meta:
        measurement_noise = bool(meta["mearurement_noise"])
    else:
        measurement_noise = True
    friction_uncertainty = bool(meta.get("friction_uncertainty", True))

    # --- Baseline binned maps already stored ---
    theta_centers = np.array(base["theta_centers"], dtype=float)
    base_counts = np.array(base["counts"], dtype=int)
    base_S = np.array(base["S_lin_med"], dtype=float)
    base_I = np.array(base["I_med"], dtype=float)


    # Baseline TE arrays may be stored; ensure present
    base_te_centers = np.array(base["te_centers"], dtype=float) if "te_centers" in base else theta_centers
    base_te21 = np.array(base["te_2to1"], dtype=float) if "te_2to1" in base else np.full_like(theta_centers, np.nan)
    base_te12 = np.array(base["te_1to2"], dtype=float) if "te_1to2" in base else np.full_like(theta_centers, np.nan)
    base_te_counts = np.array(base["te_counts"], dtype=int) if "te_counts" in base else base_counts

    # We need quartiles for overlay shading; if not saved in baseline, approximate with NaNs
    # Scenario C's maps store only medians; if you saved q1/q3, load them here.
    base_S_q1 = np.array(base["S_lin_q1"], dtype=float) if "S_lin_q1" in base else np.full_like(base_S, np.nan)
    base_S_q3 = np.array(base["S_lin_q3"], dtype=float) if "S_lin_q3" in base else np.full_like(base_S, np.nan)
    base_I_q1 = np.array(base["I_q1"], dtype=float) if "I_q1" in base else np.full_like(base_I, np.nan)
    base_I_q3 = np.array(base["I_q3"], dtype=float) if "I_q3" in base else np.full_like(base_I, np.nan)

    # Kappas to test
    kappas = list(cfg.kappas)

    # Output dirs
    out_dir = Path(".")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Collect curve dicts for plotting
    S_curves = []
    I_curves = []
    TE_curves = []

    # Common binning config
    nbins = len(theta_centers)
    min_bin_count = 25
    min_te_count = 25

    
    # --------------------------
    # Run each kappa and save results
    # --------------------------
    G_shape = "sin_MSM" #_1plusCos"
    measurement_dim = 1
    make_gif = False
    for kappa in kappas:
        print(f"\n--- Scenario D: running kappa={kappa:.2f}, f(theta)={G_shape} ---")

        base, meta = load_or_run_case(
            p=p, cfg=cfg, data_dir=data_dir,
            kappa=kappa, G_shape=G_shape,
            u_used=u_used,
            seed=cfg.seed,
            measurement_dim=measurement_dim,
            measurement_noise=measurement_noise,
            friction_uncertainty=friction_uncertainty
        )

    
        X_true, Xhat, u_out, b0_true, b1_true, S_lin, S_nonlin, info_gain, logdetP, dx, TE_global, hist_data = base_to_legacy_rollout_outputs(base)
        t_updates, c_hat_hist, Pcc_pred_hist, Pcc_upd_hist, nis_hist, Qcc_hist = hist_data

        kappa_str = f"{kappa:.2f}".replace('.', '_')
        plot_state_estimates_time(t_used[:len(X_true)], X_true, Xhat, fname="D_state_estimation_time_{}_{}_init_pi.png".format(G_shape, kappa_str))

        plot_damping_function(c_hat_hist[-1], Pcc_upd_hist[-1], b0_true, b1_true, fname="D_damping_function_fit_{}_{}_init_pi.png".format(G_shape, kappa_str))

        if make_gif:
            make_damping_gif(
                t_updates, c_hat_hist, Pcc_upd_hist,
                b0_true=b0_true, b1_true=b1_true, gif_path="D_gif_damping_function_fit_{}_{}_init_pi.gif".format(G_shape, kappa_str)
            )

        plot_coeff_uncertainty_and_mean(
            t_updates, c_hat_hist, Pcc_upd_hist,
            b0_true=b0_true, b1_true=b1_true,
            fname="D_damping_coeff_learning_summary_{}_{}".format(G_shape, kappa_str)
        )

        # Theta-binned structural + info gain maps
        maps = compute_theta_binned_maps(
            time=t_used[:len(X_true)],
            X_true=X_true,
            U=u_out,
            S_lin=S_lin,
            S_nonlin=S_nonlin,
            info_gain=info_gain,
            nbins=nbins,
            theta_wrap=True,
            theta_min=-np.pi,
            theta_max=np.pi
        )
        
        # --- Directional TE (theta-binned) ---
        start_idx = int(cfg.te_start_time / cfg.dt)

        theta_series = wrap_angle(X_true[:, 1])
        theta_used_series = theta_series[start_idx:]
        dx_used = dx[start_idx:, :4]  # only physical dx

        
        centers_te, te21, te12, te_counts = theta_binned_te(
                dx_series=dx_used,
                theta=theta_used_series,
                nbins=nbins,
                theta_min=-np.pi,
                theta_max=np.pi,
                te_lag=cfg.te_lag,
                min_count=min_te_count,
                min_seg_len=max(10, cfg.te_lag + 5),
                te_func=lambda a, b, k: float(te_logdet(a, b, k)[0])
        )

        # Store for plotting
        S_curves.append({
            "label": f"κ={kappa:.2f}",
            "med": np.array(maps["S_lin_med"]),
            "q1": np.array(maps.get("S_lin_q1", np.full_like(maps["S_lin_med"], np.nan))),
            "q3": np.array(maps.get("S_lin_q3", np.full_like(maps["S_lin_med"], np.nan))),
            "counts": np.array(maps["counts"], dtype=int),
        })
        I_curves.append({
            "label": f"κ={kappa:.2f}",
            "med": np.array(maps["I_med"]),
            "q1": np.array(maps.get("I_q1", np.full_like(maps["I_med"], np.nan))),
            "q3": np.array(maps.get("I_q3", np.full_like(maps["I_med"], np.nan))),
            "counts": np.array(maps["counts"], dtype=int),
        })
        TE_curves.append({
            "label": f"κ={kappa:.2f}",
            "te_2to1": np.array(te21),
            "te_1to2": np.array(te12),
            "te_counts": np.array(te_counts, dtype=int),
        })

        
        plot_coeff_info_gain(t_updates, Pcc_pred_hist, Pcc_upd_hist, fname=f"D_coefficients_info_gain_{G_shape}_{kappa_str}.png")
        plot_nis_and_Qcc(t_updates, nis_hist, Qcc_hist, fname=f"D_nis_Qcc_{G_shape}_{kappa_str}.png")

        plot_dIc_vs_theta_updates(base, fname=f"D_dIc_vs_theta_{G_shape}_{kappa_str}.png")


    # --------------------------
    # Comparison plots (clean, not messy)
    # --------------------------

    # Structural coupling overlay (median)
    _plot_overlay(
        theta_centers,
        base_S, base_S_q1, base_S_q3,
        S_curves,
        ylabel="structural coupling (median)",
        title=f"Structural coupling vs θ (baseline κ=0 vs κ>0, {G_shape} shaping)",
        fname=f"D_compare_structural_vs_theta_{G_shape}.png",
        min_count=min_bin_count,
        base_counts=base_counts,
        curve_counts=[c["counts"] for c in S_curves]
    )

    # Info gain overlay (median)
    _plot_overlay(
        theta_centers,
        base_I, base_I_q1, base_I_q3,
        I_curves,
        ylabel="per-update info gain (median)",
        title=f"Per-update info gain vs θ (baseline κ=0 vs κ>0, {G_shape} shaping)",
        fname=f"D_compare_info_gain_vs_theta_{G_shape}.png",
        min_count=min_bin_count,
        base_counts=base_counts,
        curve_counts=[c["counts"] for c in I_curves]
    )


    # Directional TE overlays (two panels)
    _plot_te_overlay(
        base_te_centers,
        base_te21, base_te12, base_te_counts,
        TE_curves,
        fname=f"D_compare_TE_directional_{G_shape}.png",
        min_count=min_te_count
    )

    # Delta small multiples: structural coupling, info gain, TE 2->1
    _plot_delta_small_multiples(
        theta_centers, base_S, base_counts,
        S_curves,
        ylabel="Δ structural coupling",
        title="Δ structural coupling (κ - baseline)",
        fname=f"D_delta_structural_by_kappa_{G_shape}.png",
        min_count=min_bin_count
    )

    _plot_delta_small_multiples(
        theta_centers, base_I, base_counts,
        I_curves,
        ylabel="Δ per-update info gain",
        title="Δ per-update info gain (κ - baseline)",
        fname=f"D_delta_info_gain_by_kappa_{G_shape}.png",
        min_count=min_bin_count
    )

    te_like = [{"label": c["label"], "med": c["te_2to1"], "counts": c["te_counts"]} for c in TE_curves]

    _plot_delta_small_multiples(
        base_te_centers, base_te21, base_te_counts,
        te_like,
        ylabel="Δ TE 2→1",
        title="Δ TE 2→1 (κ - baseline)",
        fname=f"D_delta_TE2to1_by_kappa_{G_shape}.png",
        min_count=min_te_count
    )

    print("\nGenerated Scenario D comparison plots:")
    print("  D_compare_structural_vs_theta.png")
    print("  D_compare_info_gain_vs_theta.png")
    print("  D_compare_TE_directional.png")
    print("  D_delta_structural_by_kappa.png")
    print("  D_delta_info_gain_by_kappa.png")
    print("  D_delta_TE2to1_by_kappa.png")


if __name__ == "__main__":
    main()