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
from run_scenario_C import run_probe_rollout, compute_theta_binned_maps, plot_state_estimates_time, generate_multisine_u
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
        ("scenario", "D_compare_to_C_baseline"),
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


def _load_baseline(npz_path: Path, json_path: Path):
    base = np.load(npz_path, allow_pickle=True)
    meta = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    return base, meta


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

    base, meta = _load_baseline(baseline_npz, baseline_json)
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
    G_shape = "sin_1plusCos"
    for kappa in kappas:
        print(f"\n--- Scenario D: running kappa={kappa:.2f}, f(theta)=sin^2(theta) ---")

        # Run rollout reusing Scenario C function (do not re-implement)
        # IMPORTANT: measurement_dim=1 means phi-only measurement update (Option A)
        X, Xhat, u_out, S_lin, S_nonlin, info_gain, logdetP, dx, TE_global = run_probe_rollout(
            p, cfg, u_used,
            kappa=float(kappa),
            n_sub=20,
            measurement_noise=measurement_noise,
            friction_uncertainty=friction_uncertainty,
            seed=cfg.seed,
            measurement_dim=1,
            G_shape=G_shape
        )

        plot_state_estimates_time(t_used[:len(X)], X, Xhat, fname="D_state_estimation_time_{}_{}_init_pi.png".format(G_shape, str(kappa).replace('.', '_')))

        # Theta-binned structural + info gain maps
        maps = compute_theta_binned_maps(
            time=t_used[:len(X)],
            X_true=X,
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

        theta_series = wrap_angle(X[:, 1])
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

    
        # ---- Save result ----
        out_npz = data_dir / f"ScenarioD_sin2_kappa{kappa:.2f}_seed{cfg.seed}.npz"
        out_json = data_dir / f"ScenarioD_sin2_kappa{kappa:.2f}_seed{cfg.seed}.json"

        np.savez_compressed(
            out_npz,
            t=t_used[:len(X)],
            u=u_out,
            X_true=X,
            X_hat=Xhat,
            S_lin=S_lin,
            S_nonlin=S_nonlin,
            info_gain=info_gain,
            logdetP=logdetP,
            dx_series=dx,
            theta_centers=maps["theta_centers"],
            counts=maps["counts"],
            S_lin_med=maps["S_lin_med"],
            S_lin_q1=maps.get("S_lin_q1", np.full_like(maps["S_lin_med"], np.nan)),
            S_lin_q3=maps.get("S_lin_q3", np.full_like(maps["S_lin_med"], np.nan)),
            I_med=maps["I_med"],
            I_q1=maps.get("I_q1", np.full_like(maps["I_med"], np.nan)),
            I_q3=maps.get("I_q3", np.full_like(maps["I_med"], np.nan)),
            te_centers=centers_te,
            te_2to1=te21,
            te_1to2=te12,
            te_counts=te_counts
        )

        meta_k = dict(meta) if meta else {}
        meta_k.update({
            "scenario": "D_compare_to_C_baseline",
            "kappa": float(kappa),
            "G_shape": "sin2",
            "seed": cfg.seed,
            "dt": cfg.dt,
            "update_period": cfg.update_period,
            "T": cfg.T,
            "sigma_phi": cfg.sigma_phi,
            "te_lag": cfg.te_lag,
            "te_start_time": cfg.te_start_time,
            "measurement_noise": measurement_noise,
            "friction_uncertainty": friction_uncertainty,
            "baseline_npz": str(baseline_npz),
        })
        out_json.write_text(json.dumps(meta_k, indent=2))

        print("Saved:", out_npz)
        print("Saved:", out_json)

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


    # --------------------------
    # Comparison plots (clean, not messy)
    # --------------------------

    # Structural coupling overlay (median)
    _plot_overlay(
        theta_centers,
        base_S, base_S_q1, base_S_q3,
        S_curves,
        ylabel="structural coupling (median)",
        title="Structural coupling vs θ (baseline κ=0 vs κ>0, sin² shaping)",
        fname="D_compare_structural_vs_theta.png",
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
        title="Per-update info gain vs θ (baseline κ=0 vs κ>0, sin² shaping)",
        fname="D_compare_info_gain_vs_theta.png",
        min_count=min_bin_count,
        base_counts=base_counts,
        curve_counts=[c["counts"] for c in I_curves]
    )


    # Directional TE overlays (two panels)
    _plot_te_overlay(
        base_te_centers,
        base_te21, base_te12, base_te_counts,
        TE_curves,
        fname="D_compare_TE_directional.png",
        min_count=min_te_count
    )

    # Delta small multiples: structural coupling, info gain, TE 2->1
    _plot_delta_small_multiples(
        theta_centers, base_S, base_counts,
        S_curves,
        ylabel="Δ structural coupling",
        title="Δ structural coupling (κ - baseline)",
        fname="D_delta_structural_by_kappa.png",
        min_count=min_bin_count
    )

    _plot_delta_small_multiples(
        theta_centers, base_I, base_counts,
        I_curves,
        ylabel="Δ per-update info gain",
        title="Δ per-update info gain (κ - baseline)",
        fname="D_delta_info_gain_by_kappa.png",
        min_count=min_bin_count
    )

    te_like = [{"label": c["label"], "med": c["te_2to1"], "counts": c["te_counts"]} for c in TE_curves]

    _plot_delta_small_multiples(
        base_te_centers, base_te21, base_te_counts,
        te_like,
        ylabel="Δ TE 2→1",
        title="Δ TE 2→1 (κ - baseline)",
        fname="D_delta_TE2to1_by_kappa.png",
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