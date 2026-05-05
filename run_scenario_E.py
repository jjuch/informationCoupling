import numpy as np
from pathlib import Path
import json

from config import FurutaParams
from scenarioE_config import load_cases_json, select_case, hash_config, meta_matches
from scenarioE_controller import simulate_closed_loop
from scenarioE_metrics import compute_structural_series, compute_theta_binned_structural_maps
from scenarioE_plots import plot_states_with_refs, plot_Snonlin_vs_theta_compare


# ============================================================
# Scenario E: Feedforward + SMC + hybrid singular-zone control
# ============================================================

# ----------------------------
# Caching helpers (E-specific)
# ----------------------------


def _load_data(npz_path: Path, json_path: Path):
    base = np.load(npz_path, allow_pickle=True)
    base = {k: base[k] for k in base.files}
    meta = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    return base, meta

def save_run(npz_path: Path, json_path: Path, base: dict, meta: dict):
    np.savez_compressed(npz_path, **base)
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[SAVE] {npz_path.name}")

def case_paths(data_dir: Path, kappa: float, G_shape: str, cfg_hash: str):
    kappa_str = f"{kappa:.2f}".replace('.', '_')
    tag = f"{G_shape}_{kappa_str}_{cfg_hash[:8]}"
    return (data_dir / f"E_data_{tag}.npz", data_dir / f"E_data_{tag}.json")


# ============================================================
# Closed-loop RK4 step
# ============================================================

# def rk4_step_closed_loop(x, t, p, omega, cfg, thdd_ff_prev):
#     dt_int = cfg.dt / cfg.n_sub
#     xk = x.copy()

#     u_last = 0.0
#     b_last = 0.0
#     e_last = 0.0
#     s_last = 0.0
#     sing_last = False
#     thdd_ff_last = thdd_ff_prev

#     for _ in range(cfg.n_sub):
#         u1, b1, e1, s1, thff1, sing1 = control_law(
#             xk, t, p, omega, cfg.lambda_, thdd_ff_last
#         )

#         u_last, b_last, e_last, s_last, sing_last = u1, b1, e1, s1, sing1
#         thdd_ff_last = thff1

#         k1 = rhs_continuous(xk, u1, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

#         u2, _, _, _, _, _ = control_law(xk + 0.5*dt_int*k1, t + 0.5*dt_int, p, omega, lambda_, thdd_ff_last)
#         k2 = rhs_continuous(xk + 0.5*dt_int*k1, u2, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

#         u3, _, _, _, _, _ = control_law(xk + 0.5*dt_int*k2, t + 0.5*dt_int, p, omega, lambda_, thdd_ff_last)
#         k3 = rhs_continuous(xk + 0.5*dt_int*k2, u3, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

#         u4, _, _, _, _, _ = control_law(xk + dt_int*k3, t + dt_int, p, omega, lambda_, thdd_ff_last)
#         k4 = rhs_continuous(xk + dt_int*k3, u4, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

#         xk = xk + (dt_int/6.0)*(k1 + 2*k2 + 2*k3 + k4)
#         t += dt_int

#     return xk, u_last, b_last, e_last, s_last, thdd_ff_last, sing_last


# ============================================================
# Main simulation + plots
# ============================================================
def run_or_load_case(cases_cfg, kappa, G_shape, data_dir: Path, show_plots=True, hash_overwrite=None):
    p = FurutaParams()
    cfg = select_case(cases_cfg, kappa, G_shape)
    if hash_overwrite is None:
        cfg_hash = hash_config(cfg)
    else:
        cfg_hash = hash_overwrite
    npz_path, json_path = case_paths(data_dir, kappa, G_shape, cfg_hash)

    if npz_path.exists() and json_path.exists():
        base, meta = _load_data(npz_path, json_path)
        print('here')
        if meta_matches(meta, kappa, G_shape, cfg_hash) or hash_overwrite is not None:
            print(f"[CACHE] {npz_path.name}")
            if show_plots and cfg["plots"]["show"]:
                plot_states_with_refs(base, cfg, meta_title=f"Scenario E (cached) kappa={kappa}, {G_shape}")
            return base, meta, cfg

    # simulate
    base, cfg_used = simulate_closed_loop(cfg)
    
    if show_plots and cfg_used["plots"]["show"]:
        plot_states_with_refs(base, cfg_used, meta_title=f"Scenario E kappa={kappa}, {G_shape}")

    # compute structural metrics
    print(f"[{G_shape}, kappa: {kappa:.2f}] Computing structural coupling...")
    S_lin, S_non = compute_structural_series(base, cfg_used, p)
    base["S_lin"] = S_lin
    base["S_nonlin"] = S_non

    print(f"[{G_shape}, kappa: {kappa:.2f}]Mapping to binned theta...")
    maps = compute_theta_binned_structural_maps(base, cfg_used, S_lin, S_non)
    base["theta_centers"] = maps["theta_centers"]
    base["S_nonlin_med"] = maps["S_nonlin_med"]
    base["S_nonlin_q1"] = maps["S_nonlin_q1"]
    base["S_nonlin_q3"] = maps["S_nonlin_q3"]
    base["counts"] = maps["counts"]

    meta = {
        "scenario": "E_hybrid_pump_smc_mpc",
        "kappa": float(kappa),
        "G_shape": G_shape,
        "cfg_hash": cfg_hash,
        "cfg": cfg_used
    }

    print(f"[{G_shape}, kappa: {kappa:.2f}] Saving...")
    save_run(npz_path, json_path, base, meta)

    return base, meta, cfg_used


def main(): 
    cases_cfg = load_cases_json(Path("scenarioE_cases.json"))
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Baseline stabilized orbit
    print("==== run kappa 0.0 ====")
    base0, meta0, cfg0 = run_or_load_case(cases_cfg, 0.0, "const", data_dir, show_plots=True, hash_overwrite=None)

    # Test case: kappa=0.09, sin_MSM
    print("==== run kappa 0.09 ====")
    base1, meta1, cfg1 = run_or_load_case(cases_cfg, 0.09, "sin_MSM", data_dir, show_plots=True)

    # Compare S_nonlin medians vs theta
    maps0 = {"theta_centers": base0["theta_centers"], "S_nonlin_med": base0["S_nonlin_med"]}
    maps1 = {"theta_centers": base1["theta_centers"], "S_nonlin_med": base1["S_nonlin_med"]}

    plot_Snonlin_vs_theta_compare(
        maps0, maps1,
        label0="kappa=0.00 baseline (stabilized)",
        label1="kappa=0.09 sin_MSM (stabilized)"
    )


if __name__ == "__main__":
    main()
