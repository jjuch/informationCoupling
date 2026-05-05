
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from config import FurutaParams
from scenarioE_config import load_cases_json, select_case, hash_config, meta_matches
from scenarioF_controller import simulate_closed_loop
from scenarioE_metrics import compute_structural_series, compute_theta_binned_structural_maps
from scenarioF_plots import plot_states_with_refs, plot_Snonlin_vs_theta_compare
from scenarioF_debug_plots import plot_debug_bundle, plot_debug_actuation_and_energy_integrals

# ----------------------------
# Caching helpers (F-specific)
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
    kappa_str = f"{float(kappa):.3f}".replace(".", "_")
    tag = f"{cfg_hash[:8]}"
    return (
        data_dir / f"F_data_k{kappa_str}_{G_shape}_{tag}.npz",
        data_dir / f"F_data_k{kappa_str}_{G_shape}_{tag}.json",
    )


def run_or_load_case(cases_cfg, kappa, G_shape, data_dir: Path, show_plots=True, hash_overwrite=None):
    p = FurutaParams()
    cfg = select_case(cases_cfg, kappa, G_shape)
    cfg_hash = hash_config(cfg) if hash_overwrite is None else hash_overwrite

    npz_path, json_path = case_paths(data_dir, kappa, G_shape, cfg_hash)

    base = None
    meta = None

    if npz_path.exists():
        base0, meta0 = _load_data(npz_path, json_path)
        if meta_matches(meta0, kappa, G_shape, cfg_hash):
            print(f"[LOAD] {npz_path.name}")
            base, meta = base0, meta0

        if show_plots and bool(cfg.get("plots", {}).get("show", True)):
            title = f"Scenario F | kappa={float(kappa):+.3f}, G_shape={G_shape} | {meta.get('case_note','')}"
            plot_states_with_refs(base, cfg, meta_title=title)

    if base is None:
        base_sim, cfg_used = simulate_closed_loop(cfg)
        base = dict(base_sim)

        


        meta = {
            "scenario": "F_dynamicVHC_orbit",
            "kappa": float(kappa),
            "G_shape": G_shape,
            "cfg_hash": cfg_hash,
            "case_note": cfg_used.get("_case_note", ""),
        }

        

        if show_plots and bool(cfg.get("plots", {}).get("show", True)):
            # --- Debug plots ---
            title = f"Scenario F DEBUG | kappa={float(kappa):+.3f}, G_shape={G_shape}"
            plot_debug_bundle(base, cfg, meta_title=title, save_path=None, show=True)

            
            title_dbg2 = f"Scenario F ENERGY DEBUG | kappa={float(kappa):+.3f}, G_shape={G_shape}"
            plot_debug_actuation_and_energy_integrals(base, cfg, meta_title=title_dbg2, save_path=None, show=True)


            title = f"Scenario F | kappa={float(kappa):+.3f}, G_shape={G_shape} | {meta.get('case_note','')}"
            plot_states_with_refs(base, cfg, meta_title=title)
        
        print("Calculating structural series...")
        S_lin, S_non = compute_structural_series(base_sim, cfg_used, p)
        maps = compute_theta_binned_structural_maps(base_sim, cfg_used, S_lin, S_non)

        base["S_lin"] = S_lin
        base["S_nonlin"] = S_non
        base["maps_theta"] = maps["theta_centers"]
        base["S_nonlin_med"] = maps["S_nonlin_med"]
        base["S_nonlin_q1"] = maps["S_nonlin_q1"]
        base["S_nonlin_q3"] = maps["S_nonlin_q3"]
        base["S_lin_med"] = maps["S_lin_med"]
        base["S_lin_q1"] = maps["S_lin_q1"]
        base["S_lin_q3"] = maps["S_lin_q3"]

        save_run(npz_path, json_path, base, meta)

    return base, meta, cfg

def main(): 
    cases_cfg = load_cases_json(Path("scenarioF_cases.json"))
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
