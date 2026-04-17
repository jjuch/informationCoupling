#
# Scenario B: comparative study of information transfer / estimation effects under different controller-design fairness rules.
#
# Implements and compares:
#   Exp1  (Fixed-K):    K designed at kappa=0, reused for all kappa
#   Exp2A (LQR retune): K(kappa) = LQR(A(kappa),B(kappa),Q,R) with fixed weights
#   Exp2B (Pole match): K_pp(kappa) via pole placement to match closed-loop poles at kappa=0
#   Exp3  (Input replay): Use u(t) from Exp1 at kappa=0 and replay same u(t) for all kappa
#
# Outputs figures are saved
#
# Options:
#  - Validation V1: deterministic sanity rollouts
#  - Validation V2: discrete-time stability check via spectral radius of expm(Acl*dt)
#  - Monte Carlo: valid-run filtering using u_max (reject trajectories), print %valid and %valid-TE


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.linalg
from scipy.signal import place_poles

from config import FurutaParams, ExperimentConfig
from furuta_model import rk4_step, rhs_continuous, wrap_angle, wrap_state_angles
from control import linearize_rhs, lqr
from ekf import EKF, R_from_sigma_phi, measure_phi, derive_phidot
from info_metrics import auc_logdet, logdet_theta_block, te_value


# -----------------------------
# Utilities: wrapping, metrics
# -----------------------------
def theta_error(theta_true, theta_hat):
    """Wrapped theta error (signed)."""
    return wrap_angle(theta_true - theta_hat)

def abs_theta_error(theta_true, theta_hat):
    return np.abs(theta_error(theta_true, theta_hat))

def state_error(x, x_ref):
    """
    dx = x - x_ref with wrapped errors for phi and theta.
    Note that the error vector centers around 0 for all states.
    """
    x = np.asarray(x, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    dx = x - x_ref
    return wrap_state_angles(dx, phi_center=0, theta_center=0)

def plant_step_substeps(x, u, dt, n_sub, p, kappa, b_theta_true=None):
    """Integrate plant for dt using n_sub RK4 substeps."""
    x = np.asarray(x, dtype=float)
    dt_int = dt / int(n_sub)
    for _ in range(int(n_sub)):
        x = rk4_step(rhs_continuous, x, u, dt_int, p, kappa=float(kappa), G_shape='const', b_theta_true=b_theta_true)
        if not np.all(np.isfinite(x)):
            break
    return x


def spectral_radius(M):
    vals = np.linalg.eigvals(M)
    return float(np.max(np.abs(vals)))



# -----------------------------
# Controller design 
# -----------------------------

def design_K_fixed(p, x_ref, u_ref, Q_lqr, R_lqr):
    """Design baseline K at kappa=0."""
    A0, B0 = linearize_rhs(p, x_ref, u_ref, kappa=0.0, G_shape="const", eps=1e-6)
    K0 = lqr(A0, B0, Q_lqr, R_lqr)
    poles0 = np.linalg.eigvals(A0 - B0 @ K0)
    return K0, poles0, A0, B0


def design_K_lqr_retune(p, x_ref, u_ref, kappa, Q_lqr, R_lqr):
    """LQR re-tuning at given kappa."""
    A, B = linearize_rhs(p, x_ref, u_ref, kappa=float(kappa), G_shape="const", eps=1e-6)
    K = lqr(A, B, Q_lqr, R_lqr)
    poles = np.linalg.eigvals(A - B @ K)
    return K, poles, A, B


def design_K_polematch(p, x_ref, u_ref, kappa, target_poles):
    """
    Pole placement at given kappa to match target_poles (complex allowed if conjugate pairs present).
    Uses scipy.signal.place_poles.
    """
    A, B = linearize_rhs(p, x_ref, u_ref, kappa=float(kappa), G_shape="const", eps=1e-6)

    # Ensure target poles are a list (place_poles expects array-like)
    target_poles = np.asarray(target_poles)

    # place_poles returns object with .gain_matrix
    placed = place_poles(A, B, target_poles, method="YT")
    K = placed.gain_matrix
    poles = np.linalg.eigvals(A - B @ K)
    return K, poles, A, B


# -----------------------------
# Single rollout (one trial) under a chosen experiment
# -----------------------------

def rollout_closed_loop(p, cfg, kappa, K, x_ref,
                        mode="ekf",
                        u_sequence=None,
                        seed=0,
                        friction_uncertainty=False,
                        measurement_noise=True,
                        n_sub=20
):
    """
    Run one rollout.

    Validity rule:
      - reject run if abs(u) > p.u_max at any time step (trajectory rejection, not saturation)
      - reject run if non-finite plant state or non-finite EKF state

    Modes:
      - mode="ekf": controller uses EKF estimate
      - mode="true": controller uses true state
      - mode="replay": ignores controller and uses provided u_sequence

    Returns dict with:
      theta_true, theta_hat, abs_err, logdetP, u, dx, valid, reason, max_u
    """

    dt = cfg.dt
    N = int(cfg.T / dt)

    rng = np.random.default_rng(seed)

    # friction mismatch (optional)
    if friction_uncertainty:
        delta = rng.uniform(-cfg.friction_uncertainty, cfg.friction_uncertainty)
    else:
        delta = 0.0
    b_theta_true = p.b_theta_nom * (1.0 + delta)

    # initial condition: use cfg.x0_true but ensure it matches your chosen study regime
    x_true = np.asarray(cfg.x0_true, dtype=float).copy()

    # EKF init
    R = R_from_sigma_phi(cfg.sigma_phi, cfg.dt)
    # Reasonable EKF initial covariances for debugging
    P0 = cfg.P0.copy()
    Qk = cfg.Q.copy()
    
    ekf = EKF(cfg.x0_hat, P0, Qk, R, dt, p, kappa=float(kappa), G_shape="const")

    # needed for correlated phidot
    if measurement_noise:
        phi_meas_prev = measure_phi(x_true[0], cfg.sigma_phi, rng)
    else:
        phi_meas_prev = x_true[0]

    # storage
    theta_true = np.zeros(N)
    theta_hat = np.zeros(N)
    abs_err = np.zeros(N)
    logdetP = np.zeros(N)
    u_hist = np.zeros(N)
    dx_series = np.zeros((N, 4))

    valid = True
    reason = "ok"
    max_u = 0.0

    for k in range(N):
        theta_true[k] = x_true[1]
        theta_hat[k] = ekf.x[1]
        abs_err[k] = abs_theta_error(x_true[1], ekf.x[1])
        logdetP[k] = logdet_theta_block(ekf.P)

        # control
        if mode == "replay":
            u = float(u_sequence[k])
        else:
            if mode == "true":
                dx = state_error(x_true, x_ref)
            else:
                dx = state_error(ekf.x, x_ref)
            u_min = K @ dx
            u = -float(u_min[0])

        u_hist[k] = u
        max_u = max(max_u, abs(u))
        
        # trajectory rejection (no saturation)
        if abs(u) > p.u_max:
            valid = False
            reason = "u_exceeds_u_max"
            # truncate
            theta_true = theta_true[:k+1]
            theta_hat = theta_hat[:k+1]
            abs_err = abs_err[:k+1]
            logdetP = logdetP[:k+1]
            u_hist = u_hist[:k+1]
            dx_series = dx_series[:k+1]
            break


        # plant step (substepped)
        x_true = plant_step_substeps(x_true, u, dt, n_sub, p, float(kappa), b_theta_true=b_theta_true)
        if not np.all(np.isfinite(x_true)):
            valid = False
            reason = "nonfinite_plant"
            # truncate
            theta_true = theta_true[:k+1]
            theta_hat = theta_hat[:k+1]
            abs_err = abs_err[:k+1]
            logdetP = logdetP[:k+1]
            u_hist = u_hist[:k+1]
            dx_series = dx_series[:k+1]
            break

        # measurement
        if measurement_noise:
            phi_meas = measure_phi(x_true[0], cfg.sigma_phi, rng)
            phidot_meas = derive_phidot(phi_meas, phi_meas_prev, dt)
            phi_meas_prev = phi_meas
        else:
            phi_meas = x_true[0]
            phidot_meas = x_true[2]
            phi_meas_prev = phi_meas

        z = np.array([phi_meas, phidot_meas], dtype=float)

        # EKF
        ekf.predict(u)
        _, dx_upd = ekf.update(z)
        if k < dx_series.shape[0]:
            dx_series[k] = dx_upd

        if not np.all(np.isfinite(ekf.x)):
            valid = False
            reason = "nonfinite_ekf"
            theta_true = theta_true[:k+1]
            theta_hat = theta_hat[:k+1]
            abs_err = abs_err[:k+1]
            logdetP = logdetP[:k+1]
            u_hist = u_hist[:k+1]
            dx_series = dx_series[:k+1]
            break

    return {
        "theta_true": theta_true,
        "theta_hat": theta_hat,
        "abs_err": abs_err,
        "logdetP": logdetP,
        "u": u_hist,
        "dx": dx_series,
        "valid": valid,
        "reason": reason,
        "max_u": max_u
    }


# -----------------------------
# Validation V1: deterministic sanity rollouts
# -----------------------------

def validation_v1():
    """
    Validation V1: deterministic checks before MC.
    Runs one rollout per experiment and kappa with:
      - no friction uncertainty
      - no measurement noise
    Prints whether run is valid and max|u|.
    """
    p = FurutaParams()
    cfg = ExperimentConfig()
    x_ref = cfg.x_ref

    Q_lqr = cfg.Q_lqr
    R_lqr = cfg.R_lqr

    K_fixed, poles_target, _, _ = design_K_fixed(p, x_ref, 0.0, Q_lqr, R_lqr)

    experiments = ["Exp1_fixedK", "Exp2A_lqrRetune", "Exp2B_poleMatch"]
    for exp in experiments:
        print(f"\n[V1] {exp}")
        for kappa in cfg.kappas:
            if exp == "Exp1_fixedK":
                Kk = K_fixed
            elif exp == "Exp2A_lqrRetune":
                Kk, _, _, _ = design_K_lqr_retune(p, x_ref, 0.0, kappa, Q_lqr, R_lqr)
            else:
                Kk, _, _, _ = design_K_polematch(p, x_ref, 0.0, kappa, poles_target)

            roll = rollout_closed_loop(
                p, cfg, kappa, Kk, x_ref,
                mode="ekf",
                seed=cfg.seed,
                friction_uncertainty=False,
                measurement_noise=False,
                n_sub=20
            )
            print(f"  kappa={kappa:+.2f} | valid={roll['valid']} | reason={roll['reason']:<16} | max|u|={roll['max_u']:.3f} | steps={len(roll['u'])}")

    print("\n[V1] Exp3_inputReplay depends on baseline u(t); validate in main after baseline is found.")


# -----------------------------
# Validation V2: discrete-time stability proxy from poles
# -----------------------------

def validation_v2():
    """
    Validation V2: for each experiment and kappa, compute spectral radius of expm(Acl*dt),
    where Acl = A - B K. If rho >= 1, sampled-data stability is not guaranteed (linearized model).
    """
    p = FurutaParams()
    cfg = ExperimentConfig()
    x_ref = cfg.x_ref
    dt = cfg.dt

    Q_lqr = cfg.Q_lqr
    R_lqr = cfg.R_lqr

    K_fixed, poles_target, _, _ = design_K_fixed(p, x_ref, 0.0, Q_lqr, R_lqr)

    experiments = ["Exp1_fixedK", "Exp2A_lqrRetune", "Exp2B_poleMatch"]
    for exp in experiments:
        print(f"\n[V2] {exp}")
        for kappa in cfg.kappas:
            A, B = linearize_rhs(p, x_ref, 0.0, kappa=float(kappa), G_shape="const", eps=1e-6)

            if exp == "Exp1_fixedK":
                Kk = K_fixed
            elif exp == "Exp2A_lqrRetune":
                Kk, _, _, _ = design_K_lqr_retune(p, x_ref, 0.0, kappa, Q_lqr, R_lqr)
            else:
                Kk, _, _, _ = design_K_polematch(p, x_ref, 0.0, kappa, poles_target)

            Acl = A - B @ Kk
            Ad = scipy.linalg.expm(Acl * dt)
            rho = spectral_radius(Ad)
            print(f"  kappa={kappa:+.2f} | rho(exp(Acl*dt))={rho:.6f}")

    print("\n[V2] Exp3_inputReplay has no feedback K; stability depends on the chosen replay input.")


# -----------------------------
# Monte Carlo harness per experiment type
# -----------------------------

def mc_experiment(p, cfg, kappas, experiment, K_fixed, poles_target, Q_lqr, R_lqr, x_ref, n_mc=50):
    """
    experiment in {"Exp1_fixedK","Exp2A_lqrRetune","Exp2B_poleMatch","Exp3_inputReplay"}.
    Returns dict keyed by kappa with arrays of metrics and one example trajectory.
    """
    dt = cfg.dt
    N = int(cfg.T / dt)
    start_idx = int(cfg.te_start_time / dt)
    te_lag = cfg.te_lag

    out = {}

    # For input replay: generate u(t) once from baseline kappa=0 under fixedK EKF
    u_replay = None
    if experiment == "Exp3_inputReplay":
        baseline = None
        for trial_seed in range(cfg.seed, cfg.seed + 200):
            baseline = rollout_closed_loop(p, cfg, kappa=0.0, K=K_fixed, x_ref=x_ref, mode="ekf", seed=trial_seed, friction_uncertainty=True, measurement_noise=True, n_sub=20)
            if baseline["valid"] and len(baseline["u"]) == N:
                break
            baseline = None
        if baseline is None:
            raise RuntimeError("Exp3_inputReplay: Could not find a valid baseline run for replay within 200 seeds.")
        u_replay = baseline["u"].copy()

    for kappa in kappas:
        kappa = float(kappa)

        auc_logd = np.full(n_mc, np.nan)
        auc_err = np.full(n_mc, np.nan)
        te_vals = np.full(n_mc, np.nan)
        max_u = np.full(n_mc, np.nan)
        valid = np.zeros(n_mc, dtype=bool)
        reasons = [""] * n_mc

        # pick a controller for this kappa under this experiment
        if experiment == "Exp1_fixedK":
            Kk = K_fixed
        elif experiment == "Exp2A_lqrRetune":
            Kk, _, _, _ = design_K_lqr_retune(p, x_ref, 0.0, kappa, Q_lqr, R_lqr)
        elif experiment == "Exp2B_poleMatch":
            Kk, _, _, _ = design_K_polematch(p, x_ref, 0.0, kappa, poles_target)
        elif experiment == "Exp3_inputReplay":
            Kk = None
        else:
            raise ValueError("Unknown experiment type.")

        rolls = []
        logdet_list = []

        for i in range(n_mc):
            print(f"{kappa}: {i + 1} / {n_mc}", end="\r")
            seed = 1000 + i

            if experiment == "Exp3_inputReplay":
                roll = rollout_closed_loop(
                    p, cfg, kappa, K_fixed, x_ref,
                    mode="replay", u_sequence=u_replay, seed=seed, friction_uncertainty=True, measurement_noise=True,
                    n_sub=20
                    )
            else:
                roll = rollout_closed_loop(
                    p, cfg, kappa, Kk, x_ref, 
                    mode="ekf", 
                    seed=seed,
                    friction_uncertainty=True,
                    measurement_noise=True,
                    n_sub=20
                )

            rolls.append(roll)
            reasons[i] = roll["reason"]
            max_u[i] = roll["max_u"]
            valid[i] = roll["valid"] and (len(roll["u"]) == N)


            # metrics
            if valid[i]:
                auc_logd[i] = auc_logdet(roll["logdetP"], dt)
                auc_err[i] = auc_logdet(roll["abs_err"], dt)

                # TE on state corrections partition (phi/phidot vs theta/thetadot)
                dx = roll["dx"]
                if dx.shape[0] > (start_idx + te_lag + 5):
                    nu1 = dx[start_idx:, [0,2]]
                    nu2 = dx[start_idx:, [1,3]]
                    te_vals[i] = te_value(nu1, nu2, te_lag)
                logdet_list.append(roll["logdetP"])

        
        # representative example: median auc_logdet among valid runs
        example = None
        idx_valid = np.where(np.isfinite(auc_logd))[0]
        if len(idx_valid) > 0:
            idx_sorted = idx_valid[np.argsort(auc_logd[idx_valid])]
            idx_pick = idx_sorted[len(idx_sorted)//2]
            example = rolls[int(idx_pick)]

        logdet_runs = np.vstack(logdet_list) if len(logdet_list) else None


        out[kappa] = {
            "auc_logdet": auc_logd,
            "auc_err": auc_err,
            "te": te_vals,
            "max_u": max_u,
            "valid": valid,
            "reasons": reasons,
            "example": example,
            "logdet_runs": logdet_runs
        }

    return out


# -----------------------------
# Plotting
# -----------------------------


def plot_poles_subplots(p, cfg, K_fixed, poles_target, Q_lqr, R_lqr, fname="B_poles_comparison.png"):
    x_ref = cfg.x_ref
    kappas = list(cfg.kappas)
    exps = ["Exp1_fixedK", "Exp2A_lqrRetune", "Exp2B_poleMatch"]

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i/(len(kappas)-1 if len(kappas)>1 else 1)) for i in range(len(kappas))]

    for ax, exp in zip(axs, exps):
        for idx, kappa in enumerate(kappas):
            A, B = linearize_rhs(p, x_ref, 0.0, kappa=float(kappa), G_shape="const", eps=1e-6)
            if exp == "Exp1_fixedK":
                Kk = K_fixed
            elif exp == "Exp2A_lqrRetune":
                Kk, _, _, _ = design_K_lqr_retune(p, x_ref, 0.0, kappa, Q_lqr, R_lqr)
            else:
                Kk, _, _, _ = design_K_polematch(p, x_ref, 0.0, kappa, poles_target)

            poles = np.linalg.eigvals(A - B @ Kk)
            ax.scatter(np.real(poles), np.imag(poles), color=colors[idx], s=35)

        ax.axvline(0.0, color="k", linewidth=1.0)
        ax.set_title(exp)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Re")

    axs[0].set_ylabel("Im")
    fig.suptitle("Closed-loop poles by experiment (color = kappa)", y=0.98)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(kappas), vmax=max(kappas)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("kappa")
    
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()




def plot_logdet_medians(cfg, results_by_exp, fname="B_logdet_median_by_experiment.png"):
    dt = cfg.dt
    N = int(cfg.T/dt)
    t = np.arange(N)*dt

    exps = list(results_by_exp.keys())
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, exp in zip(axs, exps):
        for kappa in [cfg.kappas[0], cfg.kappas[-1]]:
            runs = results_by_exp[exp][float(kappa)]["logdet_runs"]
            if runs is None:
                continue
            med = np.nanmedian(runs, axis=0)
            ax.plot(t, med, label=f"kappa={kappa:+.2f}")
        ax.set_title(exp)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axs:
        ax.set_ylim([-30, 20])
        ax.set_xlabel("time [s]")
        ax.set_ylabel(r"median $\log\det P_\theta(t)$")

    fig.suptitle("Median logdet covariance (valid runs only)", y=0.98)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def boxplot_metric(results_by_exp, kappas, key, ylabel, title, fname):
    plt.figure(figsize=(10, 5.5))
    # Arrange as: each experiment grouped, each kappa as separate box within group
    exps = list(results_by_exp.keys())

    data = []
    labels = []
    
    for exp in exps:
        for kappa in kappas:
            arr = np.asarray(results_by_exp[exp][float(kappa)].get(key, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                arr = np.array([np.nan], dtype=float)  # placeholder to keep layout stable
            data.append(arr)
            labels.append(f"{exp}\n{kappa:+.2f}")


    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_time_domain_examples(cfg, results_by_exp, kappas, fname="B_time_domain_examples.png"):
    dt = cfg.dt
    N = int(cfg.T / dt)
    t = np.arange(N) * dt
    k0 = float(kappas[0])
    k1 = float(kappas[-1])

    exps = list(results_by_exp.keys())
    fig, axs = plt.subplots(len(exps), 2, figsize=(12, 3.5*len(exps)), sharex=True)
    if len(exps) == 1:
        axs = np.array([axs])

    for i, exp in enumerate(exps):
        # compare kappa=0 and kappa=max
        for kappa, color in zip([kappas[0], kappas[-1]], ["C0", "C1"]):
            ex = results_by_exp[exp][float(kappa)]["example"]
            if ex is None:  
                # annotate missing example
                axs[i, 0].text(0.02, 0.85, f"no valid example for kappa={kappa:+.2f}",
                               transform=axs[i, 0].transAxes, fontsize=8)
                axs[i, 1].text(0.02, 0.85, f"no valid example for kappa={kappa:+.2f}",
                               transform=axs[i, 1].transAxes, fontsize=8)
                continue

            tt = t[:len(ex["theta_true"])]
            axs[i,0].plot(tt, ex["theta_true"], color=color, linewidth=1.4, label=f"true, k={kappa:+.2f}")
            axs[i,0].plot(tt, ex["theta_hat"], color=color, linestyle="--", linewidth=1.2, label=f"hat, k={kappa:+.2f}")
            axs[i,1].plot(tt, ex["u"], color=color, linewidth=1.2, label=f"u, k={kappa:+.2f}")

        axs[i,0].set_ylabel("theta [rad]")
        axs[i,0].set_title(f"{exp}: theta true vs hat (median-AUC example)")
        axs[i,0].grid(True, alpha=0.3)
        axs[i,0].legend(fontsize=8, ncol=2)

        axs[i,1].set_ylabel("u [N·m]")
        axs[i,1].set_title(f"{exp}: control input u(t)")
        axs[i,1].grid(True, alpha=0.3)
        axs[i,1].legend(fontsize=8, ncol=2)

    for j in range(2):
        axs[-1,j].set_xlabel("time [s]")

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    p = FurutaParams()
    cfg = ExperimentConfig(dt=0.01, sigma_phi=1e-3)

    x_ref = cfg.x_ref # Setpoint: DOWN equilibrium in your convention
    kappas = list(cfg.kappas) # kappas to test
    n_mc = cfg.mc_trials_debug # number of monte carlo samples

    # LQR weights
    Q_lqr = cfg.Q_lqr
    R_lqr = cfg.R_lqr

    # Baseline K at kappa=0
    K_fixed, poles_target, _, _ = design_K_fixed(p, x_ref, 0.0, Q_lqr, R_lqr)

    experiments = ["Exp1_fixedK", "Exp2A_lqrRetune", "Exp2B_poleMatch"]#, "Exp3_inputReplay"]
    results_by_exp = {}

    for exp in experiments:
        print(f"\n=== Running {exp} (n_mc={n_mc}) ===")
        res = mc_experiment(p, cfg, kappas, exp, K_fixed, poles_target, Q_lqr, R_lqr, x_ref, n_mc=n_mc)
        results_by_exp[exp] = res

        # Terminal summary per kappa: valid %, TE valid %, medians
        for kappa in kappas:
            rr = res[float(kappa)]
            valid_mask = np.asarray(rr.get("valid", np.zeros(n_mc, dtype=bool)), dtype=bool)

            valid_pct = 100.0 * float(np.mean(valid_mask)) if valid_mask.size else 0.0

            te_arr = np.asarray(rr.get("te", np.full(n_mc, np.nan)), dtype=float)
            te_valid_mask = valid_mask & np.isfinite(te_arr)
            te_valid_pct = 100.0 * float(np.mean(te_valid_mask)) if te_valid_mask.size else 0.0

            auc_ld = np.asarray(rr.get("auc_logdet", np.full(n_mc, np.nan)), dtype=float)
            auc_er = np.asarray(rr.get("auc_err", np.full(n_mc, np.nan)), dtype=float)
            max_u = np.asarray(rr.get("max_u", np.full(n_mc, np.nan)), dtype=float)

            med_auc_ld = float(np.nanmedian(auc_ld)) if np.any(np.isfinite(auc_ld)) else np.nan
            med_auc_er = float(np.nanmedian(auc_er)) if np.any(np.isfinite(auc_er)) else np.nan
            med_te = float(np.nanmedian(te_arr[np.isfinite(te_arr)])) if np.any(np.isfinite(te_arr)) else np.nan
            med_maxu = float(np.nanmedian(max_u[valid_mask])) if np.any(valid_mask) else np.nan

            
            print(
                f"  kappa={float(kappa):+0.2f} | "
                f"valid={valid_pct:5.1f}% | "
                f"med AUC(logdet)={med_auc_ld:9.3f} | "
                f"med AUC(err)={med_auc_er:7.3f} | "
                f"med TE={med_te:7.4f} (TE valid {te_valid_pct:5.1f}%) | "
                f"med max|u|={med_maxu:7.3f}"
            )


    # Plot poles comparison (Exp1 shows only kappa=0; Exp2A/Exp2B show kappa sweep)
    plot_poles_subplots(p, cfg, K_fixed, poles_target, Q_lqr, R_lqr, fname="B_poles_comparison.png")

    # Plot representative logdet curves
    plot_logdet_medians(cfg, results_by_exp, fname="B_logdet_median_by_experiment.png")

    # Boxplots
    boxplot_metric(results_by_exp, kappas, "auc_logdet",
                   ylabel=r"AUC $\int_0^T \log\det P_\theta(t)\,dt$",
                   title="AUC of logdet covariance (lower is better)",
                   fname="B_auc_logdet_box.png")

    boxplot_metric(results_by_exp, kappas, "auc_err",
                   ylabel=r"AUC $\int_0^T |\mathrm{wrap}(\theta-\hat\theta)|\,dt$",
                   title="AUC of wrapped absolute theta estimation error (lower is better)",
                   fname="B_auc_thetaerr_box.png")

    boxplot_metric(results_by_exp, kappas, "te",
                   ylabel=r"$\mathrm{TE}_{2\to 1}$ (log-det VAR score)",
                   title="TE on EKF state-corrections (higher indicates more directed predictability)",
                   fname="B_te_box.png")

    # Time-domain examples (theta true vs hat, and u(t)) for kappa=0 and kappa=max
    plot_time_domain_examples(cfg, results_by_exp, kappas, fname="B_time_domain_examples.png")

    print("\nScenario B completed.")
    print("Saved figures:")
    print("  B_poles_comparison.png")
    print("  B_logdet_median_by_experiment.png")
    print("  B_auc_logdet_box.png")
    print("  B_auc_thetaerr_box.png")
    print("  B_te_box.png")
    print("  B_time_domain_examples.png")


if __name__ == "__main__":
    # validation_v1()
    # validation_v2()
    main()
