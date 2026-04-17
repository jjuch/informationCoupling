# run_scenario_A.py
# Scenario A: theta regulation around down position with collocated sensing
# Monte Carlo evaluation of theta-covariance collapse time and TE on EKF state corrections

import numpy as np
import matplotlib.pyplot as plt

from config import FurutaParams, ExperimentConfig
from furuta_model import rk4_step, rhs_continuous, wrap_state_angles, simulate_free_response
from ekf import EKF, R_from_sigma_phi, measure_phi, derive_phidot
from control import lqr, saturate, linearize_rhs, state_error
from info_metrics import te_logdet, logdet_theta_block, auc_logdet


def closed_loop_step(x_true, x_ref, ekf, K, p, dt, kappa, rng=None, sigma_phi=None, b_theta_true=None):
    """
    One closed-loop step. If rng is None, runs deterministic (no measurement noise).
    Returns updated (x_true, innov, dx, u, z).
    """
    # control on estimate
    dxhat = state_error(ekf.x, x_ref)
    u_min = K @ dxhat
    u = -float(u_min[0])
    u = saturate(u, p.u_max)

    # propagate plant
    x_true = rk4_step(rhs_continuous, x_true, u, dt, p, kappa=kappa, G_shape="const", b_theta_true=b_theta_true)

    # measurement
    if rng is None:
        phi_meas = x_true[0]
        phidot_meas = x_true[2]
    else:
        # correlated measurement via differentiation handled outside
        raise RuntimeError("Use the trial loop to handle correlated phidot measurement.")

    z = np.array([phi_meas, phidot_meas], dtype=float)

    # EKF
    ekf.predict(u)
    innov, dx = ekf.update(z)

    return x_true, innov, dx, u, z


def sanity_check_closed_loop(p, cfg, K, kappas_to_test):
    """
    Deterministic sanity check: delta=0, no noise, check bounded trajectories and saturation.
    Produces a plot of theta(t) and u(t) for each kappa.
    """
    dt = cfg.dt
    N = int(cfg.T / dt)
    t = np.arange(N) * dt

    x0 = cfg.x0_true
    x_ref = cfg.x_ref
    P0 = cfg.P0.copy()
    Q = cfg.Q.copy()
    R = R_from_sigma_phi(cfg.sigma_phi, cfg.dt)

    plt.figure(figsize=(9, 4.5))
    for kappa in kappas_to_test:
        x_true = x0.copy()
        ekf = EKF(cfg.x0_hat, P0, Q, R, dt, p, kappa=float(kappa), G_shape="const")
        theta_hist = np.zeros(N)
        u_hist = np.zeros(N)

        for k in range(N):
            x_true, _, _, u, _ = closed_loop_step(x_true, x_ref, ekf, K, p, dt, float(kappa), rng=None)
            theta_hist[k] = x_true[1]
            u_hist[k] = u

        plt.plot(t, theta_hist, label=f"kappa={kappa:+.2f}")

    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.title("Closed-loop sanity: deterministic theta(t) for different kappa")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sanity_theta.png", dpi=200)

    plt.figure(figsize=(9, 4.5))
    for kappa in kappas_to_test:
        x_true = x0.copy()
        ekf = EKF(cfg.x0_hat, P0, Q, R, dt, p, kappa=float(kappa), G_shape="const")
        u_hist = np.zeros(N)
        for k in range(N):
            x_true, _, _, u, _ = closed_loop_step(x_true, x_ref, ekf, K, p, dt, float(kappa), rng=None)
            u_hist[k] = u
        plt.plot(t, u_hist, label=f"kappa={kappa:+.2f}")
    plt.xlabel("time [s]")
    plt.ylabel("u [N·m]")
    plt.title("Closed-loop sanity: deterministic control effort u(t)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sanity_u.png", dpi=200)

    print("Saved sanity plots: sanity_theta.png, sanity_u.png")



def run_one_trial(p: FurutaParams, cfg: ExperimentConfig, K: np.ndarray, kappa: float, rng: np.random.Generator, record:bool=False):
    """
    Run a single trial of Scenario A.

    Returns:
        dict with keys:
          - logdet: (N,) logdet time series for P_theta block
          - AUC: Area under curve of logdet
          - TE: scalar TE_{2->1} on EKF state corrections
          - delta: sampled friction mismatch
    """
    dt = cfg.dt
    N = int(cfg.T / dt)

    # sample passive-joint friction mismatch delta in [-unc, +unc]
    delta = rng.uniform(-cfg.friction_uncertainty, cfg.friction_uncertainty)
    b_theta_true = p.b_theta_nom * (1.0 + delta)

    # true state init
    x_true = np.array(cfg.x0_true, dtype=float).copy()
    x_true = wrap_state_angles(x_true, theta_center=np.pi)

    # EKF init: uses nominal friction, same kappa as plant
    R = R_from_sigma_phi(cfg.sigma_phi, cfg.dt)
    ekf = EKF(cfg.x0_hat, cfg.P0, cfg.Q, R, cfg.dt, p, kappa=kappa, G_shape="const")

    # storage
    logdet_series = np.zeros(N, dtype=float)
    dx_series = np.zeros((N, 4), dtype=float)  # EKF state corrections (x_k|k - x_k|k-1)

    # optional recording arrays
    if record:
        x_true_hist = np.zeros((N, 4), dtype=float)
        x_hat_hist = np.zeros((N, 4), dtype=float)
        sig_theta = np.zeros(N, dtype=float)
        u_hist = np.zeros(N, dtype=float)

    # measurement init
    phi_meas_prev = measure_phi(x_true[0], cfg.sigma_phi, rng)

    # simulate
    for k in range(N):
        # control based on estimate
        min_u = K @ ekf.x
        u = -float(min_u[0])
        u = saturate(u, p.u_max)

        # propagate true plant
        x_true = rk4_step(
            rhs_continuous, x_true, u, dt, p,
            kappa=kappa, G_shape="const", b_theta_true=b_theta_true
        )

        # measurements (phi + derived phidot)
        phi_meas = measure_phi(x_true[0], cfg.sigma_phi, rng)
        phidot_meas = derive_phidot(phi_meas, phi_meas_prev, dt)
        phi_meas_prev = phi_meas
        z = np.array([phi_meas, phidot_meas], dtype=float)

        # EKF
        ekf.predict(u)
        _innov, dx = ekf.update(z)
        print("dx = ", dx)

        # logdet of theta block covariance
        logdet_series[k] = logdet_theta_block(ekf.P, idx_theta=(1, 3))
        dx_series[k, :] = dx
        
        if record:
            x_true_hist[k] = x_true
            x_hat_hist[k] = ekf.x
            sig_theta[k] = np.sqrt(max(ekf.P[1, 1], 0.0))
            u_hist[k] = u


    # TE on EKF state corrections (exclude initial transient)
    start_idx = int(cfg.te_start_time / dt)
    nu1 = dx_series[start_idx:, [0, 2]]  # corrections for (phi, phidot)
    nu2 = dx_series[start_idx:, [1, 3]]  # corrections for (theta, thetadot)

    te, _, _ = te_logdet(nu1, nu2, cfg.te_lag)

    # AUC
    auc = auc_logdet(logdet_series, dt)

    out = {
        "logdet": logdet_series,
        "AUC": auc,
        "TE": te,
        "delta": delta,
    }

    if record: 
        out.update({
            "x_true": x_true_hist,
            "x_hat": x_hat_hist,
            "sig_theta": sig_theta,
            "u": u_hist
        })
    
    return out



def run_monte_carlo(p: FurutaParams, cfg: ExperimentConfig, K: np.ndarray, kappas, n_mc: int):
    """
    Run Monte Carlo over all kappas.

    Returns:
        results: dict
          results[kappa]["logdet"] shape (n_mc, N)
          results[kappa]["T_eps"]  shape (n_mc,)
          results[kappa]["TE"]     shape (n_mc,)
          results[kappa]["delta"]  shape (n_mc,)
    """
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt
    N = int(cfg.T / dt)

    results = {}
    for j, kappa in enumerate(kappas):
        print(f"{j + 1} / {len(kappas)}: {kappa}")
        logdets = np.zeros((n_mc, N), dtype=float)
        AUC = np.full(n_mc, np.nan, dtype=float)
        TE = np.full(n_mc, np.nan, dtype=float)
        delta = np.full(n_mc, np.nan, dtype=float)

        for i in range(n_mc):
            print(f"{i + 1} / {n_mc}", end="\r")
            out = run_one_trial(p, cfg, K, kappa, rng, record=False)
            logdets[i, :] = out["logdet"]
            AUC[i] = out["AUC"]
            TE[i] = out["TE"]
            delta[i] = out["delta"]

        results[float(kappa)] = {
            "logdet": logdets,
            "AUC": AUC,
            "TE": TE,
            "delta": delta
        }

    return results


def summarize(results, kappas):
    print("\n=== Scenario A summary (medians over Monte Carlo) ===")
    for kappa in kappas:
        kappa = float(kappa)
        T_med = np.nanmedian(results[kappa]["T_eps"])
        TE_med = np.nanmedian(results[kappa]["TE"])
        print(f"kappa={kappa:+.2f} | median T_eps={T_med:.3f} s | median TE={TE_med:.6f}")


def plot_results(cfg: ExperimentConfig, results, kappas, tag: str):
    dt = cfg.dt
    N = int(cfg.T / dt)
    t = np.arange(N) * dt

    # 1) median logdet curves
    plt.figure(figsize=(8.5, 4.8))
    for kappa in kappas:
        kappa = float(kappa)
        logd = results[kappa]["logdet"]
        med = np.nanmedian(logd, axis=0)
        plt.plot(t, med, label=rf"$\kappa={kappa:+.2f}$")
    plt.xlabel("time [s]")
    plt.ylabel(r"median $\log\det P_{\theta}(t)$")
    plt.title("Scenario A: EKF covariance collapse under passive-joint friction uncertainty")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"scenarioA_logdet_median_{tag}.png", dpi=200)

    # 2) boxplot AUC
    plt.figure(figsize=(7.5, 4.8))
    data = [results[float(k)]["AUC"] for k in kappas]
    plt.boxplot(data, labels=[f"{float(k):+.2f}" for k in kappas], showfliers=False)
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"AUC $\int_0^T \log\det P_\theta(t)\,dt$")
    plt.title("Readiness metric: AUC of logdet covariance (lower is better)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"scenarioA_AUC_box_{tag}.png", dpi=200)

    # 3) boxplot TE
    plt.figure(figsize=(7.5, 4.8))
    data = [results[float(k)]["TE"] for k in kappas]
    plt.boxplot(data, labels=[f"{float(k):+.2f}" for k in kappas], showfliers=False)
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$\mathrm{TE}_{2\to 1}$ (log-det score)")
    plt.title("Directed predictability on EKF state-corrections (TE surrogate)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"scenarioA_TE_box_{tag}.png", dpi=200)


def plot_time_domain_examples(cfg, p, K, kappas_to_plot, tag):
    """
    Run one representative trial for each kappa in kappas_to_plot with record=True,
    and plot theta(t) vs theta_hat(t) ±2σ and u(t).
    """
    rng = np.random.default_rng(cfg.seed + 12345)
    dt = cfg.dt
    N = int(cfg.T / dt)
    t = np.arange(N) * dt

    # --- Theta estimation plot ---
    fig, axes = plt.subplots(len(kappas_to_plot), 1, figsize=(9, 3.8 * len(kappas_to_plot)), sharex=True)
    if len(kappas_to_plot) == 1:
        axes = [axes]

    for ax, kappa in zip(axes, kappas_to_plot):
        out = run_one_trial(p, cfg, K, float(kappa), rng, record=True)
        theta_true = out["x_true"][:, 1]
        theta_hat = out["x_hat"][:, 1]
        sig = out["sig_theta"]

        ax.plot(t, theta_true, color="black", linewidth=1.5, label=r"$\theta$ (true)")
        ax.plot(t, theta_hat, color="C0", linewidth=1.2, label=rf"$\hat\theta$ (EKF), $\kappa={kappa:+.2f}$")
        ax.fill_between(t, theta_hat - 2*sig, theta_hat + 2*sig, color="C0", alpha=0.2, label=r"$\pm 2\sigma_\theta$")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("theta [rad]")
        ax.legend(fontsize=9, loc="best")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Time-domain estimation: true vs EKF estimate with uncertainty band", y=0.98)
    fig.tight_layout()
    fig.savefig(f"scenarioA_theta_estimation_{tag}.png", dpi=200)

    
    # --- Control effort plot ---
    plt.figure(figsize=(9, 4.5))
    for kappa in kappas_to_plot:
        out = run_one_trial(p, cfg, K, float(kappa), rng, record=True)
        u = out["u"]
        plt.plot(t, u, label=rf"$u(t)$, $\kappa={kappa:+.2f}$")
    plt.xlabel("time [s]")
    plt.ylabel("u [N·m]")
    plt.title("Control effort (representative trials)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"scenarioA_u_effort_{tag}.png", dpi=200)



def main():
    # --- Load parameters and config (your fixed dt and sigma_phi are already defaults) ---
    p = FurutaParams()
    cfg = ExperimentConfig(dt=0.01, sigma_phi=1e-3)

    
    # Validate open-loop free response near down
    # print("Validating free response...")
    # simulate_free_response(p, x0=[0.0, np.pi - 0.25, 0.0, 0.0], dt=cfg.dt, T=5.0, kappa=0.0, title="Open-loop free response near down (u=0)")


    # --- Linearize continuous-time RHS about down equilibrium for LQR design ---
    x_eq = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)
    u_eq = 0.0
    A, B = linearize_rhs(p, x_eq, u_eq, kappa=0.0, G_shape="const", eps=1e-6)


    # --- LQR design ---
    Q_lqr = np.diag([1.0, 2.0, 5.0, 5.0])
    R_lqr = np.array([[100.0]])
    K = lqr(A, B, Q_lqr, R_lqr)  # shape (1,4)

    # --- Kappa sweep and Monte Carlo count ---
    kappas = list(cfg.kappas)
    
    # Sanity check closed loop deterministic before MC
    # print("Starting sanity check closed loop...")
    # kappas_to_test = kappas
    # sanity_check_closed_loop(p, cfg, K, kappas_to_test=kappas_to_test)

    # --- Run MC ---
    print("Run Monte-Carlo...")
    n_mc = cfg.mc_trials_debug  # switch to cfg.mc_trials_full after debugging
    results = run_monte_carlo(p, cfg, K, kappas, n_mc)

    # --- Print + plot ---
    summarize(results, kappas)
    tag = f"mc{n_mc}"
    plot_results(cfg, results, kappas, tag)
    # kappas_to_plot = [0.0, max(kappas)]
    kappas_to_plot = kappas
    plot_time_domain_examples(cfg, p, K, kappas_to_plot, tag)


    print("\nSaved plots:")
    print(f"  scenarioA_logdet_median_{tag}.png")
    print(f"  scenarioA_Teps_box_{tag}.png")
    print(f"  scenarioA_TE_box_{tag}.png")
    print(f"  scenarioA_theta_estimation_{tag}.png")
    print(f"  scenarioA_u_effort_{tag}.png")
    print("  sanity_theta.png, sanity_u.png")



if __name__ == "__main__":
    main()
