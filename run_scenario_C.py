"""
Scenario C (Step 1 baseline, kappa=0):
- Run an *input replay probing* experiment (exogenous bounded input u(t)).
- Compute structural coupling maps vs theta using:
    (i) local discrete Jacobian block (linearized-along-trajectory)
    (ii) nonlinear finite perturbation gain (central differences)
- Compute informational coupling map vs theta using EKF theta-block entropy reduction per step.

Outputs:
- Figures saved:
    C_input_and_state_timeseries.png
    C_theta_binned_structural_vs_theta.png
    C_theta_binned_info_gain_vs_theta.png
    C_theta_binned_counts.png

Notes:
- This script uses only kappa=0.
- It is intended to validate the hypothesis that coupling/information is reduced near theta≈pi/2.
- For comparability, it uses substepped RK4 integration (n_sub=20).
"""


import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from io import BytesIO
import json
from pathlib import Path

from config import FurutaParams, ExperimentConfig
from furuta_model import rhs_continuous, rk4_step, wrap_angle, wrap_center, b_theta_true, G_kappa
from ekf import EKF, EKF_FourierFriction, R_from_sigma_phi, measure_phi, derive_phidot, b_theta_hat_fourier, grad_b_hat

from info_metrics import te_logdet, logdet_theta_block

from coupling_metrics import (
    plant_step_substeps,
    structural_coupling_metrics,
    ekf_information_gain_step,
    compute_theta_binned_maps,
    theta_binned_te
)


def generate_multisine_u(dt, T, amp=1.0, freqs=(0.3, 0.7, 1.1), seed=0, plot=False):
    """Band-limited multisine probing signal."""
    rng = np.random.default_rng(seed)
    N = int(T/dt)
    t = np.arange(N) * dt
    phases = rng.uniform(0.0, 2.0*np.pi, size=len(freqs))
    u = np.zeros(N)
    for w, ph in zip(freqs, phases):
        u += np.sin(2.0*np.pi*w*t + ph)
    u = amp * u / max(1e-12, np.max(np.abs(u)))
    if plot:
        sp = np.fft.fft(np.sin(t))
        freq = np.fft.fftfreq(t.shape[-1], d=dt)
        
        plt.figure()
        plt.subplot(121)
        plt.plot(t, u)
        plt.xlabel('t [s]')
        plt.ylabel('u(t)')
        plt.subplot(122)
        plt.plot(freq, np.abs(sp))
        plt.xlabel('f [Hz]')
        plt.ylabel('|U(f)|')
        plt.show()

    return t, u


def run_probe_rollout(p, cfg, u, kappa=0.0, n_sub=20, measurement_dim=1, measurement_noise=True, friction_uncertainty=False, seed=1, G_shape='const'):
    """Simulate plant and EKF under exogenous input u(t)."""
    dt = cfg.dt
    N = int(cfg.T/dt)
    rng = np.random.default_rng(seed)

    x_true = np.asarray(cfg.x0_true, dtype=float).copy()

    # Friction parameters
    b0_nom = p.b0_nom
    b1_nom = p.b1_nom

    # R = R_from_sigma_phi(cfg.sigma_phi, cfg.dt)
    # ekf = EKF(cfg.x0_hat, cfg.P0.copy(), cfg.Q.copy(), R, dt, p, kappa=float(kappa), G_shape=G_shape)
    
    # --- Augmented EKF: x_aug = [x_phys, c0, c1, s1, c2, s2] ---
    # initial coeffs: start with constant friction guess, others zero
    coeff0 = np.array([b0_nom, 0.0, 0.0, 0.0, 0.0], dtype=float)
    x0_aug = np.hstack([cfg.x0_hat.astype(float), coeff0])

    # build augmented P0 and Q
    P0_aug = np.zeros((9, 9), dtype=float)
    P0_aug[:4, :4] = cfg.P0.copy()
    P0_aug[4:, 4:] = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3])  # uncertainty on coefficients
    
    Q_aug = np.zeros((9, 9), dtype=float)
    Q_aug[:4, :4] = cfg.Q.copy()
    Q_aug[4:, 4:] = np.diag([1e-7, 1e-7, 1e-7, 1e-7, 1e-7])  # random-walk for coeffs

    # update of EKF every N_update steps
    N_update = max(1, int(round(cfg.update_period / dt)))
    dt_update = N_update * dt

    # dim = 1: measure phi / dim = 2: measure phi and phidot
    # measurement_dim = 1
    R = R_from_sigma_phi(cfg.sigma_phi, dt_update, dim=measurement_dim)
    ekf = EKF_FourierFriction(x0_aug, P0_aug, Q_aug, R, dt, p, kappa=float(kappa), G_shape=G_shape)
    print("DEBUG: plant G_shape =", G_shape, "EKF G_shape =", ekf.G_shape, "kappa =", kappa)

    # measurement memory for correlated phidot
    if measurement_noise:
        phi_meas_prev = float(measure_phi(x_true[0], cfg.sigma_phi, rng))
    else:
        phi_meas_prev = float(x_true[0])

    X = np.zeros((N, 4))
    Xhat = np.zeros((N, 4))
    logdetP = np.zeros(N)
    info_gain = np.zeros(N)

    # structural metrics per step
    S_lin = np.zeros(N)
    S_nonlin = np.zeros(N)

    # store EKF corrections if needed later
    dx_series = np.zeros((N, 4))
    dx_last = None # store the last dx from the EKF update

    # store update info
    coeff_hist = []
    Pcc_hist = []
    t_updates = []


    for k in range(N):
        print(f"{k + 1} / {N}", end="\r")
        
        if k in np.arange(N_update, N, 50):
            print("k", k,
                    "theta_true", x_true[1],
                    "theta_hat", ekf.x[1],
                    "sin_true", np.sin(x_true[1]),
                    "sin_hat", np.sin(ekf.x[1]))


        X[k] = x_true
        Xhat[k] = ekf.x[:4]
        logdetP[k] = float(np.nan)

        # Structural coupling metrics at current state, using current input
        sm = structural_coupling_metrics(x_true, u[k], dt, n_sub, p, kappa,
                                        rhs_continuous=rhs_continuous, rk4_step=rk4_step,
                                        eps=1e-6, norm='sv', perturb='basis', G_shape=G_shape)
        S_lin[k] = sm['S_lin']
        S_nonlin[k] = sm['S_nonlin']

        # Predict
        _, _, F = ekf.predict(u[k])
        P_pred = ekf.P_pred.copy()

        # friction mismatch (optional)
        if friction_uncertainty:
            delta0 = rng.uniform(-cfg.friction_uncertainty, cfg.friction_uncertainty)
            delta1 = rng.uniform(-cfg.friction_uncertainty, cfg.friction_uncertainty)
            b0_true = b0_nom * (1.0 + delta0)
            b1_true = b1_nom * (1.0 + delta1)
        else:
            b0_true, b1_true = b0_nom, b1_nom

        # Plant step
        x_true = plant_step_substeps(
            x_true, u[k], dt, n_sub, p, kappa, 
            rhs_continuous, rk4_step, 
            b_theta_true=lambda theta: b_theta_true(theta, b0_true, b1_true), G_shape=G_shape
            )
        if not np.all(np.isfinite(x_true)):
            # truncate
            X = X[:k+1]
            Xhat = Xhat[:k+1]
            logdetP = logdetP[:k+1]
            info_gain = info_gain[:k+1]
            S_lin = S_lin[:k+1]
            S_nonlin = S_nonlin[:k+1]
            dx_series = dx_series[:k+1]
            u = u[:k+1]
            break

        # measurement update only every N_update steps
        if (k % N_update) == 0:
            # Measurement
            if measurement_dim == 1:
                phi_meas = float(measure_phi(x_true[0], cfg.sigma_phi, rng))
                z = np.array([phi_meas], dtype=float)
            elif measurement_dim == 2:
                if measurement_noise:
                    phi_meas = float(measure_phi(x_true[0], cfg.sigma_phi, rng))
                    phidot_meas = float(derive_phidot(phi_meas, phi_meas_prev, dt_update))
                    phi_meas_prev = phi_meas
                else:
                    phi_meas = float(x_true[0])
                    phidot_meas = float(x_true[2])
                    phi_meas_prev = phi_meas
                z = np.array([phi_meas, phidot_meas], dtype=float)
            else:
                raise ValueError("measurement_dim should be one or two.")
            
            # Update
            innov, dx = ekf.update(z)
            if ekf.x.size >= 9: # augmented EKF
                coeff_hist.append(ekf.x[4:9].copy())
                Pcc_hist.append(ekf.P[4:9, 4:9].copy())
                t_updates.append(k*dt)
            
            # Info gain in theta-block
            info_gain[k] = ekf_information_gain_step(P_pred, ekf.P, idx_theta=(1,3))
            dx_last = dx

            # DEBUG EKF
            if not np.isfinite(info_gain[k]):
                # check whether it was P_pred or P_upd
                blk_pred = P_pred[np.ix_([1,3],[1,3])]
                blk_upd  = ekf.P[np.ix_([1,3],[1,3])]
                finite_pred = np.all(np.isfinite(blk_pred))
                finite_upd  = np.all(np.isfinite(blk_upd))
                # eigenvalues are a good PSD indicator
                eig_pred = np.linalg.eigvalsh(0.5*(blk_pred+blk_pred.T)) if finite_pred else np.array([np.nan, np.nan])
                eig_upd  = np.linalg.eigvalsh(0.5*(blk_upd+blk_upd.T)) if finite_upd else np.array([np.nan, np.nan])

                nan_reason = []
                if not finite_pred: 
                    nan_reason.append("P_pred_nonfinite")
                    print("phi, theta, phidot, thetadot =", X[k])             # true state
                    print("ekf.x =", ekf.x)                                   # estimated state
                    print("phi wrapped?", ekf.x[0], "theta wrapped?", ekf.x[1])
                    print("abs(phi) close to pi?", abs(wrap_angle(ekf.x[0])) > 3.0)
                    print("max|F|=", np.nanmax(np.abs(F)))#, "cond(F)=", np.linalg.cond(F))

                if not finite_upd:  nan_reason.append("P_upd_nonfinite")
                if finite_pred and np.min(eig_pred) <= 0: nan_reason.append("P_pred_not_PSD")
                if finite_upd and np.min(eig_upd) <= 0:  nan_reason.append("P_upd_not_PSD")

                # log the theta and rates at this moment (helps correlate with bins)
                print(f"NaN info_gain at k={k}, theta={X[k,1]:.3f}, thetadot={X[k,3]:.3f}, reasons={nan_reason}")
        else:
            # no measurement update at this step
            ekf.x = ekf.x_pred
            ekf.P = ekf.P_pred
            innov = None
            dx = np.zeros_like(dx_series[k])
            info_gain[k] = np.nan

        dx_series[k] = dx[:4] # only [phi, theta, phidot, thetadot]

        # logdet after update
        # (reuse info_metrics.logdet_theta_block if you prefer; keep local here)
        blk = ekf.P[np.ix_([1,3],[1,3])]
        sign, ld = np.linalg.slogdet(0.5*(blk+blk.T) + 1e-12*np.eye(2))
        logdetP[k] = ld if sign > 0 else np.nan

        
    # global TE baseline
    print("last dx = ", dx_last) # last EKF update - all coefficients
    
    P = ekf.P
    P_cc = P[4:9, 4:9]          # coeff covariance
    P_cy = P[4:9, :][:, [0,2]]  # coeff vs measured-state covariance (phi, phidot)
    print("diag(P) = ", np.diag(P))
    print("diag(P_cc) =", np.diag(P_cc))
    print("norm(P_cy) =", np.linalg.norm(P_cy))

    dx = dx_series
    start_idx = int(cfg.te_start_time / cfg.dt)
    nu1 = dx[start_idx:, [0,2]]
    nu2 = dx[start_idx:, [1,3]]
    TE_global = float(te_logdet(nu1, nu2, cfg.te_lag)[0])
    print("Global TE_{2->1}:", TE_global)

    kappa_str = "{:.2f}".format(kappa).replace('.', '_')
    coeffs_hat = ekf.x[4:9]
    P_cc = ekf.P[4:9, 4:9]
    plot_damping_function(coeffs_hat, P_cc, b0_true, b1_true, fname="C_damping_function_fit_{}_{}_init_pi.png".format(G_shape, kappa_str))

    coeff_hist = np.asarray(coeff_hist) if coeff_hist else None
    Pcc_hist = np.asarray(Pcc_hist) if Pcc_hist else None
    t_updates = np.asarray(t_updates) if t_updates else None

    make_damping_gif(
        t_updates, coeff_hist, Pcc_hist,
        b0_true=b0_true, b1_true=b1_true, gif_path="C_gif_damping_function_fit_{}_{}_init_pi.gif".format(G_shape, kappa_str)
    )

    plot_coeff_uncertainty_and_mean(
        t_updates, coeff_hist, Pcc_hist,
        b0_true=b0_true, b1_true=b1_true,
        fname="C_damping_coeff_learning_summary_{}_{}".format(G_shape, kappa_str)
    )

    return X, Xhat, u, S_lin, S_nonlin, info_gain, logdetP, dx, TE_global


def binned_2d_median(theta, v, values, theta_bins=31, v_bins=25,
                     theta_range=(-np.pi, np.pi), v_range=(0.0, None),
                     min_count=10):
    """
    Compute 2D binned median and counts for values over (theta, v).

    Parameters
    ----------
    theta : (N,) array
        Angle values (will be wrapped to (-pi,pi] outside if needed).
    v : (N,) array
        Nonnegative velocity magnitude (e.g., abs(thetadot)).
    values : (N,) array
        Quantity to bin (e.g., per-step info gain).
    theta_bins : int
    v_bins : int
    theta_range : (min, max)
    v_range : (min, max) where max can be None -> set to max(v finite)
    min_count : int
        Cells with counts < min_count are set to NaN in the median map.

    Returns
    -------
    theta_centers, v_centers, med_map, count_map
    """
    theta = np.asarray(theta, float)
    v = np.asarray(v, float)
    values = np.asarray(values, float)

    # filter finite
    m = np.isfinite(theta) & np.isfinite(v) & np.isfinite(values)
    theta = theta[m]
    v = v[m]
    values = values[m]

    # ranges
    tmin, tmax = theta_range
    vmin, vmax = v_range
    if vmax is None:
        vmax = np.nanmax(v) if v.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # bin edges
    t_edges = np.linspace(tmin, tmax, theta_bins + 1)
    v_edges = np.linspace(vmin, vmax, v_bins + 1)

    # bin indices
    ti = np.digitize(theta, t_edges) - 1
    vi = np.digitize(v, v_edges) - 1
    ti = np.clip(ti, 0, theta_bins - 1)
    vi = np.clip(vi, 0, v_bins - 1)

    # allocate
    med_map = np.full((v_bins, theta_bins), np.nan)
    count_map = np.zeros((v_bins, theta_bins), dtype=int)
    
    # collect per cell
    # Using lists is fine at this size; can optimize later if needed.
    cell_lists = [[[] for _ in range(theta_bins)] for __ in range(v_bins)]
    for a, b, val in zip(vi, ti, values):
        cell_lists[a][b].append(val)

    for a in range(v_bins):
        for b in range(theta_bins):
            vals = np.array(cell_lists[a][b], dtype=float)
            count_map[a, b] = vals.size
            if vals.size >= min_count:
                med_map[a, b] = np.median(vals)

    theta_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    return theta_centers, v_centers, med_map, count_map




def plot_and_save(cfg, t, u, X, maps, prefix='C', min_count=10):
    """Generate and save plots."""
    # Time series plot
    plt.figure(figsize=(11, 7))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t[:len(u)], u, linewidth=1.0)
    ax1.set_ylabel('u [N·m]')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.plot(t[:len(X)], wrap_center(X[:,0], 0.0), label='phi')
    ax2.plot(t[:len(X)], wrap_center(X[:,1], np.pi), label='theta')
    ax2.set_ylabel('angles [rad]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    ax3.plot(t[:len(X)], X[:,2], label='phidot')
    ax3.plot(t[:len(X)], X[:,3], label='thetadot')
    ax3.set_xlabel('time [s]')
    ax3.set_ylabel('rates [rad/s]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{prefix}_input_and_state_timeseries.png', dpi=200)
    plt.close()

    th = maps['theta_centers']

    # Structural coupling plot
    plt.figure(figsize=(10,4.5))
    plt.plot(th, maps['S_lin_med'], label='S_lin (Jacobian block)', linewidth=1.5)
    plt.fill_between(th, maps['S_lin_q1'], maps['S_lin_q3'], alpha=0.2)
    plt.plot(th, maps['S_nonlin_med'], label='S_nonlin (finite perturbation)', linewidth=1.5)
    plt.fill_between(th, maps['S_nonlin_q1'], maps['S_nonlin_q3'], alpha=0.2)
    plt.xlabel('theta [rad] (wrapped)')
    plt.ylabel('structural coupling strength')
    plt.title('Theta-binned structural coupling (kappa=0)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_theta_binned_structural_vs_theta.png', dpi=200)
    plt.close()

    # Information gain plot
    print("I_med: ", maps['I_med'])
    I_med = maps["I_med"]
    I_q1 = maps["I_q1"]
    I_q3 = maps["I_q3"]
    cnt = maps["counts"]

    mask = (cnt >= min_count) & np.isfinite(I_med) & np.isfinite(I_q1) & np.isfinite(I_q3)
    I_med[~mask] = np.nan
    I_q1[~mask] = np.nan
    I_q3[~mask] = np.nan

    plt.figure(figsize=(10,4.5))
    plt.plot(th, I_med, label='Info gain (EKF theta-block)', linewidth=1.5)
    plt.fill_between(th, I_q1, I_q3, alpha=0.2)
    plt.xlabel('theta [rad] (wrapped)')
    plt.ylabel('per-step info gain')
    plt.title('Theta-binned EKF information gain (kappa=0)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_theta_binned_info_gain_vs_theta.png', dpi=200)
    plt.close()

    # Bin counts plot
    plt.figure(figsize=(10,3.8))
    plt.bar(th, maps['counts'], width=(th[1]-th[0]) if len(th) > 1 else 0.1)
    plt.xlabel('theta [rad] (wrapped)')
    plt.ylabel('samples per bin')
    plt.title('Theta bin sample counts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{prefix}_theta_binned_counts.png', dpi=200)
    plt.close()


def plot_info_gain_heatmap(theta_centers, v_centers, med_map, count_map,
                           title="2D map: median info gain",
                           fname="C_info_gain_heatmap_theta_thetadot.png",
                           min_count=10):
    """
    Plot heatmap of median info gain over (theta, |thetadot|) bins.
    """
    plt.figure(figsize=(10, 5))

    # extent expects [xmin, xmax, ymin, ymax]
    extent = [theta_centers[0], theta_centers[-1], v_centers[0], v_centers[-1]]

    im = plt.imshow(med_map, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(im, label="median per-step info gain")

    plt.xlabel("theta [rad] (wrapped)")
    plt.ylabel("|thetadot| [rad/s]")
    plt.title(title)

    # optional: contour of counts to show data support
    # draw a contour where count >= min_count
    mask = (count_map >= min_count).astype(float)
    # contour levels at 0.5 separates valid/invalid
    plt.contour(theta_centers, v_centers, mask, levels=[0.5], colors="w", linewidths=1.0)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def plot_TE(start_idx, theta_wrapped, dx_series, te_lag, min_count=10):
    """
    Plot the theta-binned directional TE (baseline, kappa=0).
    """
    # Slice to remove the transient
    theta_used = theta_wrapped[start_idx:]
    dx_used = dx_series[start_idx:, :4]  # ensure only physical corrections are used

    # Define TE function (returns scalar TE)
    te_func = lambda a, b, k: float(te_logdet(a, b, k)[0])

    # Compute theta-binned TE in both directions:
    #   TE_2to1: dx2 -> dx1  (theta-block -> phi-block)
    #   TE_1to2: dx1 -> dx2  (phi-block -> theta-block)
    theta_centers_te, te_2to1, te_1to2, te_counts = theta_binned_te(
        dx_series=dx_used,
        theta=theta_used,
        nbins=31,
        theta_min=-np.pi,
        theta_max=np.pi,
        te_lag=te_lag,
        min_count=25,                          # increase/decrease depending on how long your run is
        min_seg_len=max(10, te_lag + 5),
        te_func=te_func
    )

    mask = (te_counts >= min_count) & np.isfinite(te_1to2) & np.isfinite(te_2to1)

    te_2to1[~mask] = np.nan
    te_1to2[~mask] = np.nan
    
    plt.figure(figsize=(10, 4.8))
    plt.plot(theta_centers_te, te_2to1, linewidth=1.6, label="TE 2→1 (θ-block → φ-block)")
    plt.plot(theta_centers_te, te_1to2, linewidth=1.6, label="TE 1→2 (φ-block → θ-block)")
    plt.xlabel("theta [rad] (wrapped)")
    plt.ylabel("TE (log-det VAR score)")
    plt.title("Theta-binned directional TE (kappa=0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("C_theta_binned_TE_directional.png", dpi=200)
    plt.close()

    # ---- Plot counts used per bin (important for interpretability) ----
    plt.figure(figsize=(10, 3.8))
    bar_width = (theta_centers_te[1] - theta_centers_te[0]) if len(theta_centers_te) > 1 else 0.1
    plt.bar(theta_centers_te, te_counts, width=bar_width)
    plt.xlabel("theta [rad] (wrapped)")
    plt.ylabel("samples used")
    plt.title("Samples per theta-bin used for TE computation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("C_theta_binned_TE_directional_counts.png", dpi=200)
    plt.close()


    return theta_centers_te, te_2to1, te_1to2, te_counts


def plot_state_estimates_time(t, X_true, X_hat, fname="C_state_estimation_time.png"):
    """
    Plots true vs estimated states and errors over time.
    X_true: (N,4), X_hat: (N,4) or (N,>=4)
    """
    X_true = np.asarray(X_true, float)
    X_hat = np.asarray(X_hat, float)
    if X_hat.shape[1] > 4:
        X_hat = X_hat[:, :4]

    phi, theta, phidot, thetadot = X_true.T
    phi_h, theta_h, phidot_h, thetadot_h = X_hat.T

    e_phi = wrap_angle(phi - phi_h)
    e_theta = wrap_angle(theta - theta_h)
    e_phidot = phidot - phidot_h
    e_thetadot = thetadot - thetadot_h

    fig, axs = plt.subplots(4, 2, figsize=(12, 9), sharex=True)

    # True vs Hat
    axs[0,0].plot(t, phi, label="phi true")
    axs[0,0].plot(t, phi_h, "--", label="phi hat")
    axs[1,0].plot(t, theta, label="theta true")
    axs[1,0].plot(t, theta_h, "--", label="theta hat")
    axs[2,0].plot(t, phidot, label="phidot true")
    axs[2,0].plot(t, phidot_h, "--", label="phidot hat")
    axs[3,0].plot(t, thetadot, label="thetadot true")
    axs[3,0].plot(t, thetadot_h, "--", label="thetadot hat")

    # Errors
    axs[0,1].plot(t, e_phi, label="wrap(phi - phi_hat)")
    axs[1,1].plot(t, e_theta, label="wrap(theta - theta_hat)")
    axs[2,1].plot(t, e_phidot, label="phidot - phidot_hat")
    axs[3,1].plot(t, e_thetadot, label="thetadot - thetadot_hat")
    
    labels = ["phi [rad]", "theta [rad]", "phidot [rad/s]", "thetadot [rad/s]"]
    for i in range(4):
        axs[i,0].set_ylabel(labels[i])
        axs[i,0].grid(True, alpha=0.3)
        axs[i,0].legend(fontsize=8, loc="best")

        axs[i,1].grid(True, alpha=0.3)
        axs[i,1].legend(fontsize=8, loc="best")

    axs[-1,0].set_xlabel("time [s]")
    axs[-1,1].set_xlabel("time [s]")

    fig.suptitle("State estimation: true vs EKF estimate (left) and errors (right)", y=0.98)
    fig.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_damping_function(coeffs_hat, P_cc, b0_true, b1_true, theta_grid=None, fname="C_damping_function_fit.png", axs=None, suptitle=None):
    """
    Plot:
      (1) true b(theta)=b0+b1*cos^2(theta) vs estimated b_hat(theta) (Fourier M=2) with ±2σ band
      (2) Fourier coefficients vs their true values (implied by b0+b1*cos^2), with ±2σ per coefficient
      (3) θ-dependent variance contribution of each coefficient to Var[b_hat(theta)]

    coeffs_hat: array-like shape (5,)
    P_cc: array-like shape (5,5) covariance for [c0,c1,s1,c2,s2]

    """
    if theta_grid is None:
        theta_grid = np.linspace(-np.pi, np.pi, 400)

    theta_grid = np.asarray(theta_grid, float)
    coeffs_hat = np.asarray(coeffs_hat, float).reshape(5,)
    P_cc = np.asarray(P_cc, float).reshape(5, 5)

    # True friction function
    b_true = b0_true + b1_true * (np.cos(theta_grid)**2)
    # Estimated friction function
    b_est = b_theta_hat_fourier(theta_grid, coeffs_hat)

    # uncertainty band from coeff covariance
    G = grad_b_hat(theta_grid)  # (N,5)
    # var = diag(G P G^T)
    var = np.einsum("ni,ij,nj->n", G, P_cc, G)
    var = np.maximum(var, 0.0)
    sigma_b = np.sqrt(var)
    
    # --- "True" coefficients implied by b0 + b1 cos^2(theta) ---
    # b0 + b1 cos^2(theta) = (b0 + b1/2) + (b1/2) cos(2theta)
    c_true = np.zeros(5, dtype=float)
    c_true[0] = b0_true + 0.5 * b1_true  # c0
    c_true[3] = 0.5 * b1_true            # c2
    # c1, s1, s2 are 0

    # --- Coefficient uncertainties ---
    sigma_c = np.sqrt(np.maximum(np.diag(P_cc), 0.0))

    # --- θ-dependent variance contribution per coefficient (diagonal-only decomposition) ---
    # Var[b(theta)] = sum_j g_j(theta)^2 * Var(c_j) + cross-terms
    # We'll show the diagonal contributions to explain "why band stays wide"
    var_terms = (G**2) * np.diag(P_cc)[None, :]  # (N,5)
    var_terms = np.maximum(var_terms, 0.0)

    names = ["c0", "c1", "s1", "c2", "s2"]

    
    # ---------------- Plot layout ----------------
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(10, 9.6), sharex=False)
    else:
        fig = axs[0].figure
        for ax in axs:
            ax.cla()

    # (1) b(theta): truth vs estimate
    ax = axs[0]
    ax.plot(theta_grid, b_true, linewidth=2.0, label="true b(θ)=b0+b1 cos²(θ)")
    ax.plot(theta_grid, b_est, linewidth=1.8, label="estimated b̂(θ) (Fourier M=2)")
    ax.fill_between(theta_grid, b_est - 2*sigma_b, b_est + 2*sigma_b,
                    alpha=0.20, label="±2σ (from P_cc)")
    ax.set_ylabel("b_θ(θ) [N·m·s/rad]")
    ax.set_title("Directional friction function: truth vs EKF estimate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (2) coefficients vs truth with ±2σ error bars
    ax = axs[1]
    x = np.arange(5)
    ax.errorbar(x, coeffs_hat, yerr=2*sigma_c, fmt="o", capsize=4, label="estimated coeffs ±2σ")
    ax.plot(x, c_true, "s", markersize=6, label="true coeffs (implied by b0+b1 cos²)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("coefficient value")
    ax.set_title("Fourier coefficients: estimate vs truth (with uncertainty)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    
    # (3) variance contributions across θ
    ax = axs[2]
    # Plot diagonal contributions; optionally also plot total var
    for j, nm in enumerate(names):
        ax.plot(theta_grid, var_terms[:, j], linewidth=1.4, label=f"Var contrib: {nm}")
    ax.plot(theta_grid, var, color="black", linewidth=2.0, alpha=0.7, label="Total Var (incl. cross-terms)")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("variance contribution")
    ax.set_title("Uncertainty decomposition of b̂(θ) from coefficient covariance")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()

    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)

def make_damping_gif(t_updates, coeff_hist, Pcc_hist, b0_true, b1_true, theta_grid=None, gif_path="C_gif_damping_function_fit.gif", fps=12):
    if theta_grid is None:
        theta_grid = np.linspace(-np.pi, np.pi, 400)

    fig, axs = plt.subplot(3, 1, figsize=(10, 9.6))
    frames = []
    
    for k in range(len(t_updates)):
        plot_damping_function(
            coeff_hist[k],
            Pcc_hist[k],
            b0_true, b1_true,
            theta_grid=theta_grid,
            axs=axs,
            suptitle=f"EKF friction learning - update {k+1}/{len(t_updates)}, t={float(t_updates[k]):.2f}s"
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
    
    plt.close(fig)
    imageio.mimsave(gif_path, frames, fps=fps)
    print("Saved GIF of daming function: ", gif_path)

def plot_coeff_uncertainty_and_mean(t_updates, coeff_hist, Pcc_hist, b0_true, b1_true, fname="C_damping_coeff_learning_summary.png"):
    
    names = ["c0", "c1", "s1", "c2", "s2"]

    # 2σ over time
    sigma2 = 2.0 * np.sqrt(np.maximum(
        np.diagonal(Pcc_hist, axis1=1, axis2=2), 0.0
    ))

    # true coefficients implied by b0 + b1 cos^2
    c_true = np.zeros(5)
    c_true[0] = b0_true + 0.5*b1_true
    c_true[3] = 0.5*b1_true

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # (1) uncertainty
    for j, name in enumerate(names):
        axs[0].plot(t_updates, sigma2[:, j], label=f"2σ({name})")
    axs[0].set_ylabel("2σ")
    axs[0].set_title("Coefficient uncertainty vs time")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(ncol=3)

    # (2) mean
    for j, name in enumerate(names):
        axs[1].plot(t_updates, coeff_hist[:, j], label=f"{name} hat")
        axs[1].axhline(c_true[j], linestyle="--", alpha=0.5)
        
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("coefficient value")
    axs[1].set_title("Coefficient estimates vs time (dashed = true)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(ncol=3)

    fig.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved:", fname)



def main():
    p = FurutaParams()
    cfg = ExperimentConfig()
    measurement_noise = True
    friction_uncertainty = True

    # baseline kappa
    kappa = 0.0

    # probing input
    t, u = generate_multisine_u(cfg.dt, cfg.T, amp=0.75, freqs=(0.2, 0.35, 0.6, 0.9), seed=0)

    X, Xhat, u_used, S_lin, S_nonlin, info_gain, logdetP, dx, TE_global, frict_coeff_hist = run_probe_rollout(
        p, cfg, u, kappa=kappa, 
        n_sub=20, measurement_noise=measurement_noise, 
        friction_uncertainty=friction_uncertainty,
        seed=cfg.seed
    )

    # theta-binned maps
    maps = compute_theta_binned_maps(
        time=t[:len(X)],
        X_true=X,
        U=u_used,
        S_lin=S_lin,
        S_nonlin=S_nonlin,
        info_gain=info_gain,
        nbins=31,
        theta_wrap=True,
        theta_min=-np.pi,
        theta_max=np.pi
    )

    theta = wrap_angle(X[:, 1])
    theta_wrapped = wrap_center(X[:, 1], 0.0)
    v = np.abs(X[:, 3])  # |thetadot|

    theta_centers, v_centers, med_map, count_map = binned_2d_median(
        theta=theta,
        v=v,
        values=info_gain,
        theta_bins=31,
        v_bins=25,
        theta_range=(-np.pi, np.pi),
        v_range=(0.0, None),
        min_count=5
    )
    
    plot_and_save(cfg, t, u_used, X, maps, prefix='C', min_count=25)
    plot_info_gain_heatmap(
        theta_centers, v_centers, med_map, count_map,
        title="Median EKF per-step info gain vs theta and |thetadot| (kappa=0)",
        fname="C_info_gain_heatmap_theta_thetadot.png",
        min_count=5
    )

    # --- theta-binned TE ---
    # Use the same burn-in as elsewhere (skip the first second, typically)
    start_idx = int(cfg.te_start_time / cfg.dt)

    centers_te, TE_2to1, TE_1to2, te_counts = plot_TE(start_idx, theta_wrapped, dx, cfg.te_lag, min_count=25)

    plot_state_estimates_time(t[:len(X)], X, Xhat, fname="C_state_estimation_time.png")

    print('Scenario C baseline maps created and saved:')
    print('  C_input_and_state_timeseries.png')
    print('  C_theta_binned_structural_vs_theta.png')
    print('  C_theta_binned_info_gain_vs_theta.png')
    print('  C_theta_binned_counts.png')
    print('  C_info_gain_heatmap_theta_thetadot.png')
    print("  C_theta_binned_TE_directional.png")
    print("  C_theta_binned_TE_directional_counts.png")
    print("  C_state_estimation_time.png")


    # Storing data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    out_npz = data_dir / f"ScenarioC_baseline_kappa0_seed{cfg.seed}.npz"
    out_json = data_dir / f"ScenarioC_baseline_kappa0_seed{cfg.seed}.json"

    coeff_hist, Pcc_hist, t_updates = frict_coeff_hist
    
    np.savez_compressed(
        out_npz,
        t=t[:len(X)],
        u=u_used,
        X_true=X,
        X_hat=Xhat,
        S_lin=S_lin,
        S_nonlin=S_nonlin,
        info_gain=info_gain,
        logdetP=logdetP,
        dx_series=dx,
        theta_wrapped=theta_wrapped,
        thetadot_abs=np.abs(X[:,3]),
        theta_centers=maps['theta_centers'],
        counts=maps['counts'],
        S_lin_med=maps['S_lin_med'],
        S_nonlin_med=maps['S_nonlin_med'],
        I_med=maps['I_med'],
        te_global=TE_global,
        te_centers=centers_te,
        te_2to1=TE_2to1,
        te_1to2=TE_1to2,
        te_counts=te_counts,
        coeff_hist=coeff_hist,
        Pcc_hist=Pcc_hist,
        t_updates=t_updates,
    )

    meta = {
        "scenario": "C_baseline",
        "kappa": 0.0,
        "seed": cfg.seed,
        "dt": cfg.dt,
        "update_period": cfg.update_period,
        "T": cfg.T,
        "sigma_phi": cfg.sigma_phi,
        "te_lag": cfg.te_lag,
        "te_start_time": cfg.te_start_time,
        "Q_diag": np.diag(cfg.Q).tolist(),
        "P0_diag": np.diag(cfg.P0).tolist(),
        "x0_true": cfg.x0_true.tolist(),
        "x0_hat": cfg.x0_hat.tolist(),
        "probe": {"type": "multisine", "amp": 1.0, "freqs": [0.2, 0.35, 0.6, 0.9]},
        "mearurement_noise": measurement_noise,
        "measurement_noise_stddev": cfg.sigma_phi,
        "friction_uncertainty": friction_uncertainty,
        "max_friction_uncertainty": cfg.friction_uncertainty,
    }
    
    out_json.write_text(json.dumps(meta, indent=2))
    print("Saved baseline:", out_npz, "and", out_json)





if __name__ == '__main__':
    main()
