# debug_lqr_chain.py
#
# Standalone diagnostics for the Furuta LQR + EKF chain (no uncertainty, no saturation by default).
# Goals:
#  1) Validate linearization (A,B), controllability, eigenvalues
#  2) Validate LQR on the LINEAR model (dx dynamics)
#  3) Validate nonlinear closed-loop with TRUE state feedback (no EKF)
#  4) Validate nonlinear closed-loop with EKF (perfect measurements, and optionally noisy)
#
# Plots are displayed.

import numpy as np
import matplotlib.pyplot as plt
import scipy

from config import FurutaParams, ExperimentConfig
from furuta_model import rk4_step, rhs_continuous
from control import linearize_rhs, lqr
from ekf import EKF, R_from_sigma_phi


# -----------------------------
# Angle / state error helpers
# -----------------------------

def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def state_error(x, x_ref):
    """
    dx = x - x_ref with wrapped angular errors for phi and theta.
    x = [phi, theta, phidot, thetadot]
    """
    x = np.asarray(x, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    dx = x - x_ref
    dx[0] = wrap_pi(dx[0])
    dx[1] = wrap_pi(dx[1])
    return dx

def controllability_rank(A, B, tol=1e-10):
    """Rank of controllability matrix [B, AB, A^2B, ..., A^{n-1}B]."""
    n = A.shape[0]
    C = B
    AB = B
    for _ in range(1, n):
        AB = A @ AB
        C = np.hstack((C, AB))
    return np.linalg.matrix_rank(C, tol=tol)


def step_with_substeps(x, u, dt, n_sub, p, kappa):
    dt_int = dt / n_sub
    for _ in range(n_sub):
        x = rk4_step(rhs_continuous, x, u, dt_int, p, kappa=kappa, G_shape="const", b_theta_true=p.b_theta_nom)
    return x



# -----------------------------
# Simulation: linear closed-loop
# -----------------------------

def simulate_linear_cl(A, B, K, dx0, dt, T):
    """
    Simulate linear closed-loop dynamics:
        d(dx)/dt = (A - B K) dx
    Forward Euler is fine for debugging.
    """
    N = int(T / dt)
    t = np.arange(N) * dt
    dx = np.asarray(dx0, dtype=float).copy()

    Acl = A - B @ K
    Ad = scipy.linalg.expm(Acl*dt)

    DX = np.zeros((N, len(dx0)), dtype=float)
    U = np.zeros(N, dtype=float)

    for k in range(N):
        DX[k] = dx
        u_min = K @ dx
        u = -float(u_min[0])
        U[k] = u
        dx = Ad @ dx

        if not np.all(np.isfinite(dx)):
            print("[Linear CL] Non-finite state encountered; stopping.")
            DX = DX[:k+1]
            U = U[:k+1]
            t = t[:k+1]
            break

    return t, DX, U


# -----------------------------
# Simulation: nonlinear closed-loop (true state feedback)
# -----------------------------

def simulate_nonlinear_true_state(p, K, x_ref, x0, dt, T, kappa, u_max=None):
    """
    Nonlinear closed-loop simulation with TRUE state feedback:
        u = -K * (x - x_ref)
    """
    N = int(T / dt)
    t = np.arange(N) * dt

    x = np.asarray(x0, dtype=float).copy()

    X = np.zeros((N, 4), dtype=float)
    U = np.zeros(N, dtype=float)

    for k in range(N):
        X[k] = x
        dx = state_error(x, x_ref)
        u_min = K @ dx
        u = -float(u_min[0])
        if u_max is not None:
            u = float(np.clip(u, -u_max, u_max))
        U[k] = u

        x = rk4_step(rhs_continuous, x, u, dt, p,
                     kappa=float(kappa), G_shape="const", b_theta_true=p.b_theta_nom)

        if not np.all(np.isfinite(x)):
            print("[Nonlinear true-state CL] Non-finite state encountered; stopping.")
            X = X[:k+1]
            U = U[:k+1]
            t = t[:k+1]
            break

    return t, X, U


# -----------------------------
# Simulation: nonlinear closed-loop (EKF feedback)
# -----------------------------

def simulate_nonlinear_ekf(p, K, x_ref, x0, dt, T, kappa,
                           use_noise=False, sigma_phi=1e-3, u_max=None,
                           Q=None, P0=None, xhat0=None):
    """
    Nonlinear closed-loop simulation with EKF feedback:
        u = -K * (xhat - x_ref)
    Measurement z = [phi, phidot]
    If use_noise=True, y_phi noisy and y_phidot computed by differentiating y_phi (correlated).
    """
    N = int(T / dt)
    t = np.arange(N) * dt

    x = np.asarray(x0, dtype=float).copy()

    # Defaults
    if Q is None:
        Q = np.diag([1e-7, 1e-5, 1e-5, 1e-3])
    if P0 is None:
        P0 = np.diag([1e-4, 1e-2, 1e-3, 1e-2])
    if xhat0 is None:
        # start estimate at reference (common debugging choice)
        xhat0 = np.asarray(x_ref, dtype=float).copy()

    R = R_from_sigma_phi(sigma_phi, dt) if use_noise else np.zeros((2, 2), dtype=float)
    ekf = EKF(xhat0, P0, Q, R, dt, p, kappa=float(kappa), G_shape="const")

    X = np.zeros((N, 4), dtype=float)
    XH = np.zeros((N, 4), dtype=float)
    U = np.zeros(N, dtype=float)
    Etheta = np.zeros(N, dtype=float)

    rng = np.random.default_rng(0)
    yphi_prev = x[0]  # for numerical differentiation when noise is used

    for k in range(N):
        X[k] = x
        XH[k] = ekf.x
        Etheta[k] = wrap_pi(x[1] - ekf.x[1])

        dxhat = state_error(ekf.x, x_ref)
        u_min = K @ dxhat
        u = -float(u_min[0])
        if u_max is not None:
            u = float(np.clip(u, -u_max, u_max))
        U[k] = u

        # propagate plant
        x = rk4_step(rhs_continuous, x, u, dt, p,
                     kappa=float(kappa), G_shape="const", b_theta_true=p.b_theta_nom)

        # measurement
        if use_noise:
            yphi = x[0] + rng.normal(0.0, sigma_phi)
            yphidot = (yphi - yphi_prev) / dt
            yphi_prev = yphi
            z = np.array([yphi, yphidot], dtype=float)
        else:
            z = np.array([x[0], x[2]], dtype=float)

        ekf.predict(u)
        ekf.update(z)

        if (not np.all(np.isfinite(x))) or (not np.all(np.isfinite(ekf.x))):
            print("[Nonlinear EKF CL] Non-finite state encountered; stopping.")
            X = X[:k+1]
            XH = XH[:k+1]
            U = U[:k+1]
            Etheta = Etheta[:k+1]
            t = t[:k+1]
            break

    return t, X, XH, U, Etheta


# -----------------------------
# Plotting utilities
# -----------------------------

def plot_three_panel(title, t_lin, DX, U_lin, t_nl, X_nl, U_nl, t_ekf, X_true, X_hat, U_ekf, Etheta, theta_ref):
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Panel 1: theta error (linear vs nonlinear)
    axs[0].plot(t_nl, wrap_pi(X_nl[:, 1] - theta_ref), label="theta error (nonlinear, true-state CL)")
    axs[0].plot(t_nl, wrap_pi(X_nl[:, 0]), label="phi error (nonlinear, true-state CL)")
    axs[0].plot(t_lin, DX[:, 1], "--", label="theta error (linear CL, dx)")
    axs[0].plot(t_lin, DX[:, 0], "--", label="phi error (linear CL, dx)")
    axs[0].set_ylabel("theta - theta_ref [rad]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Panel 2: control input
    axs[1].plot(t_nl, U_nl, label="u(t) nonlinear true-state CL")
    axs[1].plot(t_lin, U_lin, "--", label="u(t) linear CL")
    axs[1].plot(t_ekf, U_ekf, ":", label="u(t) nonlinear EKF CL")
    axs[1].set_ylabel("u [N·m]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # Panel 3: EKF theta estimation error
    axs[2].plot(t_ekf, Etheta, label="theta error (true - hat)")
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("theta error [rad]")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.show()

    # Additional overlay: theta true vs theta hat
    plt.figure(figsize=(11, 4.5))
    plt.plot(t_ekf, X_true[:, 1], label="theta true")
    plt.plot(t_ekf, X_hat[:, 1], label="theta hat (EKF)")
    plt.axhline(theta_ref, color="k", linestyle="--", linewidth=1.0, label="theta_ref")
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.title(title + " — theta trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main diagnostic driver
# -----------------------------

def debug_lqr_chain(kappas=(0.0, 0.05, 0.10, 0.20), dt=0.01, T=5.0):
    p = FurutaParams()
    cfg = ExperimentConfig(dt=dt, sigma_phi=1e-3)

    # Down equilibrium (your convention): theta_ref = pi
    x_ref = cfg.x_ref
    u_ref = 0.0

    # Small deviation from down
    x0 = cfg.x0_true
    dx0 = state_error(x0, x_ref)

    # LQR weights (tune here; key is NOT to over-penalize phi)
    Q_lqr = np.diag([1.0, 2.0, 5.0, 5.0])
    R_lqr = np.array([[100.0]])

    print("\n===== DEBUG LQR CHAIN =====")
    print(f"dt={dt}, T={T}")
    print(f"x_ref={x_ref}, x0={x0}, dx0={dx0}")
    print(f"Q_lqr={np.diag(Q_lqr)}, R_lqr={R_lqr.flatten()}")

    
    t_ekf_list = []
    E_list = []
    for i, kappa in enumerate(kappas):
        kappa = float(kappa)

        # Linearize around x_ref
        A, B = linearize_rhs(p, x_ref, u_ref, kappa=kappa, G_shape="const", eps=1e-6)

        # Diagnostics
        eigA = np.linalg.eigvals(A)
        rankC = controllability_rank(A, B)
        if i == 0:
            K = lqr(A, B, Q_lqr, R_lqr)
        eigAcl = np.linalg.eigvals(A - B @ K)

        print("\n----------------------------------")
        print(f"kappa = {kappa:+.3f}")
        print(f"controllability rank = {rankC}/4")
        print(f"eig(A)     = {np.round(eigA, 5)}")
        print(f"eig(A-BK)  = {np.round(eigAcl, 5)}")
        print(f"K          = {np.round(K, 5)}")

        # Linear CL sim (dx)
        t_lin, DX, U_lin = simulate_linear_cl(A, B, K, dx0, dt, T)

        # Nonlinear CL sim with TRUE state feedback (no EKF)
        t_nl, X_nl, U_nl = simulate_nonlinear_true_state(p, K, x_ref, x0, dt, T, kappa, u_max=None)

        # Nonlinear CL sim with EKF, perfect measurement
        t_ekf, X_true, X_hat, U_ekf, Etheta = simulate_nonlinear_ekf(
            p, K, x_ref, x0, dt, T, kappa,
            use_noise=False, sigma_phi=1e-3, u_max=None,
            Q=np.diag([1e-7, 1e-5, 1e-5, 1e-3]),
            P0=np.diag([1e-4, 1e-2, 1e-3, 1e-2]),
            xhat0=x_ref.copy()
        )

        plot_three_panel(
            title=f"LQR diagnostics (no saturation, no uncertainty) — kappa={kappa:+.3f}",
            t_lin=t_lin, DX=DX, U_lin=U_lin,
            t_nl=t_nl, X_nl=X_nl, U_nl=U_nl,
            t_ekf=t_ekf, X_true=X_true, X_hat=X_hat, U_ekf=U_ekf, Etheta=Etheta,
            theta_ref=np.pi
        )

        t_ekf_list.append(t_ekf)
        E_list.append(Etheta)

    plt.figure()
    for k, t, E in zip(kappas, t_ekf_list, E_list):
        plt.plot(t, E, label=f"kappa +{k}: theta_error EKF")
    plt.xlabel('t [s]')
    plt.ylabel('theta - theta_hat [rad]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

        

if __name__ == "__main__":
    debug_lqr_chain(kappas=(0.0, 0.05, 0.10, 0.20), dt=0.01, T=5.0)
