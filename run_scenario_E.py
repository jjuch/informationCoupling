import numpy as np
import matplotlib.pyplot as plt

from config import FurutaParams
from furuta_model import furuta_M_C_g, G_kappa, rhs_continuous, wrap_center


# ============================================================
# Scenario E: Feedforward + SMC + hybrid singular-zone control
# ============================================================

# -------------------------
# General settings
# -------------------------
KAPPA = 0.0
G_SHAPE = "const"

DT = 0.005
N_SUB = 5
T_TOTAL = 30.0
T_BURN = 5.0

# Desired theta orbit
AMP = 2.0 * np.pi / 3.0      # must cross ±pi/2
PHASE_REF = 0.0
PHI_REF = 0.0

# -------------------------
# Control tuning
# -------------------------

# Sliding surface tuning
LAMBDA_MULT = 1.0    # lambda = LAMBDA_MULT * omega
K_SMC = 65.0          # SMC gain
PHI_S = 0.15         # boundary layer thickness

# Regularized inverse
EPS_INV = 2e-3

# Phi stabilizer
KPHI_P = 0.5
KPHI_D = 0.50

USE_SAT = True

# -------------------------
# Singular zone definition
# -------------------------
BTH_MIN = 0.02
DEG_ZONE = 10.0       # ±10° around pi/2
K_SMC_SING = 1.0      # reduced SMC in singular zone


# ============================================================
# Helpers
# ============================================================

def theta_dev(theta):
    return wrap_center(theta, np.pi) - np.pi

def sat(x):
    return float(np.clip(x, -1.0, 1.0))

def omega_small_angle(p):
    return float(np.sqrt(p.delta / p.beta))

def theta_ref(t, omega):
    return np.pi + AMP * np.sin(omega * t + PHASE_REF)

def thetadot_ref(t, omega):
    return AMP * omega * np.cos(omega * t + PHASE_REF)

def thetaddot_ref(t, omega):
    return -AMP * omega**2 * np.sin(omega * t + PHASE_REF)

def phi_ref(t, omega):
    if isinstance(t, float):
        amp = np.exp(-t/10.0) if t < 5.0 else np.exp(-5.0/10.0)
    else:
        amp = np.array([np.exp(-t_el/10.0) if t_el < 5.0 else np.exp(-5.0/10.0) for t_el in t]) 

    return 3*np.pi/2 * amp * np.sin(omega*t) + PHI_REF

def phidot_ref(t, omega):
    if isinstance(t, float):
        d_amp = -np.exp(-t/10.0)/10.0 if t < 5.0 else 0.0
        amp = np.exp(-t/10.0) if t < 5.0 else np.exp(-5.0/10.0)
    else:
        d_amp = np.array([-np.exp(-t_el/10.0)/10.0 if t_el < 5.0 else 0.0 for t_el in t])
        amp = np.array([np.exp(-t_el/10.0) if t_el < 5.0 else np.exp(-5.0/10.0) for t_el in t])
    return 3*np.pi/2 * (d_amp * np.sin(omega*t) + amp * omega * np.cos(omega*t))

    # return 0.0

def sat_u(u, umax):
    umax=30
    return float(np.clip(u, -umax, umax))


def theta_affine_terms(x, p):
    """
    theta_ddot = f_theta(x) + b_theta(x) * u
    """
    phi, theta, phidot, thetadot = x
    q = np.array([phi, theta])
    qd = np.array([phidot, thetadot])

    M, C, gvec = furuta_M_C_g(q, qd, p)
    G = G_kappa(q, KAPPA, M, p, shape=G_SHAPE)

    tau_f = np.array([p.b_phi * phidot, p.b_theta_nom * thetadot])

    qdd0 = np.linalg.solve(M, -(C + G) @ qd - gvec - tau_f)
    f_theta = float(qdd0[1])

    Minv = np.linalg.inv(M)
    b_theta = float(Minv[1, 0])

    return f_theta, b_theta


def in_singular_zone(theta, b_theta):
    eta = abs(theta_dev(theta))
    near_pi2 = abs(eta - np.pi/2) < np.deg2rad(DEG_ZONE)
    weak_input = abs(b_theta) < BTH_MIN
    return near_pi2 or weak_input


# ============================================================
# Control law
# ============================================================

def control_law(x, t, p, omega, lambda_, thdd_ff_prev):
    phi, theta, phidot, thetadot = x

    th_r = theta_ref(t, omega)
    thd_r = thetadot_ref(t, omega)
    thdd_r = thetaddot_ref(t, omega)

    e = wrap_center(theta, np.pi) - th_r
    ed = thetadot - thd_r
    s = ed + lambda_ * e

    f_theta, b_theta = theta_affine_terms(x, p)
    sing = in_singular_zone(theta, b_theta)

    # Feedforward theta_ddot
    if sing and np.isfinite(thdd_ff_prev):
        thdd_ff = thdd_ff_prev
    else:
        thdd_ff = thdd_r

    # Sliding mode correction
    K_here = K_SMC_SING if sing else K_SMC
    thdd_des = thdd_ff - K_here * sat(s / PHI_S)

    # Regularized inverse
    u_track = (b_theta / (b_theta*b_theta + EPS_INV)) * (thdd_des - f_theta)

    # Phi stabilizer
    u_phi = -KPHI_P * (phi - phi_ref(t, omega)) - KPHI_D * (phidot - phidot_ref(t, omega))

    u = u_track + u_phi
    if USE_SAT:
        u = sat_u(u, p.u_max)

    return u, b_theta, e, s, thdd_ff, sing


# ============================================================
# Closed-loop RK4 step
# ============================================================

def rk4_step_closed_loop(x, t, p, omega, lambda_, thdd_ff_prev):
    dt_int = DT / N_SUB
    xk = x.copy()

    u_last = 0.0
    b_last = 0.0
    e_last = 0.0
    s_last = 0.0
    sing_last = False
    thdd_ff_last = thdd_ff_prev

    for _ in range(N_SUB):
        u1, b1, e1, s1, thff1, sing1 = control_law(
            xk, t, p, omega, lambda_, thdd_ff_last
        )

        u_last, b_last, e_last, s_last, sing_last = u1, b1, e1, s1, sing1
        thdd_ff_last = thff1

        k1 = rhs_continuous(xk, u1, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

        u2, _, _, _, _, _ = control_law(xk + 0.5*dt_int*k1, t + 0.5*dt_int, p, omega, lambda_, thdd_ff_last)
        k2 = rhs_continuous(xk + 0.5*dt_int*k1, u2, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

        u3, _, _, _, _, _ = control_law(xk + 0.5*dt_int*k2, t + 0.5*dt_int, p, omega, lambda_, thdd_ff_last)
        k3 = rhs_continuous(xk + 0.5*dt_int*k2, u3, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

        u4, _, _, _, _, _ = control_law(xk + dt_int*k3, t + dt_int, p, omega, lambda_, thdd_ff_last)
        k4 = rhs_continuous(xk + dt_int*k3, u4, p, kappa=KAPPA, G_shape=G_SHAPE, b_theta_true=None)

        xk = xk + (dt_int/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt_int

    return xk, u_last, b_last, e_last, s_last, thdd_ff_last, sing_last


# ============================================================
# Main simulation + plots
# ============================================================

def main():
    p = FurutaParams()

    omega = omega_small_angle(p)
    lambda_ = LAMBDA_MULT * omega

    print("\n=== Scenario E: Feedforward + SMC + singular zone ===")
    print(f"omega = {omega:.4f} rad/s ({omega/(2*np.pi):.4f} Hz)")
    print(f"lambda = {lambda_:.4f}")

    x = np.array([0.0, np.pi + AMP, 0.0, 0.0])

    N = int(T_TOTAL / DT)
    t = np.arange(N) * DT

    X = np.zeros((N, 4))
    U = np.zeros(N)
    E = np.zeros(N)
    S = np.zeros(N)
    BTH = np.zeros(N)
    SING = np.zeros(N)

    thdd_ff_prev = np.nan

    for k in range(N):
        print(f"{k + 1} / {N}", end="\r")
        X[k] = x
        x, u, bth, e, s, thdd_ff_prev, sing = rk4_step_closed_loop(
            x, t[k], p, omega, lambda_, thdd_ff_prev
        )
        U[k] = u
        E[k] = e
        S[k] = s
        BTH[k] = bth
        SING[k] = 1.0 if sing else 0.0

    # References
    # phi_r = np.full_like(t, PHI_REF)
    phi_r = phi_ref(t, omega)
    # phid_r = np.zeros_like(t)
    phid_r = phidot_ref(t, omega)
    th_r = theta_ref(t, omega)
    thd_r = thetadot_ref(t, omega)

    # ========================================================
    # REQUIRED FIGURE: u + 4 states with references
    # ========================================================
    plt.figure(figsize=(11, 10))

    plt.subplot(5,1,1)
    plt.plot(t, U)
    plt.ylabel("u [Nm]")
    plt.grid(True)

    plt.subplot(5,1,2)
    plt.plot(t, X[:,0], label="phi")
    plt.plot(t, phi_r, "--", label="phi_ref")
    plt.ylabel("phi [rad]")
    plt.legend()
    plt.grid(True)

    plt.subplot(5,1,3)
    plt.plot(t, theta_dev(X[:,1]), label="theta-pi")
    plt.plot(t, theta_dev(th_r), "--", label="theta_ref-pi")
    plt.axhline(+AMP, ls=":")
    plt.axhline(-AMP, ls=":")
    plt.ylabel("theta-pi [rad]")
    plt.legend()
    plt.grid(True)

    plt.subplot(5,1,4)
    plt.plot(t, X[:,2], label="phidot")
    plt.plot(t, phid_r, "--", label="phidot_ref")
    plt.ylabel("phidot [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.subplot(5,1,5)
    plt.plot(t, X[:,3], label="thetadot")
    plt.plot(t, thd_r, "--", label="thetadot_ref")
    plt.xlabel("time [s]")
    plt.ylabel("thetadot [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Diagnostics: singular zone and b_theta
    # ========================================================
    plt.figure(figsize=(11, 5))
    plt.subplot(2,1,1)
    plt.plot(t, BTH)
    plt.axhline(+BTH_MIN, ls="--")
    plt.axhline(-BTH_MIN, ls="--")
    plt.ylabel("b_theta")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t, SING)
    plt.ylabel("singular zone")
    plt.xlabel("time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Diagnostics: error and sliding surface
    # ========================================================
    plt.figure(figsize=(11, 5))
    plt.subplot(2,1,1)
    plt.plot(t, E)
    plt.ylabel("e")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t, S)
    plt.ylabel("s")
    plt.xlabel("time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # FFT of theta-pi
    # ========================================================
    mask = t >= T_BURN
    eta = theta_dev(X[mask,1])
    eta -= np.mean(eta)

    fft = np.fft.rfft(eta)
    freqs = np.fft.rfftfreq(len(eta), d=DT)
    mag = np.abs(fft)

    f_forcing = omega / (2*np.pi)
    f_dom = freqs[np.argmax(mag[1:])+1]

    plt.figure(figsize=(10,4))
    plt.plot(freqs, mag)
    plt.axvline(f_forcing, ls="--", label="forcing")
    plt.axvline(f_dom, ls=":", label="dominant")
    plt.xlim(0, 6)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("|FFT|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nDone.")
    print(f"Forcing frequency = {f_forcing:.4f} Hz")
    print(f"Dominant frequency = {f_dom:.4f} Hz")


if __name__ == "__main__":
    main()
