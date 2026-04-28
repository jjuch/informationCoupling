import numpy as np
from config import FurutaParams
from furuta_model import furuta_M_C_g, G_kappa, rhs_continuous, rk4_step, wrap_center

def theta_dev(theta):
    return wrap_center(theta, np.pi) - np.pi

def sat(x):
    return float(np.clip(x, -1.0, 1.0))

def sat_u(u, umax):
    return float(np.clip(u, -umax, umax))

def omega_small_angle(p: FurutaParams):
    return float(np.sqrt(p.delta / p.beta))


def theta_ref(t, omega, amp, phase):
    return np.pi + amp * np.sin(omega * t + phase)

def thetadot_ref(t, omega, amp, phase):
    return amp * omega * np.cos(omega * t + phase)

def thetaddot_ref(t, omega, amp, phase):
    return -amp * (omega**2) * np.sin(omega * t + phase)

def phi_ref_fun(t, omega, phi_cfg):
    if not phi_cfg["enabled"]:
        return 0.0
    amp0 = float(phi_cfg.get("amp0", 0.0))
    scale_amp = float(phi_cfg.get("scale", 1.0))
    omega_mult = float(phi_cfg.get("omega_mult", 1.0))
    phase = float(phi_cfg.get("phase", 0.0))
    
    # optional frequency decay parameters
    omega_lo = float(phi_cfg.get("omega_lo", omega_mult * omega))
    omega_hi = float(phi_cfg.get("omega_hi", omega_mult * omega))
    tau_omega = float(phi_cfg.get("tau_omega", 1.0))   # seconds

    tau = float(phi_cfg.get("tau", 10.0))
    freeze_after = float(phi_cfg.get("freeze_after", 0.0))
    w = omega_mult * omega

    if t < freeze_after:
        a = np.exp(-t / tau)
    else:
        a = np.exp(-freeze_after / tau)

    if phi_cfg.get("type") == "sin_with_decay_freq_decay":
        # frequency schedule and phase integral
        w_lo = omega_lo
        w_hi = omega_hi
        varphi = phase + w_lo*t + (w_hi - w_lo)*tau_omega*(1.0 - np.exp(-t/tau_omega))
    elif phi_cfg.get("type") == "sin_with_decay":
        varphi = w * t + phase

    return scale_amp * amp0 * a * np.sin(varphi)


def phidot_ref_fun(t, omega, phi_cfg):
    if not phi_cfg["enabled"]:
        return 0.0
    amp0 = float(phi_cfg.get("amp0", 0.0))
    scale_amp = float(phi_cfg.get("scale", 1.0))
    # print("scale_amp: ", scale_amp)
    omega_mult = float(phi_cfg.get("omega_mult", 1.0))
    phase = float(phi_cfg.get("phase", 0.0))
    tau = float(phi_cfg.get("tau", 10.0))
    freeze_after = float(phi_cfg.get("freeze_after", 0.0))
    w = omega_mult * omega

    if t < freeze_after:
        a = np.exp(-t / tau)
        da = -np.exp(-t / tau) / tau
    else:
        a = np.exp(-freeze_after / tau)
        da = 0.0
    return scale_amp * amp0 * (da * np.sin(w*t + phase) + a * w * np.cos(w*t + phase))


def theta_affine_terms(x, p, kappa, G_shape):
    phi, theta, phidot, thetadot = x
    q = np.array([phi, theta])
    qd = np.array([phidot, thetadot])

    M, C, gvec = furuta_M_C_g(q, qd, p)
    G = G_kappa(q, kappa, M, p, shape=G_shape)
    tau_f = np.array([p.b_phi * phidot, p.b_theta_nom * thetadot])

    qdd0 = np.linalg.solve(M, -(C + G) @ qd - gvec - tau_f)
    f_theta = float(qdd0[1])

    Minv = np.linalg.inv(M)
    b_theta = float(Minv[1, 0])

    return f_theta, b_theta

def in_singular_zone(theta, b_theta, sing_cfg):
    if not sing_cfg["enabled"]:
        return False
    deg_zone = float(sing_cfg.get("deg_zone", 10.0))
    bth_min = float(sing_cfg.get("bth_min", 0.02))
    eta = abs(theta_dev(theta))
    near_pi2 = abs(eta - np.pi/2) < np.deg2rad(deg_zone)
    weak_input = abs(b_theta) < bth_min
    return bool(near_pi2 or weak_input)

def control_law(x, t, p, omega, cfg, thdd_ff_prev):
    ctrl = cfg["controller"]
    sing_cfg = cfg["singular_zone"]
    theta_cfg = cfg["theta_ref"]
    phi_cfg = cfg["phi_ref"]

    kappa = cfg["_case_key"]["kappa"]
    G_shape = cfg["_case_key"]["G_shape"]

    phi, theta, phidot, thetadot = x

    amp = float(theta_cfg["amp"])
    phase = float(theta_cfg["phase"])

    th_r   = theta_ref(t, omega, amp, phase)
    thd_r  = thetadot_ref(t, omega, amp, phase)
    thdd_r = thetaddot_ref(t, omega, amp, phase)

    ph_r  = phi_ref_fun(t, omega, phi_cfg)
    phd_r = phidot_ref_fun(t, omega, phi_cfg)

    e  = wrap_center(theta, np.pi) - th_r
    ed = thetadot - thd_r

    lam = float(ctrl["lambda_mult"]) * omega
    s = ed + lam * e

    f_theta, b_theta = theta_affine_terms(x, p, kappa, G_shape)
    sing = in_singular_zone(theta, b_theta, sing_cfg)

    freeze_ff = bool(sing_cfg.get("freeze_ff", True))
    if sing and freeze_ff and np.isfinite(thdd_ff_prev):
        thdd_ff = thdd_ff_prev
    else:
        thdd_ff = thdd_r

    K_smc = float(ctrl["K_smc"])
    K_smc_sing = float(sing_cfg.get("K_smc_sing", K_smc))
    K_here = K_smc_sing if sing else K_smc

    phi_s = float(ctrl["phi_s"])
    thdd_des = thdd_ff - K_here * sat(s / phi_s)

    eps_inv = float(ctrl["eps_inv"])
    u_track = (b_theta / (b_theta*b_theta + eps_inv)) * (thdd_des - f_theta)

    Kphi_p = float(ctrl["Kphi_p"])
    Kphi_d = float(ctrl["Kphi_d"])
    u_phi = -Kphi_p * (phi - ph_r) - Kphi_d * (phidot - phd_r)

    u = u_track + u_phi

    if bool(ctrl.get("use_sat", True)):
        umax = float(ctrl.get("u_max_override", p.u_max))
        u = sat_u(u, umax)

    return float(u), float(b_theta), float(e), float(s), float(thdd_ff), float(ph_r), float(phd_r), (1.0 if sing else 0.0)

def rk4_step_closed_loop(x, t0, dt, n_sub, p, omega, cfg, thdd_ff_prev):
    """
    Closed-loop RK4 step with substepping.
    Key property: the control input u is recomputed at:
      - the start of each RK4 stage (k1..k4)
      - each substep (dt_int = dt/n_sub)

    Returns:
      x_next,
      u_last, b_theta_last, e_last, s_last,
      thdd_ff_last, phi_ref_last, phidot_ref_last,
      sing_last (0/1)
    """
    dt_int = dt / float(n_sub)
    xk = np.asarray(x, dtype=float).copy()

    # diagnostics (returned)
    u_last = 0.0
    b_last = np.nan
    e_last = np.nan
    s_last = np.nan
    sing_last = 0.0
    thdd_ff_last = thdd_ff_prev
    ph_r_last = 0.0
    phd_r_last = 0.0

    kappa = float(cfg["_case_key"]["kappa"])
    G_shape = cfg["_case_key"]["G_shape"]

    for _ in range(int(n_sub)):

        # Stage 1
        u1, b1, e1, s1, thff1, ph_r1, phd_r1, sing1 = control_law(xk, t0, p, omega, cfg, thdd_ff_last)
        thdd_ff_last = thff1

        k1 = rhs_continuous(
            xk, u1, p,
            kappa=kappa, G_shape=G_shape, b_theta_true=None
        )

        # Stage 2
        x2 = xk + 0.5 * dt_int * k1
        u2, _, _, _, _, _, _, _ = control_law(x2, t0 + 0.5*dt_int, p, omega, cfg, thdd_ff_last)

        k2 = rhs_continuous(
            x2, u2, p,
            kappa=kappa, G_shape=G_shape, b_theta_true=None
        )

        # Stage 3
        x3 = xk + 0.5 * dt_int * k2
        u3, _, _, _, _, _, _, _ = control_law(x3, t0 + 0.5*dt_int, p, omega, cfg, thdd_ff_last)

        k3 = rhs_continuous(
            x3, u3, p,
            kappa=kappa, G_shape=G_shape, b_theta_true=None
        )

        # Stage 4
        x4 = xk + dt_int * k3
        u4, _, _, _, _, _, _, _ = control_law(x4, t0 + dt_int, p, omega, cfg, thdd_ff_last)

        k4 = rhs_continuous(
            x4, u4, p,
            kappa=kappa, G_shape=G_shape, b_theta_true=None
        )

        # RK4 combine
        xk = xk + (dt_int / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # store "last" diagnostics from Stage 1 evaluation (consistent)
        u_last = float(u1)
        b_last = float(b1)
        e_last = float(e1)
        s_last = float(s1)
        sing_last = float(sing1)
        ph_r_last = float(ph_r1)
        phd_r_last = float(phd_r1)

        t0 += dt_int

    return xk, u_last, b_last, e_last, s_last, thdd_ff_last, ph_r_last, phd_r_last, sing_last


def simulate_closed_loop(cfg):
    p = FurutaParams()
    dt = float(cfg["dt"])
    n_sub = int(cfg["n_sub"])
    T_total = float(cfg["T_total"])
    N = int(T_total / dt)
    t = np.arange(N) * dt

    theta_cfg = cfg["theta_ref"]
    if theta_cfg["omega_mode"] == "small_angle":
        omega = omega_small_angle(p)
    else:
        omega = float(theta_cfg["omega_mode"])

    amp = float(theta_cfg["amp"])
    x = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)

    X = np.zeros((N,4), float)
    U = np.zeros(N, float)
    BTH = np.zeros(N, float)
    E = np.zeros(N, float)
    S = np.zeros(N, float)
    SING = np.zeros(N, float)
    PHREF = np.zeros(N, float)
    PHDREF = np.zeros(N, float)

    thdd_ff_prev = np.nan

    dt_int = dt / float(n_sub)

    for k in range(N):
        print(f"{k + 1} / {N}", end="\r")
        X[k] = x

        x, u, bth, e, s, thdd_ff_prev, ph_r, phd_r, sing = rk4_step_closed_loop(
            x, t[k], dt, n_sub, p, omega, cfg, thdd_ff_prev
        )


        U[k] = u
        BTH[k] = bth
        E[k] = e
        S[k] = s
        SING[k] = sing
        PHREF[k] = ph_r
        PHDREF[k] = phd_r

    base = {
        "t": t,
        "X": X,
        "U": U,
        "BTH": BTH,
        "E": E,
        "S": S,
        "SING": SING,
        "PHREF": PHREF,
        "PHDREF": PHDREF,
        "omega": np.array([omega], float)
    }
    return base, cfg
