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


def _amp_envelope(t, tau, freeze_after):
    """Amplitude envelope e^{-t/tau} until freeze_after, then constant."""
    if t < freeze_after:
        a = np.exp(-t / tau)
        da = -np.exp(-t / tau) / tau
    else:
        a = np.exp(-freeze_after / tau)
        da = 0.0
    return a, da


def _omega_phi_and_phase(t, omega_base, omega_hi_mult, omega_lo_mult, tau_omega, phase0=0.0, mode="linear"):
    """
    Returns instantaneous omega_phi(t) [rad/s] and phase phi(t) = ∫ omega_phi dt + phase0.
    Schedules omega_phi from omega_hi_mult*omega_base -> omega_lo_mult*omega_base over tau_omega seconds,
    then holds at omega_lo_mult*omega_base.
    """
    w_hi = omega_hi_mult * omega_base
    w_lo = omega_lo_mult * omega_base

    if tau_omega <= 0.0:
        # no decay, constant
        return w_lo, phase0 + w_lo * t

    if mode == "linear":
        if t < tau_omega:
            w_t = w_hi + (w_lo - w_hi) * (t / tau_omega)
            # phase = ∫(w_hi + (w_lo-w_hi)t/tau) dt = w_hi t + 0.5(w_lo-w_hi)t^2/tau
            phase = phase0 + w_hi * t + 0.5 * (w_lo - w_hi) * (t**2) / tau_omega
        else:
            w_t = w_lo
            # phase at tau + w_lo*(t-tau)
            phase_tau = phase0 + w_hi * tau_omega + 0.5 * (w_lo - w_hi) * tau_omega
            phase = phase_tau + w_lo * (t - tau_omega)

        return w_t, phase

    elif mode == "exp_hit_tau":
        # Smooth exponential-like schedule that hits w_lo exactly at t=tau_omega, then holds.

        if t < tau_omega:
            # normalized exponential factor in [0,1]
            a = (np.exp(-t / tau_omega) - np.exp(-1.0)) / (1.0 - np.exp(-1.0))
            w_t = w_lo + (w_hi - w_lo) * a

            # phase integral for this normalized schedule:
            # w(t)=w_lo+(w_hi-w_lo)*a(t)
            # a(t)= (e^{-t/tau}-e^{-1})/(1-e^{-1})
            # ∫ a(t) dt = [ -tau e^{-t/tau} - e^{-1} t ] / (1-e^{-1}) + const
            denom = (1.0 - np.exp(-1.0))
            phase = phase0 + w_lo * t + (w_hi - w_lo) * (
                (-tau_omega * np.exp(-t / tau_omega) - np.exp(-1.0) * t + tau_omega) / denom
            )
        else:
            w_t = w_lo
            # compute phase at tau_omega using t=tau_omega in the formula above
            denom = (1.0 - np.exp(-1.0))
            phase_tau = phase0 + w_lo * tau_omega + (w_hi - w_lo) * (
                (-tau_omega * np.exp(-1.0) - np.exp(-1.0) * tau_omega + tau_omega) / denom
            )
            phase = phase_tau + w_lo * (t - tau_omega)

        return w_t, phase

    else:
        raise ValueError("mode must be 'linear' or 'exp_hit_tau'")



def phi_ref_fun(t, omega, phi_cfg):
    if not phi_cfg["enabled"]:
        return 0.0
    
    typ = phi_cfg.get("type", "sin_with_decay")
    amp0 = float(phi_cfg.get("amp0", 0.0))
    scale_amp = float(phi_cfg.get("scale", 1.0))
    phase = float(phi_cfg.get("phase", 0.0))

    tau = float(phi_cfg.get("tau", 10.0))
    freeze_after = float(phi_cfg.get("freeze_after", 0.0))

    a, _ = _amp_envelope(t, tau, freeze_after)

    if typ == "sin_with_decay_freq_decay":
        omega_hi_mult = float(phi_cfg.get("omega_hi_mult", 1.0))
        omega_lo_mult = float(phi_cfg.get("omega_lo_mult", 1.0))
        tau_omega = float(phi_cfg.get("tau_omega", 1.0))
        mode = phi_cfg.get("freq_decay_mode", "linear")  # "linear" or "exp_hit_tau"

        w_t, phase = _omega_phi_and_phase(t, omega, omega_hi_mult, omega_lo_mult, tau_omega, phase0=phase, mode=mode)
        return scale_amp * amp0 * a * np.sin(phase)

    elif typ == "sin_with_decay":
        omega_mult = float(phi_cfg.get("omega_mult", 1.0))
        w = omega_mult * omega
        return scale_amp * amp0 * a * np.sin(w * t + phase)

    else:
        raise ValueError(f"[phi_ref_fun] The type {typ} does not exist.")



def phidot_ref_fun(t, omega, phi_cfg):
    if not phi_cfg["enabled"]:
        return 0.0
    
    typ = phi_cfg.get("type", "sin_with_decay")

    amp0 = float(phi_cfg.get("amp0", 0.0))
    scale_amp = float(phi_cfg.get("scale", 1.0))
    phase = float(phi_cfg.get("phase", 0.0))

    tau = float(phi_cfg.get("tau", 10.0))
    freeze_after = float(phi_cfg.get("freeze_after", 0.0))

    a, da = _amp_envelope(t, tau, freeze_after)

    if typ == "sin_with_decay_freq_decay":
        omega_hi_mult = float(phi_cfg.get("omega_hi_mult", 1.0))
        omega_lo_mult = float(phi_cfg.get("omega_lo_mult", 1.0))
        tau_omega = float(phi_cfg.get("tau_omega", 1.0))
        mode = phi_cfg.get("freq_decay_mode", "linear")

        # instantaneous omega_phi(t) and phase integral
        w_t, phase = _omega_phi_and_phase(
            t, omega,
            omega_hi_mult=omega_hi_mult,
            omega_lo_mult=omega_lo_mult,
            tau_omega=tau_omega,
            phase0=phase,
            mode=mode
        )

        # phi_ref = A a(t) sin(phase(t))
        # phidot_ref = A [ da sin(phase) + a cos(phase) * d(phase)/dt ]
        # and d(phase)/dt = w_t by construction
        return scale_amp * amp0 * (da * np.sin(phase) + a * np.cos(phase) * w_t)
    elif typ == "sin_with_decay":
        omega_mult = float(phi_cfg.get("omega_mult", 1.0))
        w = omega_mult * omega
        # d/dt [A a(t) sin(w t + phase)] = A[ da sin(.) + a w cos(.) ]
        return scale_amp * amp0 * (da * np.sin(w * t + phase) + a * w * np.cos(w * t + phase))

    else:
        raise ValueError(f"[phidot_ref_fun] The type {typ} does not exist.")


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


def smoothstep(z):
    """C^1 smooth step from 0 to 1 for z in [0,1]."""
    z = float(np.clip(z, 0.0, 1.0))
    return z*z*(3.0 - 2.0*z)

def singular_weight(theta, b_theta, sing_cfg):
    """
    Continuous weight w in [0,1] indicating 'how singular' we are.
    w≈1 deep in singular region, w≈0 far away.
    Combines angle proximity to pi/2 and small b_theta.
    """
    if not sing_cfg["enabled"]:
        return 0.0

    deg_zone = float(sing_cfg.get("deg_zone", 10.0))
    bth_min = float(sing_cfg.get("bth_min", 0.02))

    # angle proximity: eta close to pi/2
    eta = abs(theta_dev(theta))
    dist = abs(eta - np.pi/2)
    band = np.deg2rad(deg_zone)

    # map dist=0 -> 1, dist>=band -> 0
    w_ang = 1.0 - smoothstep(dist / max(1e-12, band))

    # b_theta proximity: small b_theta -> 1, big b_theta -> 0
    # use a soft threshold around bth_min
    babs = abs(b_theta)
    w_b = 1.0 - smoothstep(babs / max(1e-12, bth_min))

    # combine (OR-like): if either is singular, weight increases
    w = max(w_ang, w_b)

    return float(np.clip(w, 0.0, 1.0))


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
    w = singular_weight(theta, b_theta, sing_cfg)

    freeze_ff = bool(sing_cfg.get("freeze_ff", True))
    if sing and freeze_ff and np.isfinite(thdd_ff_prev):
        thdd_ff = (1.0 - w) * thdd_r + w * thdd_ff_prev
    else:
        thdd_ff = thdd_r

    K_smc = float(ctrl["K_smc"])
    K_smc_sing = float(sing_cfg.get("K_smc_sing", K_smc))
    K_here = (1.0 - w) * K_smc + w * K_smc_sing

    phi_s = float(ctrl["phi_s"])
    
    # --- singular-safe desired acceleration component ---
    # In the singular region, bias toward damping velocity error rather than position error:
    kd_sing = float(sing_cfg.get("kd_sing", 2.0))
    thdd_des_sing = -kd_sing * (thetadot - thd_r)

    thdd_des_nom = thdd_ff - K_here * sat(s / phi_s)
    thdd_des = (1.0 - w) * thdd_des_nom + w * thdd_des_sing

    eps_inv = float(ctrl["eps_inv"])
    u_track = (b_theta / (b_theta*b_theta + eps_inv)) * (thdd_des - f_theta)

    Kphi_p = float(ctrl["Kphi_p"])
    Kphi_d = float(ctrl["Kphi_d"])
    u_phi = -Kphi_p * (wrap_center(phi, 0.0) - ph_r) - Kphi_d * (phidot - phd_r)

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
