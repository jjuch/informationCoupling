import numpy as np
from config import FurutaParams
from furuta_model import furuta_M_C_g, G_kappa, rhs_continuous, rk4_step, wrap_center, b_theta_true

def theta_dev(theta):
    return wrap_center(theta, np.pi) - np.pi

def sat(x):
    return float(np.clip(x, -1.0, 1.0))

def sat_u(u, umax):
    return float(np.clip(u, -umax, umax))

def omega_small_angle(p: FurutaParams):
    return float(np.sqrt(p.delta / p.beta))

def theta_phase_estimate(theta, thetadot, amp, omega):
    eta = wrap_center(theta, np.pi) - np.pi
    return np.arctan2(eta / max(1e-9, amp), thetadot / max(1e-9, amp*omega))

def theta_ref(t, omega, amp, phase):
    return np.pi + amp * np.sin(omega * t + phase)

def thetadot_ref(t, omega, amp, phase):
    return amp * omega * np.cos(omega * t + phase)

def thetaddot_ref(t, omega, amp, phase):
    return -amp * (omega**2) * np.sin(omega * t + phase)

def theta_energy(eta, thetadot, beta, delta):
    # energy around theta = pi
    return 0.5 * beta * thetadot**2 + delta * (1.0 - np.cos(eta))


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


def rollout_dynamics(x0, t0, u_seq, dt, n_sub, p, kappa, G_shape):
    """
    Roll out dynamics forward using constant u per step.
    Uses RK4 substepping for stability but with low n_sub for speed.
    Returns X_pred shape (N+1,4).
    """
    x = np.asarray(x0, float).copy()
    X = np.zeros((len(u_seq)+1, 4), float)
    X[0] = x
    dt_int = dt / float(n_sub)
    b0_true, b1_true = p.b0_nom, p.b1_nom

    for k, u in enumerate(u_seq):
        # integrate one MPC step with n_sub plant substeps
        for _ in range(int(n_sub)):
            x = rk4_step(rhs_continuous, x, float(u), dt_int, p,
                         kappa=float(kappa), G_shape=G_shape, b_theta_true=lambda theta: b_theta_true(theta, b0_true, b1_true))
        X[k+1] = x
        t0 += dt
    return X


def cost_theta_tracking(X_pred, t_grid, theta_ref_fun, thetadot_ref_fun,
                        u_seq, u_prev,
                        w_theta=100.0, w_thdot=20.0, w_u=0.05, w_du=5.0,
                        w_phi=0.0, w_phdot=0.0, phi_ref_fun=None, phidot_ref_fun=None):
    """
    Stage cost:
      w_theta*(theta-theta_ref)^2 + w_thdot*(thetadot-thetadot_ref)^2
      + w_u*u^2 + w_du*(du)^2
    Optionally add mild phi penalties.
    """
    J = 0.0
    # X_pred has N+1 states; u_seq length N
    for k in range(len(u_seq)):
        xk = X_pred[k]
        tk = t_grid[k]
        phi, theta, phidot, thetadot = xk

        th_r = theta_ref_fun(tk)
        thd_r = thetadot_ref_fun(tk)

        e_th = (theta - th_r)
        e_thd = (thetadot - thd_r)

        J += w_theta * (e_th*e_th) + w_thdot * (e_thd*e_thd)

        if w_phi > 0.0 and phi_ref_fun is not None:
            e_phi = (phi - phi_ref_fun(tk))
            J += w_phi * (e_phi*e_phi)

        if w_phdot > 0.0 and phidot_ref_fun is not None:
            e_phd = (phidot - phidot_ref_fun(tk))
            J += w_phdot * (e_phd*e_phd)

        uk = float(u_seq[k])
        J += w_u * (uk*uk)

        du = uk - (float(u_prev) if k == 0 else float(u_seq[k-1]))
        J += w_du * (du*du)

    return float(J)

def terminal_cost_peak(XN, u_seq, u_prev,
                       eta_target, thdot_target,
                       w_eta=500.0, w_thdot=200.0,
                       w_u=0.02, w_du=1.0):
    """
    Terminal-only cost + small regularization on u and du.
    """
    if not np.all(np.isfinite(XN)):
        return np.inf

    thetaN = wrap_center(float(XN[1]), 0.0)
    thdotN = wrap_center(float(XN[3]), 0.0)
    etaN = float(theta_dev(thetaN))

    J = w_eta * (etaN - eta_target)**2 + w_thdot * (thdotN - thdot_target)**2

    # small regularization to avoid wild plans
    for k, uk in enumerate(u_seq):
        uk = float(uk)
        J += w_u * uk*uk
        duk = uk - (float(u_prev) if k == 0 else float(u_seq[k-1]))
        J += w_du * duk*duk

    return float(J)


def cem_mpc(x0, t0, p, kappa, G_shape,
            theta_ref_fun, thetadot_ref_fun,
            u_prev,
            u_max=15.0,
            N=15, dt=0.02, n_sub=2,
            iters=4, pop=250, elite_frac=0.15,
            init_sigma=5.0,
            w_theta=120.0, w_thdot=30.0, w_u=0.05, w_du=8.0,
            warm_start=None,
            rng=None):
    """
    Cross-Entropy Method MPC:
      - sample u sequences
      - evaluate cost
      - update mean/sigma using elite samples
    Returns:
      u0 (first control), best_seq, info dict
    """
    if rng is None:
        rng = np.random.default_rng(0)

    elite_n = max(5, int(elite_frac * pop))

    # Initialize mean sequence
    if warm_start is not None and len(warm_start) >= N:
        mu = np.asarray(warm_start[:N], float).copy()
    else:
        mu = np.zeros(N, float)

    sigma = np.ones(N, float) * float(init_sigma)

    best_J = np.inf
    best_seq = None

    # time grid for references
    t_grid = t0 + np.arange(N) * dt

    for _ in range(int(iters)):
        # sample sequences
        U = rng.normal(loc=mu, scale=sigma, size=(pop, N))
        U = np.clip(U, -u_max, u_max)

        costs = np.zeros(pop, float)
        for i in range(pop):
            X_pred = rollout_dynamics(x0, t0, U[i], dt, n_sub, p, kappa, G_shape)
            costs[i] = cost_theta_tracking(
                X_pred, t_grid,
                theta_ref_fun, thetadot_ref_fun,
                U[i], u_prev,
                w_theta=w_theta, w_thdot=w_thdot, w_u=w_u, w_du=w_du
            )

        # update best
        jmin = int(np.argmin(costs))

        if costs[jmin] < best_J:
            best_J = float(costs[jmin])
            best_seq = U[jmin].copy()

        # elite set
        elite_idx = np.argsort(costs)[:elite_n]
        elite = U[elite_idx]

        # update distribution
        mu = np.mean(elite, axis=0)
        sigma = np.std(elite, axis=0) + 1e-6  # avoid collapse
        
    u0 = float(best_seq[0]) if best_seq is not None else 0.0
    info = {"best_J": best_J, "mu0": float(mu[0]), "sigma0": float(sigma[0])}
    return u0, best_seq, info

def compute_diagnostics(x, t, p, omega, cfg, sing_mem):
    """
    Compute diagnostic signals at (x, t) regardless of whether MPC or SMC produced u.
    Returns:
        b_theta, sing_flag, ph_r, phd_r, th_r_used, thd_r_used, e, s
    """
    ctrl = cfg["controller"]
    theta_cfg = cfg["theta_ref"]
    phi_cfg = cfg["phi_ref"]
    sing_cfg = cfg["singular_zone"]

    kappa = cfg["_case_key"]["kappa"]
    G_shape = cfg["_case_key"]["G_shape"]

    phi, theta, phidot, thetadot = x

    # references for phi always available
    ph_r = phi_ref_fun(t, omega, phi_cfg)
    phd_r = phidot_ref_fun(t, omega, phi_cfg)

    # compute input effectiveness and singular flag
    f_theta, b_theta = theta_affine_terms(x, p, kappa, G_shape)
    sing_flag = in_singular_zone(theta, b_theta, sing_cfg)

    # piecewise theta reference (sin outside, linear inside)
    th_r, thd_r, _ = theta_refs_piecewise(t, omega, theta_cfg, sing_flag, sing_mem)
    th_r = th_r(t)
    thd_r = thd_r(t)

    # errors and sliding surface
    e = wrap_center(theta, np.pi) - th_r
    ed = thetadot - thd_r
    lam = float(ctrl["lambda_mult"]) * omega
    s = ed + lam * e

    return float(b_theta), float(sing_flag), float(ph_r), float(phd_r), float(th_r), float(thd_r), float(e), float(s)

def cem_solve_zone_mpc(
    x0, t0, p, kappa, G_shape,
    theta_exit, thetadot_exit,
    u_prev, u_max,
    dt=0.02, N=10, n_sub=1,
    iters=2, pop=100, elite_frac=0.2, init_sigma=6.0,
    w_exit_theta=200.0,
    w_exit_thdot=80.0,
    w_exit_energy=5.0,
    w_u=0.02,
    w_du=2.0,
    rng=None,
    warm_start=None
):

    if rng is None:
        rng = np.random.default_rng(0)

    elite_n = max(5, int(elite_frac * pop))

    if warm_start is not None and len(warm_start) >= N:
        mu = np.asarray(warm_start[:N], float)
    else:
        mu = np.zeros(N)

    sigma = np.ones(N) * init_sigma

    E_target = 1.5 * float(p.delta)  # corresponds to amplitude 2*pi/3

    best_J = np.inf
    best_u = None

    for _ in range(iters):
        U = rng.normal(mu, sigma, size=(pop, N))
        U = np.clip(U, -u_max, u_max)

        costs = np.zeros(pop)
        for i in range(pop):
            X = rollout_dynamics(x0, t0, U[i], dt, n_sub, p, kappa, G_shape)
            _, thetaN, _, thetadotN = X[-1]
            etaN = theta_dev(thetaN)
            EN = theta_energy(etaN, thetadotN, p.beta, p.delta)

            J = (
                w_exit_theta * (thetaN - theta_exit)**2 +
                w_exit_thdot * (thetadotN - thetadot_exit)**2 +
                w_exit_energy * (EN - E_target)**2
            )

            for k, uk in enumerate(U[i]):
                J += w_u * uk**2
                duk = uk - (u_prev if k == 0 else U[i][k-1])
                J += w_du * duk**2

            costs[i] = J

        j = np.argmin(costs)
        if costs[j] < best_J:
            best_J = costs[j]
            best_u = U[j].copy()


        elite = U[np.argsort(costs)[:elite_n]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-6

        # guaranteed fallback
        if best_u is None:
            best_u = np.clip(mu, -u_max, u_max)

    return best_u, best_J

def cem_solve_peak_mpc(x0,t0, p, kappa, G_shape,
                       eta_target, thdot_target,
                       u_prev, u_max,
                       dt=0.02, N=10, n_sub=1,
                       iters=1, pop=120, elite_frac=0.2, init_sigma=6.0,
                       w_eta=500.0, w_thdot=200.0, w_u=0.02, w_du=1.0,
                       rng=None, warm_start=None):
    """
    Solve for u_seq that drives (eta, thetadot) to (eta_target, 0) at horizon end.
    - If iters=1: random shooting
    - If iters>1: CEM refinement
    """
    if rng is None:
        rng = np.random.default_rng(0)

    elite_n = max(5, int(elite_frac * pop))

    if warm_start is not None and len(warm_start) >= N:
        mu = np.asarray(warm_start[:N], float).copy()
    else:
        mu = np.zeros(N, float)

    sigma = np.ones(N, float) * float(init_sigma)

    best_J = np.inf
    best_u = None
    best_xpred = None

    for _ in range(int(iters)):
        U = rng.normal(loc=mu, scale=sigma, size=(pop, N))
        U = np.clip(U, -u_max, u_max)

        # inject safe candidate at index 0
        U[0, :] = np.clip(mu, -u_max, u_max)

        costs = np.full(pop, np.inf, float)
        Xpred = np.zeros((pop, 4))

        for i in range(pop):
            Xpred_temp = rollout_dynamics(x0, t0, U[i], dt, n_sub, p, kappa, G_shape)
            Xpred[i] = Xpred_temp[-1]
            costs[i] = terminal_cost_peak(
                Xpred[i], U[i], u_prev,
                eta_target, thdot_target,
                w_eta=w_eta, w_thdot=w_thdot,
                w_u=w_u, w_du=w_du
            )

        j = int(np.argmin(costs))
        if costs[j] < best_J:
            best_xpred = Xpred[j]
            best_xpred[1] = wrap_center(best_xpred[1], np.pi) 
            best_J = float(costs[j])
            best_u = U[j].copy()

        # update distribution (if iters>1)
        elite_idx = np.argsort(costs)[:elite_n]
        elite = U[elite_idx]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-6

    print("best Xpred: ", best_xpred)
    if best_u is None:
        best_u = np.clip(mu, -u_max, u_max)

    return best_u, best_J


def init_sing_mem():
    return {
        "prev_in_zone": False,
        "t_entry": None,
        "eta_entry": 0.0,      # eta = theta - pi (wrapped)
        "thetadot_entry": 0.0
    }

def update_sing_mem(sing_mem, in_zone, t, theta, thetadot):
    """
    On rising edge into singular zone, snapshot theta and thetadot.
    """
    if in_zone and (not sing_mem["prev_in_zone"]):
        sing_mem["t_entry"] = float(t)
        sing_mem["eta_entry"] = float(wrap_center(theta, np.pi) - np.pi)
        sing_mem["thetadot_entry"] = float(thetadot)
    sing_mem["prev_in_zone"] = bool(in_zone)
    return sing_mem


def theta_refs_piecewise(t, omega, theta_cfg, in_zone, sing_mem):
    """
    Returns (theta_ref, thetadot_ref, thetaddot_ref).

    Outside singular zone:
      theta_ref = sinusoid
    Inside singular zone:
      theta_ref = linear continuation using slope at entry
    """
    amp = float(theta_cfg["amp"])
    phase = float(theta_cfg.get("phase", 0.0))

    # --- inside singular zone: linear continuation ---
    if in_zone and sing_mem["t_entry"] is not None:
        dt = float(t) - sing_mem["t_entry"]
        eta_ref = sing_mem["eta_entry"] + sing_mem["thetadot_entry"] * dt
        theta_rf = lambda tt: np.pi + eta_ref
        thetadot_rf = lambda tt: sing_mem["thetadot_entry"]
        thetaddot_rf = lambda tt: 0.0
        return theta_rf, thetadot_rf, thetaddot_rf

    # --- outside singular zone: original sinusoid ---
    theta_rf = lambda tt: theta_ref(tt, omega, amp, phase)
    thetadot_rf = lambda tt: thetadot_ref(tt, omega, amp, phase)
    thetaddot_rf = lambda tt: thetaddot_ref(tt, omega, amp, phase)
    return theta_rf, thetadot_rf, thetaddot_rf



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


def control_law(x, t, p, omega, cfg, thdd_ff_prev, sing_mem):
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
    sing_mem = update_sing_mem(sing_mem, sing, t, theta, thetadot)
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

    return float(u), float(b_theta), float(e), float(s), float(thdd_ff), float(ph_r), float(phd_r), (1.0 if sing else 0.0), sing_mem


def control_law_hybrid_mpc(x, t, p, omega, cfg, thdd_ff_prev, mpc_state, sing_mem):
    """
    Returns final u after blending:
      u = (1-alpha)*u_smc + alpha*u_mpc
    where alpha depends on singular weight w with hysteresis.
    """
    ctrl = cfg["controller"]
    mpc_cfg = cfg.get("mpc", {"enabled": False})
    u_max = float(ctrl.get("u_max_override", p.u_max))
    solve_every = int(mpc_cfg.get("solve_every", 10))

    # --- nominal control (FF+SMC) ---
    u_smc, b_theta, e, s, thdd_ff, ph_r, phd_r, w, sing_mem = control_law(
        x, t, p, omega, cfg, thdd_ff_prev, sing_mem
    )

    if not mpc_cfg.get("enabled", False):
        return u_smc, b_theta, e, s, thdd_ff, ph_r, phd_r, w, mpc_state

    # --- hysteresis for turning MPC on/off ---
    w_enable = float(mpc_cfg.get("w_enable", 0.35))
    w_disable = float(mpc_cfg.get("w_disable", 0.20))

    if (not mpc_state["active"]) and (w >= w_enable):
        mpc_state["active"] = True
    elif mpc_state["active"] and (w <= w_disable):
        mpc_state["active"] = False

    # --- blending weight alpha ---
    # even if MPC inactive, allow alpha=0
    blend_w0 = float(mpc_cfg.get("blend_w0", 0.25))
    blend_w1 = float(mpc_cfg.get("blend_w1", 0.60))
    if w <= blend_w0:
        alpha = 0.0
    elif w >= blend_w1:
        alpha = 1.0
    else:
        alpha = smoothstep((w - blend_w0) / max(1e-12, (blend_w1 - blend_w0)))

    # if MPC not active, force alpha to 0
    if not mpc_state["active"]:
        alpha = 0.0
    # --- compute MPC action if needed ---
    u_mpc = 0.0
    if alpha > 0.0:
        # define theta references used by MPC (same theta ref as controller)
        theta_cfg = cfg["theta_ref"]
        amp = float(theta_cfg["amp"])
        # phase = float(theta_cfg["phase"])
        # phase_hat = theta_phase_estimate(x[1], x[3], amp, omega)
        # phase = wrap_center(phase_hat - omega*t, np.pi)

        # def th_ref_fun(tt):
        #     return theta_ref(tt, omega, amp, phase)

        # def thd_ref_fun(tt):
        #     return thetadot_ref(tt, omega, amp, phase)
        th_ref_fun, thd_ref_fun, _ = theta_refs_piecewise(
            t, omega, theta_cfg, w, sing_mem
        )

        # warm start plan: shift last plan
        warm = None
        if mpc_state.get("u_plan") is not None:
            up = np.asarray(mpc_state["u_plan"], float)
            if up.size >= 2:
                warm = np.r_[up[1:], up[-1]]
        if mpc_state.get("k_counter", 0) % solve_every != 0:
            # reuse last u_hold
            u_mpc = float(mpc_state.get("u_hold", 0.0))
        else:
            u_mpc, best_seq, info = cem_mpc(
                x0=x, t0=t, p=p,
                kappa=cfg["_case_key"]["kappa"],
                G_shape=cfg["_case_key"]["G_shape"],
                theta_ref_fun=th_ref_fun,
                thetadot_ref_fun=thd_ref_fun,
                u_prev=u_smc,             # use smc as prior control
                u_max=u_max,
                N=int(mpc_cfg.get("N", 15)),
                dt=float(mpc_cfg.get("dt", 0.02)),
                n_sub=int(mpc_cfg.get("n_sub", 2)),
                iters=int(mpc_cfg.get("iters", 4)),
                pop=int(mpc_cfg.get("pop", 250)),
                elite_frac=float(mpc_cfg.get("elite_frac", 0.15)),
                init_sigma=float(mpc_cfg.get("init_sigma", 5.0)),
                w_theta=float(mpc_cfg.get("w_theta", 120.0)),
                w_thdot=float(mpc_cfg.get("w_thdot", 30.0)),
                w_u=float(mpc_cfg.get("w_u", 0.05)),
                w_du=float(mpc_cfg.get("w_du", 8.0)),
                warm_start=warm,
                rng=mpc_state["rng"]
            )
            mpc_state["u_plan"] = best_seq
            mpc_state["u_hold"] = u_mpc
        mpc_state["k_counter"] = mpc_state.get("k_counter", 0) + 1

    # blend and saturate
    u = (1.0 - alpha) * u_smc + alpha * u_mpc
    if bool(ctrl.get("use_sat", True)):
        u = sat_u(u, u_max)

    return float(u), b_theta, e, s, thdd_ff, ph_r, phd_r, w, mpc_state, sing_mem



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


def rk4_step_closed_loop_mpc(x, t0, dt, n_sub, p, omega, cfg, thdd_ff_prev, mpc_state, sing_mem, u_prev=0.0):
    # # compute u once for the whole dt (with hybrid logic)
    # # u, bth, e, s, thdd_ff, ph_r, phd_r, w, mpc_state, sing_mem = control_law_hybrid_mpc(
    # #     x, t0, p, omega, cfg, thdd_ff_prev, mpc_state, sing_mem
    # # )
    # mpc_cfg = cfg["mpc"]
    # theta = x[1]
    # thetadot = x[3]

    # # pre-zone detection (before exact singularity)
    # eta = abs(wrap_center(theta, np.pi) - np.pi)
    # zone_now = abs(eta - np.pi/2) < np.deg2rad(mpc_cfg["pre_deg"])

    # toward_peak = (eta * thetadot) > 0.0

    # zone_entry = zone_now and not mpc_state["prev_zone"]
    # zone_exit = (not zone_now) and mpc_state["prev_zone"]
    # mpc_state["prev_zone"] = zone_now

    # # update singular memory
    # sing_mem = update_sing_mem(sing_mem, zone_now, t0, theta, thetadot)

    # if mpc_cfg["enabled"] and zone_entry:
    #     # build exit target from linear continuation
    #     dt_mpc = mpc_cfg["dt"]
    #     N_mpc = mpc_cfg["N"]
    #     t_exit = t0 + dt_mpc * N_mpc

    #     theta_exit, thetadot_exit, _ = theta_refs_piecewise(
    #         t_exit, omega, cfg["theta_ref"], True, sing_mem
    #     )
    #     theta_exit = theta_exit(t_exit)
    #     thetadot_exit = thetadot_exit(t_exit)

    #     u_max_mpc = cfg["mpc"]["u_max_override"]

    #     plan, _ = cem_solve_zone_mpc(
    #         x0=x,
    #         t0=t0,
    #         p=p,
    #         kappa=cfg["_case_key"]["kappa"],
    #         G_shape=cfg["_case_key"]["G_shape"],
    #         theta_exit=theta_exit,
    #         thetadot_exit=thetadot_exit,
    #         u_prev=u_prev,
    #         u_max=u_max_mpc,
    #         dt=dt_mpc,
    #         N=N_mpc,
    #         n_sub=mpc_cfg["n_sub"],
    #         iters=mpc_cfg["iters"],
    #         pop=mpc_cfg["pop"],
    #         elite_frac=mpc_cfg["elite_frac"],
    #         init_sigma=mpc_cfg["init_sigma"],
    #         w_exit_theta=mpc_cfg["w_exit_theta"],
    #         w_exit_thdot=mpc_cfg["w_exit_thdot"],
    #         w_exit_energy=mpc_cfg["w_exit_energy"],
    #         w_u=mpc_cfg["w_u"],
    #         w_du=mpc_cfg["w_du"],
    #         rng=mpc_state["rng"],
    #         warm_start=mpc_state["u_plan"]
    #     )

    #     mpc_state["active"] = True
    #     mpc_state["u_plan"] = plan
    #     mpc_state["idx"] = 0
        
    # if mpc_cfg["enabled"] and mpc_state["active"] and zone_now:
    #     if mpc_state["idx"] < len(mpc_state["u_plan"]):
    #         u = mpc_state["u_plan"][mpc_state["idx"]]
    #         mpc_state["idx"] += 1
    #     else:
    #         u = mpc_state["u_plan"][-1]
    #     sing = zone_now
    # else:
    #     u, bth, e, s, thdd_ff_prev, ph_r, phd_r, sing, sing_mem = control_law(
    #         x, t0, p, omega, cfg, thdd_ff_prev, sing_mem
    #     )

    # if zone_exit:
    #     mpc_state["active"] = False
    #     mpc_state["u_plan"] = None
    #     mpc_state["idx"] = 0

    eta = wrap_center(x[1], np.pi) - np.pi
    thdot = x[3]
    A = float(cfg["theta_ref"]["amp"])
    u_max = float(cfg["controller"].get("u_max_override", p.u_max))
    u_max_mpc = float(cfg["mpc"].get("u_max_override", u_max))
    
    mpc_peak = cfg.get("mpc", {"enabled": False})
    # pre-zone band around pi/2
    pre_deg = float(mpc_peak.get("pre_deg", 20.0))

    near_zone = abs(abs(eta) - np.pi/2.0) < np.deg2rad(pre_deg)

    # trigger condition: near zone AND moving toward the nearby peak
    toward_peak = (eta * thdot) > 0.0

    # --- trigger MPC once ---
    if bool(mpc_peak.get("enabled", False)) and (not mpc_state["active"]) and near_zone and toward_peak:
        dt_mpc = float(mpc_peak.get("dt", 0.02))
        n_sub_mpc = int(mpc_peak.get("n_sub", 1))

        T_peak = compute_T_to_peak(eta_entry=eta, A=A, omega=omega)
        N_mpc = int(np.ceil(T_peak *2.0 / dt_mpc))
        N_mpc = max(3, min(N_mpc, int(mpc_peak.get("N_max", 25))))  # cap for safety/speed

        # target peak sign = sign(eta) (since moving toward peak of that sign)
        eta_target = float(np.sign(eta) * A)
        thdot_target = 0.0
        
        warm = mpc_state["u_plan"]
        plan, Jbest = cem_solve_peak_mpc(
            x0=x,t0=t0, p=p,
            kappa=cfg["_case_key"]["kappa"],
            G_shape=cfg["_case_key"]["G_shape"],
            eta_target=eta_target,
            thdot_target=thdot_target,
            u_prev=u_prev, u_max=u_max_mpc,
            dt=dt_mpc, N=N_mpc, n_sub=n_sub_mpc,
            iters=int(mpc_peak.get("iters", 1)),
            pop=int(mpc_peak.get("pop", 120)),
            elite_frac=float(mpc_peak.get("elite_frac", 0.2)),
            init_sigma=float(mpc_peak.get("init_sigma", 6.0)),
            w_eta=float(mpc_peak.get("w_eta", 500.0)),
            w_thdot=float(mpc_peak.get("w_thdot", 10.0)),
            w_u=float(mpc_peak.get("w_u", 0.02)),
            w_du=float(mpc_peak.get("w_du", 1.0)),
            rng=mpc_state["rng"],
            warm_start=warm
        )

        mpc_state["active"] = True
        mpc_state["u_plan"] = plan
        mpc_state["t_end"] = t0 + dt_mpc * len(plan)
        mpc_state["idx"] = 0
        mpc_state["last_J"] = Jbest
        mpc_state["eta_target"] = eta_target
    

    use_mpc_now = mpc_state["active"] and (t0 < mpc_state.get("t_end", -np.inf))

    if use_mpc_now:
        idx = mpc_state["idx"]
        if idx < len(mpc_state["u_plan"]):
            u = float(mpc_state["u_plan"][idx])
            mpc_state["idx"] += 1
        else:
            # Plan exhausted early -> stop MPC and fall back immediately
            mpc_state["active"] = False
            mpc_state["u_plan"] = None
            mpc_state["idx"] = 0
            u, bth, e, s, thdd_ff_prev, ph_r, phd_r, sing, sing_mem = control_law(
                x, t0, p, omega, cfg, thdd_ff_prev, sing_mem
            )
        sing = near_zone
    else:
        # Time-based stop -> stop MPC and fall back
        if mpc_state["active"] and t0 >= mpc_state.get("t_end", -np.inf):
            mpc_state["active"] = False
            mpc_state["u_plan"] = None
            mpc_state["idx"] = 0

        u, bth, e, s, thdd_ff_prev, ph_r, phd_r, sing, sing_mem = control_law(
            x, t0, p, omega, cfg, thdd_ff_prev, sing_mem
        )



    
    bth, _, ph_r, phd_r, th_r_used, thd_r_used, e, s = compute_diagnostics(
        x, t0, p, omega, cfg, sing_mem
    )


    # integrate plant for dt using n_sub substeps with constant u
    b0_true, b1_true = p.b0_nom, p.b1_nom
    dt_int = dt / float(n_sub)
    xk = np.asarray(x, float).copy()
    for _ in range(int(n_sub)):
        xk = rk4_step(rhs_continuous, xk, u, dt_int, p,
                      kappa=float(cfg["_case_key"]["kappa"]),
                      G_shape=cfg["_case_key"]["G_shape"],
                      b_theta_true = lambda theta: b_theta_true(theta, b0_true, b1_true))
    return xk, u, bth, e, s, thdd_ff_prev, ph_r, phd_r, sing, mpc_state, sing_mem


def compute_T_to_peak(eta_entry, A, omega):
    """
    Time from current |eta| to |eta|=A assuming sinusoidal phase advance.
    """
    r = min(0.999999, abs(eta_entry) / max(1e-9, A))
    dpsi = (np.pi/2.0) - np.arcsin(r)
    return float(dpsi / max(1e-9, omega))




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

    mpc_peak = cfg.get("mpc_peak", {"enabled": False})

    mpc_state = {
        "active": False,
        "u_plan": None,
        "rng": np.random.default_rng(0),
        "idx": 0,
        "prev_zone": False,
        "u_hold": 0.0,
        "k_counter": 0,
        "last_J": np.nan
    }

    sing_mem = init_sing_mem()

    mpc_cfg = cfg["mpc"]

    for k in range(N):
        print(f"{k + 1} / {N}", end="\r")
        X[k] = x
        u_prev = U[k-1] if k > 0 else 0.0

        x, u, bth, e, s, thdd_ff_prev, ph_r, phd_r, sing, mpc_state, sing_mem = rk4_step_closed_loop_mpc(
            x, t[k], dt, n_sub, p, omega, cfg, thdd_ff_prev, mpc_state, sing_mem, u_prev=u_prev
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
