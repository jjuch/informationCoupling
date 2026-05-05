import numpy as np
from config import FurutaParams
from furuta_model import (
    furuta_M_C_g,
    G_kappa,
    rhs_continuous,
    rk4_step,
    wrap_center, b_theta_true
)

def theta_dev(theta: float) -> float:
    """eta = wrap(theta - pi) in (-pi,pi]."""
    return float(wrap_center(theta, np.pi) - np.pi)

def sat_u(u: float, u_max: float) -> float:
    return float(np.clip(u, -u_max, u_max))

def sat(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))

def omega_small_angle(p: FurutaParams, scaling: float = 1.0) -> float:
    """Small-angle natural frequency about theta=pi for Gafvert model."""
    # Linearization around theta=pi gives omega^2 ≈ delta / beta
    return float(scaling * np.sqrt(p.delta / p.beta))

def theta_energy_about_pi(eta: float, thetadot: float, beta: float, delta: float) -> float:
    """
    A simple 'virtual energy' around theta=pi.
    For eta = theta - pi wrapped, stable equilibrium is eta=0.
    """
    return float(0.5 * beta * (thetadot ** 2) + delta * (1.0 - np.cos(eta)))

# -----------------------------
# Collocated VHC shape: phi = varphi(theta)
# -----------------------------

def varphi(theta: float, a_phi: float) -> float:
    """phi shape as a function of theta: a_phi * sin(eta), eta = wrap(theta-pi)."""
    eta = theta_dev(theta)
    return float(a_phi * np.sin(eta))

def dvarphi(theta: float, a_phi: float) -> float:
    """Derivative d/dtheta varphi(theta) ≈ a_phi * cos(eta)."""
    eta = theta_dev(theta)
    return float(a_phi * np.cos(eta))

def ddvarphi(theta: float, a_phi: float) -> float:
    """Second derivative d^2/dtheta^2 varphi(theta) ≈ -a_phi * sin(eta)."""
    eta = theta_dev(theta)
    return float(-a_phi * np.sin(eta))


# -----------------------------
# Affine decomposition (includes nonlinear damping)
# -----------------------------

def affine_qdd_terms(x, p: FurutaParams, kappa: float, G_shape: str):
    """
    Extract affine-in-u accelerations:
      qdd = f(q,qd) + Minv[:,0] * u   because tau=[u,0]^T.

    Returns:
      f_phi, f_theta, b_phi, b_theta, Minv
    where:
      phiddot = f_phi + b_phi*u
      thetaddot = f_theta + b_theta*u
    """
    phi, theta, phidot, thetadot = x
    q = np.array([phi, theta], dtype=float)
    qd = np.array([phidot, thetadot], dtype=float)

    M, C, gvec = furuta_M_C_g(q, qd, p)
    G = G_kappa(q, float(kappa), M, p, shape=G_shape)

    # friction 
    b0_true, b1_true = p.b0_nom, p.b1_nom
    theta_fric = b_theta_true(theta, b0_true, b1_true)
    tau_f = np.array([p.b_phi * phidot, theta_fric * thetadot], dtype=float)

    # drift accelerations with u=0:
    qdd0 = np.linalg.solve(M, -(C + G) @ qd - gvec - tau_f)

    Minv = np.linalg.inv(M)
    b_vec = Minv[:, 0]  # column multiplying u

    f_phi = float(qdd0[0])
    f_theta = float(qdd0[1])
    b_phi = float(b_vec[0])
    b_theta = float(b_vec[1])
    return f_phi, f_theta, b_phi, b_theta, Minv



# -----------------------------
# Control law: Dynamic VHC + energy-based orbit stabilizer
# -----------------------------

def control_law(z, t, p: FurutaParams, cfg: dict):
    """
    Dynamic VHC + orbit stabilizer.

    Augmented state:
      z = [phi, theta, phidot, thetadot, s, sdot]
    with sddot = v.

    Constraint:
      e = theta - Phi(phi) - s
      ed = thetadot - dPhi(phi)*phidot - sdot


    Using affine dynamics:
      phiddot   = f_phi   + b_phi*u
      thetaddot = f_theta + b_theta*u

    Then:
      e_dd = (b_phi - varphi'(theta)*b_theta) * u
             + (f_phi - varphi'(theta)*f_theta - varphi''(theta)*thetadot^2)
             - v

    Choose:
      u = (v - alpha - kd*ed - kp*e)/Adec
    where:
      Adec  = b_phi - varphi'(theta)*b_theta
      alpha = f_phi - varphi'(theta)*f_theta - varphi''(theta)*thetadot^2

    => e_dd + kd*ed + kp*e = 0 (independent of v).

    Orbit stabilizer (v): regulate theta-energy about pi to match amplitude A,
    while driving s, sdot -> 0 so constraint collapses back to nominal.
    """
    phi, theta, phidot, thetadot, s, sdot = z
 
    # ---- target amplitude and omega (omega used only for diagnostics/optional deadbands) ----
    Aamp = float(cfg["theta"]["A"])
    omega_mode = cfg["theta"]["omega_mode"]
    omega_scaling = float(cfg["theta"].get("omega_scaling", 1.0))
    if omega_mode == "small_angle":
        omega_star = omega_small_angle(p, scaling=omega_scaling)
    else:
        omega_star = float(omega_mode)

    # --- VHC shape parameter a_phi ---
    vhc_cfg = cfg["vhc"]
    a_phi = float(vhc_cfg.get("a_phi", 0.5))

    # --- constraint signals ---
    var = varphi(theta, a_phi)
    dvar = dvarphi(theta, a_phi)
    ddvar = ddvarphi(theta, a_phi)

    # constraint signals
    e = float(phi - var - s)
    ed = float(phidot - dvar * thetadot - sdot)


    # affine acceleration terms
    kappa = float(cfg["_case_key"]["kappa"])
    G_shape = cfg["_case_key"]["G_shape"]
    f_phi, f_theta, b_phi, b_theta, _ = affine_qdd_terms(
        np.array([phi, theta, phidot, thetadot], float), p, kappa, G_shape
    )

    Adec = float(b_phi - dvar * b_theta)

    # Avoid division blow-up if Adec is tiny (regularization):
    Areg = 1e-6
    if abs(Adec) < Areg:
        Adec = np.sign(Adec) * Areg if Adec != 0.0 else Areg

    alpha = float(f_phi - dvar * f_theta - ddvar * (thetadot ** 2))

    # =========================================================
    # Orbit stabilizer (v): energy pumping about theta=pi
    # =========================================================
    orb = cfg["orbit"]
    ks = float(orb["ks"])
    ksd = float(orb["ksd"])
    kE = float(orb["kE"])
    k_align = float(orb.get("k_align", 0.0))
    v_max = float(orb.get("v_max", 100.0))

    # eta = wrap(theta-pi) - pi
    eta = theta_dev(theta)

    # energy about pi
    Etheta = theta_energy_about_pi(eta, thetadot, p.beta, p.delta)
    E_star = float(p.delta * (1.0 - np.cos(Aamp)))
    Eerr = float(Etheta - E_star)

    
    # base term: keep s -> 0 and sdot -> 0
    v_s = -ks * s - ksd * sdot

    # energy pump term: (E-E*) * thetadot
    #   - if E too high and thetadot has sign, term dissipates
    #   - if E too low, term injects energy phase-consistently
    v_energy = -kE * Eerr * thetadot

    
    # alignment term, but ONLY when energy is below target (Eerr < 0)
    #     and with its own clip to avoid forcing v to saturate.
    #     Also scale by a "deficit factor" so it turns off as Eerr -> 0.
    deficit = max(0.0, -Eerr)            # positive only if below target energy
    deficit_scale = float(orb.get("deficit_scale", 0.02))  # tune: sets when injection ramps
    gate = np.tanh(deficit / max(1e-9, deficit_scale))
    
    v_align_raw = k_align * Adec * phidot * gate        # makes u include ~k_align*phidot
    
    # clip alignment contribution to a fraction of v_max
    v_align_frac = float(orb.get("v_align_frac", 0.15))     # e.g. 15% of v_max
    v_align_max = v_align_frac * v_max
    v_align = float(np.clip(v_align_raw, -v_align_max, v_align_max))

    
    # Now allocate headroom: first saturate v_s, then add others within remaining margin
    v_s_sat = float(np.clip(v_s, -v_max, v_max))
    headroom = max(0.0, v_max - abs(v_s_sat))

    v_extra = float(v_energy + v_align)
    v_extra_sat = float(np.clip(v_extra, -headroom, headroom))

    v = v_s_sat + v_extra_sat
    v = float(np.clip(v, -v_max, v_max))

    v = float(np.clip(v, -v_max, v_max))


    # ---- Dynamic VHC enforcement: compute u that makes e second-order stable ----
    kp = float(vhc_cfg["kp"])
    kd = float(vhc_cfg["kd"])

    u_raw = (v - alpha - kd * ed - kp * e) / Adec

    u_max = float(cfg["limits"]["u_max"])
    u = sat_u(u_raw, u_max)

    diag = {
        "e": e,
        "ed": ed,
        "Adec": float(Adec),
        "alpha": float(alpha),
        "u_raw": float(u_raw),
        "u_sat": float(u),
        "u": float(u),
        "v": float(v),
        "v_s": float(v_s),
        "v_s_sat": float(v_s_sat),
        "v_align": float(v_align),
        "v_align_raw": float(v_align_raw),
        "gate": float(gate),
        "headroom": float(headroom),
        "omega_star": float(omega_star),
        "a_phi": float(a_phi),
        "varphi": float(var),
        "dvarphi": float(dvar),
        "ddvarphi": float(ddvar),
        "eta": float(eta),
        "Etheta": float(Etheta),
        "Etheta_star": float(E_star),
        "Eerr": float(Eerr),
        "v_energy": float(v_energy),
        "s": float(s),
        "sdot": float(sdot),
    }

    return float(u), float(v), diag

def simulate_closed_loop(cfg: dict):
    """
    Closed-loop simulation with RK4 substepping on the augmented 6D state:
      z = [x(4), s, sdot].
    """
    p = FurutaParams()
    dt = float(cfg["dt"])
    n_sub = int(cfg["n_sub"])
    T_total = float(cfg["T_total"])
    N = int(T_total / dt)
    t = np.arange(N) * dt

    # initial physical state near down position
    x0 = np.array([0.0, np.pi + 0.05, 0.0, 0.0], dtype=float)
    # initial VHC parameter
    s0 = 0.0
    sdot0 = 0.0

    z = np.hstack([x0, [s0, sdot0]]).astype(float)

    Z = np.zeros((N, 6), dtype=float)
    X = np.zeros((N, 4), dtype=float)
    U = np.zeros(N, dtype=float)
    V = np.zeros(N, dtype=float)

    E = np.zeros(N, dtype=float)
    ED = np.zeros(N, dtype=float)
    SARR = np.zeros(N, dtype=float)
    SDARR = np.zeros(N, dtype=float)
    OMEGA = np.zeros(N, dtype=float)
    ADEC = np.zeros(N, dtype=float)
    ALPHA = np.zeros(N, float)
    
    U_RAW = np.zeros(N, float)
    U_SAT = np.zeros(N, float)
    V_RAW = np.zeros(N, float)
    V_S = np.zeros(N, float)      # v - v_energy
    V_S_SAT = np.zeros(N, float)
    V_ENERGY = np.zeros(N, float)
    V_ALIGN = np.zeros(N, float)
    V_ALIGN_RAW = np.zeros(N, float)
    GATE = np.zeros(N, float)
    HEADROOM = np.zeros(N, float)


    PIN = np.zeros(N, float)      # u * phidot
    PD_TH = np.zeros(N, float)    # b(theta) * thetadot^2
    PD_PH = np.zeros(N, float)    # b_phi * phidot^2

    THETA_DEV = np.zeros(N, float)
    THETADOT = np.zeros(N, float)
    PHIDOT = np.zeros(N, float)
    
    ETA = np.zeros(N, dtype=float)
    ETH = np.zeros(N, dtype=float)
    ETHS = np.zeros(N, dtype=float)
    EERR = np.zeros(N, dtype=float)
    VENER = np.zeros(N, dtype=float)

    dt_int = dt / float(n_sub)

    def closed_loop_rhs(z_local, t_local):
        # compute control from current state
        u_local, v_local, diag = control_law(z_local, t_local, p, cfg)
        x_local = z_local[:4]

        b0_true, b1_true = p.b0_nom, p.b1_nom
        b_theta = lambda th: b_theta_true(th, b0_true, b1_true)
        # plant derivative
        dx = rhs_continuous(
            x_local, u_local, p,
            kappa=float(cfg["_case_key"]["kappa"]),
            G_shape=cfg["_case_key"]["G_shape"],
            b_theta_true=b_theta
        )

        # s dynamics: sdot = z[5], sddot = v
        ds = float(z_local[5])
        dsdot = float(v_local)

        dz = np.zeros_like(z_local)
        dz[:4] = dx
        dz[4] = ds
        dz[5] = dsdot
        return dz, u_local, v_local, diag


    for k in range(N):
        print(f"{k + 1} / {N}", end="\r")
        Z[k] = z
        X[k] = z[:4]

        # control for logging at macro-step
        u_k, v_k, diag_k = None, None, None
        # integrate with RK4 substeps using control recomputed at each stage
        for _ in range(n_sub):
            k1, u1, v1, d1 = closed_loop_rhs(z, t[k])
            k2, _,  _,  _  = closed_loop_rhs(z + 0.5 * dt_int * k1, t[k] + 0.5 * dt_int)
            k3, _,  _,  _  = closed_loop_rhs(z + 0.5 * dt_int * k2, t[k] + 0.5 * dt_int)
            k4, _,  _,  _  = closed_loop_rhs(z + dt_int * k3, t[k] + dt_int)
            z = z + (dt_int / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            u_k, v_k, diag_k = u1, v1, d1

        
        if k < 2:
            print(f"\n=== DEBUG k={k} ===")
            print("theta_dev =", theta_dev(X[k,1]), "thetadot =", X[k,3])
            print("Etheta =", diag_k.get("Etheta"), "E* =", diag_k.get("Etheta_star"), "Eerr =", diag_k.get("Eerr"))
            print("v_energy =", diag_k.get("v_energy"), "v =", diag_k.get("v"))
            print("u_raw =", diag_k.get("u_raw"), "u =", u_k)
            print("Adec =", diag_k.get("Adec"), "alpha =", diag_k.get("alpha"))
            print("================\n")


        U[k] = float(u_k)
        V[k] = float(v_k)
        E[k] = float(diag_k["e"])
        ED[k] = float(diag_k["ed"])
        SARR[k] = float(diag_k["s"])
        SDARR[k] = float(diag_k["sdot"])
        OMEGA[k] = float(diag_k["omega_star"])
        ADEC[k] = float(diag_k["Adec"])
        ALPHA[k] = float(diag_k.get("alpha", np.nan))
        
        U_RAW[k] = float(diag_k.get("u_raw", np.nan))
        U_SAT[k] = float(diag_k.get("u_sat", U[k]))
        V_RAW[k] = float(diag_k.get("v", np.nan))
        V_S[k] = float(diag_k.get("v", 0.0) - diag_k.get("v_energy", 0.0))
        V_S_SAT[k] = float(diag_k.get("v_s_sat", np.nan))
        V_ENERGY[k] = float(diag_k.get("v_energy", np.nan))
        V_ALIGN[k] = float(diag_k.get("v_align", np.nan))
        V_ALIGN_RAW[k] = float(diag_k.get("v_align_raw", np.nan))
        GATE[k] = float(diag_k.get("gate", np.nan))
        HEADROOM[k] = float(diag_k.get("headroom", np.nan))

        THETA_DEV[k] = theta_dev(X[k, 1])
        THETADOT[k] = X[k, 3]
        PHIDOT[k] = X[k, 2]
        
        # power accounting (use same nonlinear damping you simulate)
        b0_true, b1_true = p.b0_nom, p.b1_nom
        bth = b_theta_true(X[k, 1], b0_true, b1_true)

        PIN[k] = U[k] * X[k, 2]
        PD_TH[k] = bth * (X[k, 3] ** 2)
        PD_PH[k] = p.b_phi * (X[k, 2] ** 2)


        ETA[k] = float(diag_k["eta"])
        ETH[k] = float(diag_k["Etheta"])
        ETHS[k] = float(diag_k["Etheta_star"])
        EERR[k] = float(diag_k["Eerr"])
        VENER[k] = float(diag_k["v_energy"])

    base = {
        "t": t,
        "X": X,
        "Z": Z,
        "U": U,
        "V": V,
        "E": E,
        "ED": ED,
        "S": SARR,
        "SD": SDARR,
        "omega": OMEGA,
        "Adec": ADEC,
        "alpha": ALPHA,
        "u_raw": U_RAW,
        "u_sat": U_SAT,
        "v_raw": V_RAW,
        "v_s": V_S,
        "v_s_sat": V_S_SAT,
        "v_energy": V_ENERGY,
        "v_align": V_ALIGN,
        "v_align_raw": V_ALIGN_RAW,
        "gate": GATE,
        "headroom": HEADROOM,
        "theta_dev": THETA_DEV,
        "thetadot": THETADOT,
        "phidot": PHIDOT,
        "P_in": PIN,
        "P_d_theta": PD_TH,
        "P_d_phi": PD_PH,
        "eta": ETA,
        "Etheta": ETH,
        "Etheta_star": ETHS,
        "Eerr": EERR,
        "v_energy": VENER,
    }

    return base, cfg
