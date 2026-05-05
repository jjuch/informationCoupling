import numpy as np
from copy import copy
from config import FurutaParams
from furuta_model import rhs_continuous, rk4_step, wrap_center, b_theta_true
from scenarioE_mpc import solve_mpc_bridge, MPCBridgeSolver


# ---------- helpers ----------
def theta_dev(theta):
    return wrap_center(theta, np.pi) - np.pi

def sat_u(u, u_max):
    return float(np.clip(u, -u_max, u_max))

def omega_small_angle(p: FurutaParams, scaling=1.0):
    return float(scaling * np.sqrt(p.delta / p.beta))

def phase_from_state(eta, thetadot, A, omega):
    # eta = A sin(psi), thetadot = A omega cos(psi)
    return float(np.arctan2(eta / max(1e-9, A), thetadot / max(1e-9, A*omega)))


# ---------- mode enum ----------
MODE_PUMP = 0
MODE_SMC  = 1
MODE_MPC_UP = 2
MODE_MPC_DOWN = 3

MODE_NAME = {
    MODE_PUMP: "PUMP",
    MODE_SMC: "SMC",
    MODE_MPC_UP: "MPC_UP",
    MODE_MPC_DOWN: "MPC_DOWN"
}

# ---------- controller components ----------
def init_pump_state(cfg):
    pump = cfg["pump"]
    return {
        "u_amp": float(pump["u0"]),
        "u_sign": 1.0,                  # current sign (+1 or -1)
        "eta_prev": None,               # previous eta for sign-change detection
        "last_flip_t": -1e9,            # last time we flipped sign
        "cycle_count": 0,               # how many flips occurred (optional),
        "increase_u": True,
        "cross_state": 0
    }


def pump_control(x, t, cfg, u_max_global, pump_state):
    """
    Pumping law:
      - use eta = wrap(theta-pi)
      - when eta crosses 0 (with debounce), flip torque sign and ramp amplitude by du
      - apply u = u_sign * u_amp - k_phidot * phidot (optional)

    """
    phi, theta, phidot, thetadot = x
    pump = cfg["pump"]

    u_max_pump = float(pump.get("u_max_pump", u_max_global))
    du = float(pump["du"])
    eta_db = float(pump.get("eta_deadband", 0.05))
    min_flip_time = float(pump.get("min_flip_time", 0.08))
    k_phidot = float(pump.get("k_phidot", 0.0))
    k_phi = float(pump.get("k_phi", 0.0))

    eta = theta_dev(theta)

    # initialize eta_prev on first call
    if pump_state["eta_prev"] is None:
        pump_state["eta_prev"] = eta

    eta_prev = float(pump_state["eta_prev"])

    # -------- Debounced zero-crossing detection --------
    # Count a crossing only if:
    #  1) eta changes sign across zero
    #  2) both previous and current are outside deadband (avoid chattering near 0)
    #  3) enough time passed since last flip
    # crossed = (eta_prev < eta and eta >= eta_db) or (eta_prev > eta and eta <= -eta_db)
    old_cross_state = pump_state["cross_state"]
    if eta > eta_db:
        cross_state = 1
    elif eta < -eta_db:
        cross_state = -1
    else:
        cross_state = old_cross_state
    
    crossed = cross_state != old_cross_state
    pump_state["cross_state"] = cross_state
    far_enough = True # (abs(eta_prev) > eta_db) and (abs(eta) > eta_db)
    time_ok = True #(t - pump_state["last_flip_t"]) >= min_flip_time
    
    if crossed and far_enough and time_ok:
        pump_state["u_sign"] *= -1.0
        if pump_state["increase_u"]:
            pump_state["u_amp"] = min(u_max_pump, pump_state["u_amp"] + du)
        else:
            pump_state["u_amp"] = min(u_max_pump, pump_state["u_amp"])
        pump_state["last_flip_t"] = float(t)
        pump_state["cycle_count"] += 1

    # update memory
    pump_state["eta_prev"] = eta

    # torque command
    
    u = pump_state["u_sign"] * pump_state["u_amp"] - k_phidot * phidot - k_phi * phi
    u = float(np.clip(u, -u_max_pump, u_max_pump))


    # for plotting “pump reference”, we can optionally expose the current target sign/amplitude
    # here we just return NaNs (since pump is torque-based, not phi-ref-based)
    return u, pump_state, crossed



def smc_control(x, t, omega, phase_state, cfg, u_max):
    """
    SMC to maintain oscillation outside singular zone.
    Reference is sinusoid continuation defined by phase_state.
    """
    phi, theta, phidot, thetadot = x
    A = float(cfg["theta"]["A"])
    lam = float(cfg["smc"]["lambda"])
    K = float(cfg["smc"]["K"])
    phi_s = float(cfg["smc"]["phi_s"])
    k_phi = float(cfg["smc"]["k_phi"])
    k_phidot = float(cfg["smc"]["k_phidot"])

    # advance phase
    phase_state["psi"] = phase_state["psi"] + omega * phase_state["dt"]
    psi = phase_state["psi"]

    eta = theta_dev(theta)
    eta_ref = A * np.sin(psi)
    etadot_ref = A * omega * np.cos(psi)

    phi_ref = A * np.sin(psi - np.pi/6)
    phidot_ref = A * omega * np.cos(psi - np.pi/6)

    e = (eta - eta_ref)
    ed = (thetadot - etadot_ref)
    s = ed + lam * e

    u = -K * np.clip(s / phi_s, -1.0, 1.0) - k_phi * (phi - phi_ref) - k_phidot * (phidot - phidot_ref)
    return sat_u(u, u_max), eta_ref, etadot_ref, phi_ref, phidot_ref, s


# ---------- simulation ----------
def simulate_closed_loop(cfg):
    p = FurutaParams()

    dt = float(cfg["dt"])
    n_sub = int(cfg["n_sub"])
    T_total = float(cfg["T_total"])
    N = int(T_total / dt)
    t = np.arange(N) * dt

    # omega
    if cfg["theta"]["omega_mode"] == "small_angle":
        omega = omega_small_angle(p, scaling=cfg["theta"]["omega_scaling"])
    else:
        omega = float(cfg["theta"]["omega_mode"])

    A = float(cfg["theta"]["A"])
    u_max = float(cfg["limits"]["u_max"])

    pump_state = init_pump_state(cfg)

    # state init
    x = np.array([0.0, np.pi + 0.1, 0.0, 0.0], float)

    # storage
    X = np.zeros((N,4), float)
    U = np.zeros(N, float)
    MODE = np.zeros(N, int)
    PHREF = np.full(N, np.nan, float)
    PHDREF = np.full(N, np.nan, float)
    THREF = np.full(N, np.nan, float)      # reference for theta-pi (eta_ref)
    THDREF = np.full(N, np.nan, float)     # reference for thetadot (etadot_ref)
    S_SURF = np.full(N, np.nan, float)
    
    MPC_OBJ = np.full(N, np.nan, float)
    MPC_STATUS = np.array([""]*N, dtype=object)


    # mode switching thresholds
    pre_deg = float(cfg["modes"]["pre_sing_deg"])
    sing_deg = float(cfg["modes"]["sing_deg"])
    pre_band = np.deg2rad(pre_deg)
    sing_band = np.deg2rad(sing_deg)

    # pump stop by energy (simple proxy)
    # E_target = 1.5 * float(p.delta)
    # E_frac_stop = float(cfg["pump"]["E_frac_stop"])

    # phase state for SMC continuation
    phase_state = {"psi": 0.0, "dt": dt}

    # MPC state
    mpc_cfg = cfg["mpc"]
    mpc_enabled = bool(mpc_cfg["enabled"])
    mpc_u_plan = None
    mpc_u_idx = 0
    mpc_mode = None
    mpc_info_last = None

    # horizon = quarter period
    Tquarter = (2.0*np.pi/omega) / 4.0
    bridge = None
    if mpc_enabled:
        bridge = MPCBridgeSolver(p=p, kappa=float(cfg["_case_key"]["kappa"]),
                        G_shape=cfg["_case_key"]["G_shape"], Tquarter=Tquarter, u_max=u_max, mpc_cfg=cfg["mpc"])

    mpc_u_plan = None
    mpc_u_idx = 0   

    mode = MODE_PUMP if cfg["pump"]["enabled"] else MODE_SMC

    dt_int = dt / float(n_sub)

    b0_true, b1_true = p.b0_nom, p.b1_nom
    keep_track = True
    in_preband = False
    crossed = True

    for k in range(N):
        print(f"{k + 1} / {N}", end="\r")
        X[k] = x
        phi, theta, phidot, thetadot = x
        eta = theta_dev(theta)

        # distance to singular
        dist = abs(abs(eta) - np.pi/2)

        # phase init when entering SMC
        if k == 0:
            phase_state["psi"] = phase_from_state(eta, thetadot, A, omega)

        # energy proxy
        E = 0.5 * float(p.beta) * (thetadot**2) + float(p.delta) * (1.0 - np.cos(eta))

        # ---------- mode transitions ----------
        if mode == MODE_PUMP:
            # stop pumping once close to pre-zone or enough energy
            if pump_state["increase_u"]: print("\t\t\t\t", pump_state["increase_u"], end="\r")
            else: print("\t\t\t\t----------", end="\r")

            if dist < pre_band: #or E >= E_frac_stop * E_target:
                
                pump_state["increase_u"] = False
                if not in_preband and crossed:
                    pump_state["u_sign"] *= -1
                    in_preband = True
                    crossed = False
                if keep_track:
                    pump_state["u_amp"] *= 1.05
                    # pump_state["u_sign"] *=-1
                    keep_track = False
                mode = MODE_SMC
                phase_state["psi"] = phase_from_state(eta, thetadot, A, omega)


        if mode == MODE_SMC and mpc_enabled:
            # enter MPC when close to singular band
            if dist < sing_band:
                # decide direction: outward -> go to peak first
                outward = (eta * thetadot) > 0.0
                if outward:
                    mode = MODE_MPC_UP
                else:
                    # if already turning inward, we go DOWN to pi/2
                    mode = MODE_MPC_DOWN

                # clear plan so we solve immediately
                mpc_u_plan = None
                mpc_u_idx = 0

        # after MPC segments, return to SMC
        # if mode in (MODE_MPC_UP, MODE_MPC_DOWN) and (mpc_u_plan is None) and (mpc_u_idx == 0):
        #     pass

        # ---------- control ----------
        u = 0.0
        print("\t\t\t\t\t ", mode, end="\r")
        if mode == MODE_PUMP:
            u, pump_state, crossed_pump = pump_control(x=x,
                                         t=t[k],
                                         cfg=cfg,
                                         u_max_global=u_max,pump_state=pump_state
                                        )
            if crossed_pump:
                crossed = True
            # PHREF[k] = np.nan
            # PHDREF[k] = np.nan
            MODE[k] = MODE_PUMP

        elif mode == MODE_SMC:
            u, eta_ref, etadot_ref, phi_ref, phidot_ref, s = smc_control(x, t[k], omega, phase_state, cfg, u_max)
            THREF[k] = eta_ref
            THDREF[k] = etadot_ref
            PHREF[k] = phi_ref
            PHDREF[k] = phidot_ref
            S_SURF[k] = s
            MODE[k] = MODE_SMC

        else:
            # ---------- MPC bridge ----------
            # Solve a plan once per segment; then execute it open-loop
            if mpc_u_plan is None:
                # Solve new plan
                if mode == MODE_MPC_UP:
                    eta_target = float(np.sign(eta) * A)
                    thdot_target = 0.0
                else:
                    eta_target = float(np.sign(eta) * (np.pi/2))
                    # desired speed at pi/2 for sinusoid amplitude A
                    vmag = omega * np.sqrt(max(0.0, A*A - (np.pi/2)**2))
                    # coming down from peak: sign opposite eta
                    thdot_target = float(-np.sign(eta) * vmag)

                # U_opt, X_opt, info = solve_mpc_bridge(
                #     x0=x,
                #     dt_total=Tquarter,
                #     omega=omega,
                #     A=A,
                #     eta_target=eta_target,
                #     thdot_target=thdot_target,
                #     p=p,
                #     kappa=float(cfg["_case_key"]["kappa"]),
                #     G_shape=cfg["_case_key"]["G_shape"],
                #     u_max=u_max,
                #     mpc_cfg=mpc_cfg,
                #     u_prev=float(U[k-1]) if k > 0 else 0.0
                # )
                u_prev = float(U[k - 1]) if k > 0 else 0.0

                U_opt, info = bridge.solve(x, eta_target, thdot_target, u_prev)
                mpc_u_plan = U_opt
                mpc_u_idx = 0
                mpc_info_last = info

                print(f"[MPC] {MODE_NAME[mode]} solved at t={t[k]:.3f}s "
                      f"status={info['status']} obj={info['obj']:.3f} "
                      f"N={info['N']} dt={info['dt']:.4f}")
                
                MPC_OBJ[k] = info["obj"]
                MPC_STATUS[k] = info["status"]

            # apply plan
            if mpc_u_idx < len(mpc_u_plan):
                u = float(mpc_u_plan[mpc_u_idx])
                mpc_u_idx += 1
            else:
                # plan finished -> switch to next mode
                mpc_u_plan = None
                mpc_u_idx = 0
                if mode == MODE_MPC_UP:
                    mode = MODE_MPC_DOWN
                else:
                    mode = MODE_SMC
                    phase_state["psi"] = phase_from_state(theta_dev(x[1]), x[3], A, omega)
                u = 0.0

            MODE[k] = mode

        U[k] = u

        # ---------- integrate plant ----------
        for _ in range(n_sub):
            x = rk4_step(rhs_continuous, x, u, dt_int, p,
                         kappa=float(cfg["_case_key"]["kappa"]),
                         G_shape=cfg["_case_key"]["G_shape"],
                         b_theta_true=lambda theta: b_theta_true(theta, b0_true, b1_true))

        # optional: wrap angles to avoid wind-up
        # x[0] = wrap_center(x[0], 0.0)
        # x[1] = wrap_center(x[1], np.pi)

    base = {
        "t": t,
        "X": X,
        "U": U,
        "omega": np.array([omega]),
        "MODE": MODE,
        "PHREF": PHREF,
        "PHDREF": PHDREF,
        "ETAREF": THREF,
        "ETADOTREF": THDREF,
        "S_SURF": S_SURF
    }
    return base, cfg
