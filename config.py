from dataclasses import dataclass, field
import numpy as np

@dataclass
class FurutaParams:
    """
    Furuta model in Gäfvert's (1998) alpha-beta-gamma-delta parameterization
    (Eq. 17-20). Defaults use the example 'Real pendulum model parameters'
    from Table 2. [1](https://www.control.lth.se/fileadmin/control/Education/EngineeringProgram/FRTN10/2020/Gafvert1998.pdf)
    """
    # Gäfvert model parameters
    alpha: float = 0.0033472   # [kg*m^2]
    beta:  float = 0.0038852   # [kg*m^2]
    gamma: float = 0.0024879   # [kg*m^2]
    delta: float = 0.097625    # [kg^2*m^2/s^2]

    # gravity
    g: float = 9.81

    # viscous friction (nominal)
    b_phi: float = 0.02           # actuated joint viscous friction
    b_theta_nom: float = 0.01     # passive joint viscous friction (nominal)
    # --- True directional friction (plant): b0 + b1 cos^2(theta) ---
    b0_nom: float = b_theta_nom
    b1_nom: float = 2.0 * b_theta_nom

    # actuator saturation
    u_max: float = 5.0            # N·m (example)

@dataclass
class ExperimentConfig:
    dt: float = 0.005
    T: float = 20.0
    sigma_phi: float = 1e-3
    friction_uncertainty: float = 0.20   # ±20%

    # Monte Carlo
    mc_trials_debug: int = 5 #50
    mc_trials_full: int = 200
    seed: int = 1

    # Covariance-collapse metric
    eps_logdet_drop: float = 4.0

    # TE settings
    te_lag: int = 3               # VAR order for TE (fixed first)
    te_start_time: float = 0.10    # ignore part for TE in seconds

    # initial conditions [phi,theta,phidot,thetadot]
    x0_true: np.ndarray = field(default_factory=lambda: np.array([0.0, np.pi/2, 0.0, 0.0], dtype=float))
    x0_hat:  np.ndarray = field(default_factory=lambda: np.array([0.0, np.pi, 0.0, 0.0], dtype=float))
    P0: np.ndarray = field(default_factory=lambda: np.diag([0.01**2, 0.3**2, 0.2**2, 1.0**2]).astype(float))

    # setpoint
    x_ref: np.ndarray = field(default_factory=lambda: np.array([0.0, np.pi, 0.0, 0.0], dtype=float))

    # LQR
    Q_lqr: np.ndarray = field(default_factory=lambda: np.diag([1.0, 2.0, 5.0, 5.0]).astype(float))
    R_lqr: np.ndarray = field(default_factory=lambda: np.array([[100.0]]).astype(float))

    # process noise (tune later)
    Q: np.ndarray = field(default_factory=lambda: np.diag([1e-7, 1e-5, 1e-5, 1e-2]).astype(float))
    nis_hi: float =  20000.0   # strong inconsistency trigger (phi-only => p=1)
    nis_warn: float = 175000.0   # mild trigger
    Pcc_inflate_strong: float = 50.0
    Pcc_inflate_mild: float = 10.0
    # Adaptive Qcc params
    q_min: float = 1e-7
    q_max: float = 1e-5
    q_gamma:float = 2.0
    q_tau: float = -25.0

    # estimater update period
    update_period: float = 0.05 # 50 ms

    # kappa sweep
    # kappas: tuple = (-0.10, 0.10)
    # kappas: tuple = (-0.2, -0.10, -0.05, 0.05, 0.10, 0.2)
    # kappas: tuple = (0.2/1.2, )
    kappas: tuple = (0.0, 0.09, 0.2, 0.25, 0.4)
