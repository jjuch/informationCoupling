import numpy as np
from control import finite_difference_jacobian
from furuta_model import rk4_step, rhs_continuous, wrap_state_angles

def R_from_sigma_phi(sigma_phi, dt, dim=2):
    """
    Correlated measurement noise covariance induced by differentiation:
      y_phi = phi + eta
      y_phidot = (y_phi[k] - y_phi[k-1]) / dt
    """
    s2 = sigma_phi**2
    if dim == 1:
        return np.array([[s2]], dtype=float)
    elif dim == 2:
        return np.array([[s2, s2/dt],
                        [s2/dt, 2*s2/(dt**2)]], dtype=float)
    else:
        raise ValueError("When construction R, dimensions larger than 2 are not implemented yet.")


def measure_phi(phi_true, sigma_phi, rng):
    return phi_true + rng.normal(0.0, sigma_phi)

def derive_phidot(phi_meas_k, phi_meas_km1, dt):
    return (phi_meas_k - phi_meas_km1) / dt

def h_meas(x):
    """z = [phi, phidot]."""
    return np.array([x[0], x[2]], dtype=float)

def H_meas(_x):
    """Jacobian of h_meas is constant."""
    n = _x.size
    H = np.zeros((2, n), dtype=float)
    H[0, 0] = 1.0
    H[1, 2] = 1.0
    return H


def b_theta_hat_fourier(theta, coeffs, b_min=1e-4):
    """
    Fourier basis M=2:
      coeffs = [c0, c1, s1, c2, s2]
      b_hat(theta) = c0 + c1 cos(theta) + s1 sin(theta) + c2 cos(2theta) + s2 sin(2theta)
    Enforce positivity by lower-bounding at b_min.
    """
    c0, c1, s1, c2, s2 = coeffs
    raw = c0 + c1*np.cos(theta) + s1*np.sin(theta) + c2*np.cos(2*theta) + s2*np.sin(2*theta)
    return np.maximum(b_min, raw)
    
    # # Stable softplus: log(1+exp(x)) computed without overflow
    # softplus = np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0)

    # return b_min + softplus



def grad_b_hat(theta):
    # gradient wrt [c0, c1, s1, c2, s2]
    return np.vstack([
        np.ones_like(theta),
        np.cos(theta),
        np.sin(theta),
        np.cos(2*theta),
        np.sin(2*theta)
    ]).T  # shape (N,5)



def rhs_continuous_aug(x_aug, u, p, kappa=0.0, G_shape="const", **kwargs):
    """
    Augmented continuous dynamics:
      x_aug = [phi, theta, phidot, thetadot, c0, c1, s1, c2, s2]
    Coefficients are modeled as random-walk via process noise (deterministic derivative = 0).
    """
    x_aug = np.asarray(x_aug, dtype=float)
    x_phys = x_aug[:4]
    coeffs = x_aug[4:9]
    theta = x_phys[1]
    b_hat = b_theta_hat_fourier(theta, coeffs)

    xdot_phys = rhs_continuous(x_phys, u, p, kappa=kappa, G_shape=G_shape, b_theta_true=b_hat)

    xdot = np.zeros_like(x_aug)
    xdot[:4] = xdot_phys
    # xdot[4:9] = 0  (random walk handled by Q)
    return xdot


def make_spd(P, jitter_list=(0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)):
    """
    Robustly make P SPD by adding diagonal jitter until Cholesky succeeds.
    """
    P = 0.5 * (P + P.T)
    n = P.shape[0]
    I = np.eye(n)
    for j in jitter_list:
        try:
            np.linalg.cholesky(P + j * I)
            return P + j * I
        except np.linalg.LinAlgError:
            print(f"Failed for {j}")
            continue
    return P + 1e-2 * I


def apply_pseudo_measurement_zero_params(x_upd, P_upd, idx_list, sigma_p=1e-3):
    """
    Softly regularize selected state entries toward zero via a pseudo-measurement.

    We impose: z = 0 ≈ H x, where H selects the states in idx_list.
    This is equivalent to adding a Gaussian prior that those entries are near zero,
    with "measurement noise" sigma_p.

    Parameters
    ----------
    x_upd : (n,) ndarray
        Current updated state (after the real measurement update).
    P_upd : (n,n) ndarray
        Current updated covariance.
    idx_list : list[int]
        Indices of state components to regularize toward zero (e.g. [5,6,8]).
    sigma_p : float
        Std dev of pseudo-measurement. Smaller = stronger pull toward zero.

    Returns
    -------
    x_new : (n,) ndarray
    P_new : (n,n) ndarray
    """
    x_upd = np.asarray(x_upd, dtype=float)
    P_upd = np.asarray(P_upd, dtype=float)
    n = x_upd.size
    m = len(idx_list)

    H = np.zeros((m, n), dtype=float)
    for j, idx in enumerate(idx_list):
        H[j, idx] = 1.0

    z = np.zeros(m, dtype=float)
    z_pred = H @ x_upd
    R = (sigma_p**2) * np.eye(m, dtype=float)

    S = H @ P_upd @ H.T + R
    S = 0.5 * (S + S.T)

    # Invert safely (small m, so inv is OK)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.inv(S + 1e-12*np.eye(m))

    K = P_upd @ H.T @ S_inv
    innov = z - z_pred

    x_new = x_upd + K @ innov

    # Joseph-form covariance update
    I = np.eye(n)
    P_new = (I - K @ H) @ P_upd @ (I - K @ H).T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new





class EKF:
    def __init__(self, x0, P0, Q, R, dt, p, kappa=0.0, G_shape="const", phi_center=0.0, theta_center=np.pi):
        self.phi_center = float(phi_center)
        self.theta_center = float(theta_center)

        self.x = np.asarray(x0, dtype=float).copy()
        # self.x = wrap_state_angles(self.x, phi_center=self.phi_center, theta_center=self.theta_center)

        self.P = np.asarray(P0, dtype=float).copy()
        self.P = 0.5*(self.P + self.P.T)

        self.Q = np.asarray(Q, dtype=float).copy()
        self.Q = 0.5*(self.Q + self.Q.T)

        self.R = np.asarray(R, dtype=float).copy()
        self.R = 0.5*(self.R + self.R.T)

        self.dt = float(dt)
        self.p = p
        self.kappa = float(kappa)
        self.G_shape = G_shape

        self.x_pred = None
        self.P_pred = None

        self.rhs_continuous = rhs_continuous

    def _f_discrete(self, x, u):
        # Discrete propagation via RK4 on continuous dynamics (nominal friction)
        return rk4_step(self.rhs_continuous, x, u, self.dt, self.p,
                        kappa=self.kappa, G_shape=self.G_shape, b_theta_true=None, wrap_angles=False)

    def predict(self, u):
        # state propagation
        self.x_pred = self._f_discrete(self.x, u)

        # discrete-time Jacobian F ≈ ∂f/∂x
        F = finite_difference_jacobian(lambda xx: self._f_discrete(xx, u), self.x)

        self.P_pred = F @ self.P @ F.T + self.Q
        self.P_pred = 0.5*(self.P_pred + self.P_pred.T)
        return self.x_pred, self.P_pred, F

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)

        # --- Measurement model selection based on z dimension ---
        n = len(self.x_pred)

        if z.size == 1:
            # Option A: z = [phi]
            z_pred = np.array([self.x_pred[0]], dtype=float) # phi
            H = np.zeros((1, n), dtype=float)
            H[0, 0] = 1.0

            # self.R must be (1,1) here
            R = np.asarray(self.R, dtype=float)
            if R.shape != (1, 1):
                raise ValueError(f"EKF.update expected self.R shape (1,1) for phi-only update, got {R.shape}")
        elif z.size == 2:
            # Case B: z = [phi, phidot]
            z_pred = h_meas(self.x_pred)
            H = H_meas(self.x_pred)
            
            R = np.asarray(self.R, dtype=float)
            if R.shape != (2, 2):
                raise ValueError(f"EKF.update expected self.R shape (2,2) for phi+phidot update, got {R.shape}")
        else:
            raise ValueError(f"Unsupported measurement dimension {z.size}. Use 1 (phi) or 2 (phi, phidot).")

        # Kalman update (real measurement)
        S = H @ self.P_pred @ H.T + R
        S = 0.5*(S + S.T)
        
        # More stable than inv for small S
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # tiny jitter if needed
            S_inv = np.linalg.inv(S + 1e-12 * np.eye(S.shape[0]))

        K = self.P_pred @ H.T @ S_inv

        innov = z - z_pred
        x_upd = self.x_pred + K @ innov

        # Joseph stabilized covariance update: PSD-preserving
        I = np.eye(len(self.x_pred))
        P_upd = (I - K @ H) @ self.P_pred @ (I - K @ H).T + K @ R @ K.T
        P_upd = make_spd(P_upd)

        # state correction for TE
        dx = x_upd - self.x_pred
    
        self.x, self.P = x_upd, P_upd

        return innov, dx
    

    
    
class EKF_FourierFriction(EKF):
    """
    EKF on augmented state including Fourier friction coefficients (M=2).

    State:
      [phi, theta, phidot, thetadot, c0, c1, s1, c2, s2]
    Measurement:
      z = [phi, phidot]

    Uses finite-difference Jacobians (consistent with your current EKF approach).
    """

    def __init__(self, x0_aug, P0, Q, R, dt, p, kappa=0.0, G_shape="const"):
        super().__init__(x0_aug, P0, Q, R, dt, p, kappa=kappa, G_shape=G_shape)
        self.rhs_continuous = rhs_continuous_aug

    def update(self, z):
        innov, dx_meas = super().update(z)

        # ============================
        # pseudo-measurement regularization
        # ============================
        # Apply only for the augmented friction EKF (state length 9):
        # state = [phi, theta, phidot, thetadot, c0, c1, s1, c2, s2]
        
        # Softly regularize odd/unneeded coefficients toward zero: c1, s1, s2
        # indices: c1=5, s1=6, s2=8
        sigma_p = 1e-3   
        # sigma_p = 1e-2 → very weak (almost no effect)
        # sigma_p = 3e-3 → gentle pull toward zero
        # sigma_p = 1e-3 → strong-ish, usually good starting point
        # sigma_p = 3e-4 → quite strong; can bias if the model actually needs sine terms
        self.x, P = apply_pseudo_measurement_zero_params(
            self.x, self.P,
            idx_list=[5, 6, 8],
            sigma_p=sigma_p
        )
        self.P = make_spd(P)
        # ============================

        # Ift TE/correction series to include the pseudo-measurement effect,
        # compute total correction here:
        dx_total = self.x - self.x_pred

        return innov, dx_meas
