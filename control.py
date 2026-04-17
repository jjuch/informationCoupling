import numpy as np
import scipy as sp

from furuta_model import rhs_continuous

def saturate(u, u_max):
    return float(np.clip(u, -u_max, u_max))

def lqr(A, B, Q, R):
    """Continuous-time LQR gain K (u = -Kx)."""
    P = sp.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ P)
    return K

def finite_difference_jacobian(f, x, eps=1e-6):
    """
    Numerical Jacobian of vector function f(x) w.r.t x.
    f: R^n -> R^m, returns shape (m,n).
    """
    x = np.asarray(x)
    y0 = np.asarray(f(x))
    m = y0.size
    n = x.size
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (np.asarray(f(x + dx)) - y0) / eps
    return J


def state_error(x, x_ref, theta_center=np.pi):
    """
    Compute delta x = x - x_ref with angle wrapping for phi and theta.
    Assumes x = [phi, theta, phidot, thetadot].
    """
    dx = np.asarray(x, dtype=float) - np.asarray(x_ref, dtype=float)
    # wrap angular errors
    dx[0] = (dx[0] + np.pi) % (2*np.pi) - np.pi
    dx[1] = (dx[1] + np.pi) % (2*np.pi) - np.pi  # because x_ref[1] = pi, this is wrap(theta - pi)
    return dx



def linearize_rhs(p, x0, u0, kappa=0.0, G_shape="const", eps=1e-6):
    """
    Linearize continuous-time dynamics xdot = f(x,u) about (x0,u0):
      A = df/dx, B = df/du
    """
    x0 = np.asarray(x0, dtype=float)

    def fx(x):
        return rhs_continuous(x, u0, p, kappa=kappa, G_shape=G_shape, b_theta_true=None)

    A = finite_difference_jacobian(fx, x0, eps=eps)

    # B via finite difference on u
    f0 = rhs_continuous(x0, u0, p, kappa=kappa, G_shape=G_shape, b_theta_true=None)
    f1 = rhs_continuous(x0, u0 + eps, p, kappa=kappa, G_shape=G_shape, b_theta_true=None)
    B = ((f1 - f0) / eps).reshape(-1, 1)

    return A, B
