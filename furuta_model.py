import numpy as np
import matplotlib.pyplot as plt

TWOPI = 2.0 * np.pi

def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_center(a, center):
    """
    Wrap angle a to be near 'center', i.e. center + wrap(a-center).
    Useful to keep theta near pi (down equilibrium).
    """
    return center + wrap_angle(a - center)



def wrap_state_angles(x, phi_center=0.0, theta_center=np.pi):
    """
    Wrap phi around 0 and theta around pi by default.
    """
    x = np.asarray(x, dtype=float).copy()
    x[0] = wrap_center(x[0], phi_center)
    x[1] = wrap_center(x[1], theta_center)
    return x


def G_kappa(q, kappa: float, shape: str = "const"):
    """Skew-symmetric gyroscopic coupling matrix."""
    phi, theta = q
    if shape == "const":
        s = 1.0
    elif shape == "cos":
        s = np.cos(theta)
    elif shape == "1+theta2":
        s = 1.0 + theta**2
    elif shape == "sin2":
        eps = 0.02
        beta = 5
        sigma = lambda z: 1/(1 + np.exp(-z))
        s = np.sin(theta)**2 * (eps + (1 - eps) * sigma(beta*np.sin(theta)))
    elif shape == "sin_1plusCos":
        s = np.sin(theta)*(1 + np.cos(theta))
    else:
        raise ValueError("Unknown shape")
    return kappa * s * np.array([[0.0, 1.0], [-1.0, 0.0]])


def b_theta_true(theta, b0_true, b1_true):
    return b0_true + b1_true * (np.cos(theta) ** 2)

def furuta_M_C_g(q, qd, p):
    """
    Gäfvert (1998) Furuta model in matrix form (Eq. 19-20):
      D(q) qdd + C(q,qd) qd + k(q) = tau
    with
      D = [[alpha + beta sin^2(theta), gamma cos(theta)],
           [gamma cos(theta),         beta]]
      k = [0, delta sin(theta)]^T
    and a C-factorization chosen so that C(q,qd) qd reproduces Eq. (18)'s velocity terms. Note that theta=0 is upright and theta=pi is downward (stable).
    [1](https://www.control.lth.se/fileadmin/control/Education/EngineeringProgram/FRTN10/2020/Gafvert1998.pdf)
    """
    phi, theta = q
    phidot, thetadot = qd

    s = np.sin(theta)
    c = np.cos(theta)

    alpha = p.alpha
    beta  = p.beta
    gamma = p.gamma
    delta = p.delta

    # Inertia matrix M(q)
    M = np.array([
        [alpha + beta * s*s,  gamma * c],
        [gamma * c,           beta]
    ])

    # Coriolis/centripetal matrix C(q,qd)
    C = np.array([
        [ beta * c * s * thetadot,   beta * c * s * phidot - gamma * s * thetadot ],
        [ -beta * c * s * phidot,    0.0 ]
    ])
    
    # Gravity/potential term k(q)
    gvec = np.array([0.0, -delta * s])

    return M, C, gvec


def rhs_continuous(x, u, p, kappa=0.0, G_shape="const", b_theta_true=None):
    """
    Continuous-time dynamics for x=[phi, theta, phidot, thetadot].
    Implements:
        M(q) qdd + (C(q,qd)+G_kappa(q)) qd + g(q) + tau_f = [u, 0]^T
    """
    x = np.asarray(x, dtype=float)

    # Wrap angles for numerical robustness around the intended equilibrium
    # xw = wrap_state_angles(x, phi_center=0.0, theta_center=np.pi)
    # phi, theta, phidot, thetadot = xw
    phi, theta, phidot, thetadot = x


    q = np.array([phi, theta])
    qd = np.array([phidot, thetadot])

    M, C, gvec = furuta_M_C_g(q, qd, p)
    G = G_kappa(q, kappa, shape=G_shape)

    if b_theta_true is None:
        b_theta = p.b_theta_nom
    elif callable(b_theta_true): 
        b_theta = float(b_theta_true(theta))
    else:
        b_theta = float(b_theta_true)

    tau_f = np.array([p.b_phi * phidot, b_theta * thetadot])

    # input torque only on phi
    tau = np.array([u, 0.0])

    qdd = np.linalg.solve(M, tau - (C + G) @ qd - gvec - tau_f)
    return np.array([phidot, thetadot, qdd[0], qdd[1]], dtype=float)

def rk4_step(f, x, u, dt, *args, wrap_angles=False, **kwargs):
    k1 = f(x, u, *args, **kwargs)
    k2 = f(x + 0.5*dt*k1, u, *args, **kwargs)
    k3 = f(x + 0.5*dt*k2, u, *args, **kwargs)
    k4 = f(x + dt*k3, u, *args, **kwargs)
    
    x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    if wrap_angles:
        x_next = wrap_state_angles(x_next, phi_center=0.0, theta_center=np.pi)
    return x_next


def simulate_free_response(p, x0, dt=0.01, T=10.0, kappa=0.0, G_shape="const", b_theta_true=None, title=None):
    """
    Open-loop free response with u=0. Plots phi, theta, phidot, thetadot.
    Useful sanity check before closing the loop.
    """
    N = int(T/dt)
    t = np.arange(N) * dt
    x = np.asarray(x0, dtype=float).copy()
    X = np.zeros((N, 4), dtype=float)

    for k in range(N):
        X[k] = x
        x = rk4_step(rhs_continuous, x, 0.0, dt, p, kappa=kappa, G_shape=G_shape, b_theta_true=b_theta_true)

    fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    labels = ["phi [rad]", "theta [rad]", "phidot [rad/s]", "thetadot [rad/s]"]
    for i in range(4):
        axs[i].plot(t, X[:, i], linewidth=1.2)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
    axs[-1].set_xlabel("time [s]")
    fig.suptitle(title if title else f"Free response (u=0), kappa={kappa:+.2f}", y=0.98)
    fig.tight_layout()
    plt.show()
    return t, X
