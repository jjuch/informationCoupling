import numpy as np
import casadi as ca

from furuta_model import rhs_continuous, rk4_step, wrap_center, b_theta_true

def theta_dev(theta):
    return wrap_center(theta, np.pi) - np.pi


def _wrap_eta_casadi(theta):
    # wrap (theta - pi) into (-pi, pi] using atan2(sin,cos)
    return ca.atan2(ca.sin(theta - np.pi), ca.cos(theta - np.pi))


class StepMapCallback(ca.Callback):
    """
    casADi Callback: x_next =  F(x, u)
    Uses RK4 substepped numeric integration under the hood.
    """
    def __init__(self, name, dt, n_sub, p, kappa, G_shape):
        super().__init__()
        self.dt = float(dt)
        self.n_sub = int(n_sub)
        self.p = p
        self.kappa = float(kappa)
        self.G_shape = G_shape
        self.construct(name, {"enable_fd": True})  # finite-diff Jacobians
        
    def get_n_in(self):  return 2
    def get_n_out(self): return 1
    def get_name_in(self, i):  return ["x", "u"][i]
    def get_name_out(self, i): return ["x_next"][i]
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(4,1) if i == 0 else ca.Sparsity.dense(1,1)
    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(4,1)

    def eval(self, arg):
        x = np.array(arg[0]).reshape(4,)
        u = float(arg[1])
        dt_int = self.dt / float(self.n_sub)
        b0_true, b1_true = self.p.b0_nom, self.p.b1_nom
        for _ in range(self.n_sub):
            x = rk4_step(rhs_continuous, x, u, dt_int, self.p,
                         kappa=self.kappa, G_shape=self.G_shape,
                         b_theta_true=lambda theta: b_theta_true(theta, b0_true, b1_true))
            if not np.all(np.isfinite(x)):
                x[:] = np.nan
                break
        return [ca.DM(x).reshape((4,1))]
    
class MPCBridgeSolver:
    """
    Reusable single-shooting MPC bridge:
      variables: U[0..N-1]
      parameters: x0 (4), eta_target (1), thdot_target (1), u_prev (1)
    """
    def __init__(self, p, kappa, G_shape,
                 Tquarter, u_max, mpc_cfg):
        
        self.p = p
        self.kappa = float(kappa)
        self.G_shape = G_shape
        self.u_max = float(u_max)

        self.N = int(mpc_cfg["N"])
        self.dt = float(Tquarter) / float(self.N)
        self.n_sub = int(mpc_cfg.get("n_sub", 1))

        self.w_eta = float(mpc_cfg["w_eta_term"])
        self.w_thd = float(mpc_cfg["w_thdot_term"])
        self.w_u = float(mpc_cfg["w_u"])
        self.w_du = float(mpc_cfg["w_du"])

        self.solver_name = str(mpc_cfg.get("solver", "ipopt"))
        self.max_iter = int(mpc_cfg.get("max_iter", 80))
        self.max_cpu_time = float(mpc_cfg.get("ipopt_max_cpu_time", 1.5))
        self.print_level = int(mpc_cfg.get("ipopt_print_level", 0))
        self.print_time = bool(mpc_cfg.get("ipopt_print_time", False))
        self.acceptable_tol = float(mpc_cfg.get("ipopt_acceptable_tol", 1e-2))
        self.tol = float(mpc_cfg.get("ipopt_tol", 1e-3))

        # step map
        self.F = StepMapCallback(
            "Fstep",
            dt=self.dt,
            n_sub=self.n_sub,
            p=p,
            kappa=self.kappa,
            G_shape=self.G_shape
        )

        # sanity check
        out_sp = self.F.sparsity_out(0)
        if (out_sp.size1() != 4) or (out_sp.size2() != 1):
            raise RuntimeError(
                f"StepMapCallback output sparsity is {out_sp.size1()}x{out_sp.size2()}, expected 4x1. "
                "Callback is not configured correctly."
            )

        # build solver once
        self._build()

    
    def _build(self):
        N = self.N

        U = ca.MX.sym("U", N, 1)
        x0 = ca.MX.sym("x0", 4, 1)
        eta_t = ca.MX.sym("eta_t", 1, 1)
        thd_t = ca.MX.sym("thd_t", 1, 1)
        u_prev = ca.MX.sym("u_prev", 1, 1)

        # rollout
        x = ca.reshape(x0, 4, 1)

        for k in range(N):
            uk = ca.reshape(U[k], 1, 1)
            x = self.F(x, uk)#[0]
            x = ca.reshape(x, 4, 1)   # enforce shape every step

        thetaN = x[1, 0]
        thdotN = x[3, 0]

        etaN = _wrap_eta_casadi(thetaN)

        # objective
        J = self.w_eta * (etaN - eta_t)**2 + self.w_thd * (thdotN - thd_t)**2
        J += self.w_u * ca.sumsqr(U)
        J += self.w_du * (U[0] - u_prev)**2
        for k in range(1, N):
            J += self.w_du * (U[k] - U[k-1])**2

        nlp = {"x": U, "p": ca.vertcat(x0, eta_t, thd_t, u_prev), "f": J}

        if self.solver_name == "ipopt":
            opts = {
                "print_time": self.print_time,
                "ipopt.print_level": self.print_level,
                "ipopt.max_iter": self.max_iter,
                "ipopt.max_cpu_time": self.max_cpu_time,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.tol": 1e-3
            }
        else:
            # sqpmethod (often faster for FD-heavy problems)
            opts = {
                "print_header": False,
                "print_iteration": False,
                "max_iter": self.max_iter
            }

        self.solver = ca.nlpsol("solver", self.solver_name, nlp, opts)

        self.lbx = [-self.u_max] * self.N
        self.ubx = [ self.u_max] * self.N
        self.U0 = np.zeros((self.N,1))

        

    def solve(self, x0_np, eta_target, thdot_target, u_prev):
        pvec = np.vstack([
            np.asarray(x0_np, float).reshape((4,1)),
            np.array([[eta_target]], float),
            np.array([[thdot_target]], float),
            np.array([[u_prev]], float)
        ])

        sol = self.solver(x0=self.U0, lbx=self.lbx, ubx=self.ubx, p=pvec)
        U_opt = np.array(sol["x"]).reshape((self.N,))
        obj = float(sol["f"])

        self.U0 = U_opt.reshape((self.N,1))  # warm-start next solve
        
        stats = self.solver.stats()
        info = {
            "status": stats.get("return_status", ""),
            "success": bool(stats.get("success", False)),
            "obj": obj,
            "N": self.N,
            "dt": self.dt
        }
        return U_opt, info






def solve_mpc_bridge(x0, dt_total, omega, A, eta_target, thdot_target,
                     p, kappa, G_shape, u_max,
                     mpc_cfg,
                     u_prev=0.0):
    """
    Solve an MPC bridge over dt_total with N steps, to reach eta_target and thdot_target.
    """
    N = int(mpc_cfg["N"])
    dt = float(dt_total) / float(N)
    n_sub = 2  # keep small; plant substep already fine

    # step map
    F = StepMapCallback("Fstep", dt=dt, n_sub=n_sub, p=p, kappa=kappa,
                        G_shape=G_shape)

    # decision variables
    U = ca.MX.sym("U", N, 1)
    X = ca.MX.sym("X", 4, N+1)

    # constraints list
    g = []
    g_lb = []
    g_ub = []

    # initial state constraint
    g.append(X[:,0] - ca.DM(x0))
    g_lb += [0,0,0,0]
    g_ub += [0,0,0,0]

    # dynamics constraints
    for k in range(N):
        xk = X[:,k]
        uk = U[k]
        xnext = F(ca.reshape(xk,(4,1)), ca.reshape(uk,(1,1)))[0]
        g.append(X[:,k+1] - xnext)
        g_lb += [0,0,0,0]
        g_ub += [0,0,0,0]

    # objective
    w_eta = float(mpc_cfg["w_eta_term"])
    w_thd = float(mpc_cfg["w_thdot_term"])
    w_u = float(mpc_cfg["w_u"])
    w_du = float(mpc_cfg["w_du"])

    # terminal eta and thetadot
    thetaN = X[1, N]
    thdotN = X[3, N]
    etaN = ca.atan2(ca.sin(thetaN - np.pi), ca.cos(thetaN - np.pi))  # wrap in CasADi

    J = w_eta*(etaN - eta_target)**2 + w_thd*(thdotN - thdot_target)**2
    J += w_u * ca.sumsqr(U)

    # smoothness penalty
    du0 = U[0] - u_prev
    J += w_du * du0**2
    for k in range(1, N):
        J += w_du * (U[k] - U[k-1])**2

    # bounds on U
    lbx = []
    ubx = []

    # X bounds (unbounded)
    for _ in range((N+1)*4):
        lbx.append(-ca.inf); ubx.append(ca.inf)

    # U bounds
    for _ in range(N):
        lbx.append(-u_max); ubx.append(u_max)

    # pack variables
    Z = ca.vertcat(ca.reshape(X, (4*(N+1), 1)), ca.reshape(U, (N,1)))

    nlp = {"x": Z, "f": J, "g": ca.vertcat(*g)}
    opts = {
        "ipopt.print_level": int(mpc_cfg.get("ipopt_print_level", 0)),
        "print_time": mpc_cfg.get("ipopt_print_time", False),
        "ipopt.max_iter": int(mpc_cfg.get("max_iter", 200)),
        "ipopt.max_cpu_time": float(mpc_cfg.get("ipopt_max_cpu_time", 2.0)),
        "ipopt.acceptable_tol": float(mpc_cfg.get("ipopt_acceptable_tol", 1e-2)),
        "ipopt.tol": float(mpc_cfg.get("ipopt_tol", 1e-3))
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # initial guess
    z0 = np.zeros((4*(N+1)+N, 1))
    z0[:4] = np.array(x0).reshape((4,1))
    # warm-start U with constant u_prev
    z0[4*(N+1):] = float(u_prev)

    sol = solver(x0=z0, lbx=lbx, ubx=ubx, lbg=g_lb, ubg=g_ub)
    z_opt = np.array(sol["x"]).reshape((-1,))
    U_opt = z_opt[4*(N+1):]
    X_opt = z_opt[:4*(N+1)].reshape((4, N+1), order="F")

    info = {
        "status": solver.stats()["return_status"],
        "obj": float(sol["f"]),
        "dt": dt,
        "N": N
    }
    return U_opt, X_opt, info