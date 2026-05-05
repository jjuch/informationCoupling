import numpy as np
import matplotlib.pyplot as plt
from furuta_model import wrap_center

def theta_dev(theta):
    return wrap_center(theta, np.pi) - np.pi

def _autoscale_y_on_visible_x(ax, margin=0.08):
    """
    Autoscale y-limits for `ax` based on points visible in current x-limits.
    """
    x0, x1 = ax.get_xlim()
    ymins, ymaxs = [], []

    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if xd.size == 0 or yd.size == 0:
            continue

        m = (xd >= x0) & (xd <= x1) & np.isfinite(yd)
        if not np.any(m):
            continue
        yv = yd[m]
        ymins.append(np.min(yv))
        ymaxs.append(np.max(yv))

    if not ymins:
        return  # nothing visible to scale on

    ymin, ymax = float(np.min(ymins)), float(np.max(ymaxs))
    if np.isclose(ymin, ymax):
        # avoid singular y-lims
        pad = 1.0 if np.isclose(ymin, 0.0) else 0.05 * abs(ymin)
        ymin -= pad
        ymax += pad
    else:
        pad = margin * (ymax - ymin)
        ymin -= pad
        ymax += pad

    ax.set_ylim(ymin, ymax)


def plot_states_with_refs(base, cfg, meta_title="Scenario F"):
    t = base["t"]
    X = base["X"]
    Z = base["Z"]
    U = base["U"]
    V = base["V"]
    omega = float(np.nanmedian(base["omega"]))
    A = float(cfg["theta"]["A"])

    # two convenient refs:
    # 1) nominal VHC shape vs phi: theta = pi + A sin(phi)
    th_ref_phi = np.pi + A * np.sin(X[:, 0])
    # 2) time ref for visualization only: pi + A sin(omega t + phase0)
    phase0 = float(np.arcsin(np.clip(theta_dev(X[0, 1]) / max(1e-9, A), -1.0, 1.0)))
    th_ref_t = np.pi + A * np.sin(omega * t + phase0)

    fig, axes = plt.subplots(6, 1, figsize=(11, 11), sharex=True, constrained_layout=True)

    # 1) u
    axes[0].plot(t, U, lw=1.0)
    axes[0].set_ylabel("u [Nm]")
    axes[0].grid(True, alpha=0.3)

    # 2) V
    axes[1].plot(t, V, lw=1.0)
    axes[1].set_ylabel("v (= sddot)")
    axes[1].grid(True, alpha=0.3)

    # 3) phi / phidot
    axes[2].plot(t, X[:, 0], lw=1.2, label="phi")
    axes[2].plot(t, X[:, 2], lw=1.0, ls="--", label="phidot")
    axes[2].legend()
    axes[2].set_ylabel("phi / phidot")
    axes[2].grid(True, alpha=0.3)

    
    axes[3].plot(t, theta_dev(X[:, 1]), lw=1.2, label="theta-pi")
    axes[3].plot(t, theta_dev(th_ref_phi), lw=1.0, ls="--", label="ref (via phi)")
    axes[3].plot(t, theta_dev(th_ref_t), lw=0.9, ls=":", label="ref (via time)")
    axes[3].axhline(+A, ls=":", alpha=0.4)
    axes[3].axhline(-A, ls=":", alpha=0.4)
    axes[3].legend()
    axes[3].set_ylabel("theta-pi [rad]")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t, Z[:, 4], lw=1.2, label="s")
    axes[4].plot(t, Z[:, 5], lw=1.2, label="sdot")
    axes[4].legend()
    axes[4].set_ylabel("s, sdot")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(t, base["E"], lw=1.2, label="e")
    axes[5].plot(t, base["ED"], lw=1.2, label="ed")
    axes[5].legend()
    axes[5].set_xlabel("time [s]")
    axes[5].set_ylabel("e, ed")
    axes[5].grid(True, alpha=0.3)

    fig.suptitle(meta_title, y=0.99)

    # ---- Initial y autoscale (full range) ----
    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    # ---- Dynamic y autoscale when x-limits change (zoom/pan) ----
    _in_callback = {"busy": False}
    # ---- Dynamic y autoscale when x-limits change (zoom/pan) ----
    _in_callback = {"busy": False}

    def _on_xlim_changed(ax):
        if _in_callback["busy"]:
            return
        _in_callback["busy"] = True
        try:
            # Shared x already syncs, but we rescale y on all axes
            for a in axes:
                _autoscale_y_on_visible_x(a, margin=0.08)
            fig.canvas.draw_idle()
        finally:
            _in_callback["busy"] = False

    # Connect once (xlim changes propagate due to sharex)
    axes[0].callbacks.connect("xlim_changed", _on_xlim_changed)

    # Trigger once so y-lims are nice at start
    _on_xlim_changed(axes[0])

    plt.show()


def plot_Snonlin_vs_theta_compare(maps0, maps1, label0, label1):
    th = maps0["theta_centers"]
    plt.figure(figsize=(10,4.6))
    plt.plot(th, maps0["S_nonlin_med"], lw=2.0, color="black", label=label0)
    plt.plot(th, maps1["S_nonlin_med"], lw=1.8, label=label1)
    plt.xlabel("theta [rad] (wrapped)")
    plt.ylabel("S_nonlin median")
    plt.title("Nonlinear structural coupling vs theta (stabilized trajectories)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()