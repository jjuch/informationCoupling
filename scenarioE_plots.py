
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


def _segments_from_mask(t, mask):
    """
    Convert boolean mask over time samples into contiguous [t_start, t_end] segments.
    """
    t = np.asarray(t, float)
    mask = np.asarray(mask, bool)
    if t.size == 0 or mask.size == 0:
        return []
    mask = mask[:t.size]

    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    # find contiguous runs
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    segs = []
    for a, b in zip(starts, ends):
        segs.append((t[a], t[b]))
    return segs


def _theta_piecewise_refs_from_sing(t, theta, thetadot, sing_mask, omega, amp, phase):
    """
    Reconstruct the piecewise theta references:
      - outside singular zone: sinusoid
      - inside singular zone: linear continuation from entry slope/value

    Returns:
      th_ref_used (absolute theta ref),
      thd_ref_used (thetadot ref)
    """
    t = np.asarray(t, float)
    theta = np.asarray(theta, float)
    thetadot = np.asarray(thetadot, float)
    sing_mask = np.asarray(sing_mask, bool)

    # Nominal sinusoidal refs
    th_ref_nom = np.pi + amp * np.sin(omega * t + phase)
    thd_ref_nom = amp * omega * np.cos(omega * t + phase)

    # Start with nominal everywhere
    th_ref_used = th_ref_nom.copy()
    thd_ref_used = thd_ref_nom.copy()

    # Detect entry indices (rising edges)
    in_zone = sing_mask
    if in_zone.size == 0:
        return th_ref_used, thd_ref_used

    entry_idx = np.where(in_zone & ~np.r_[False, in_zone[:-1]])[0]
    if entry_idx.size == 0:
        return th_ref_used, thd_ref_used

    # For each contiguous segment, build linear ref inside
    for idx0 in entry_idx:
        # find segment end

        j = idx0
        while j + 1 < len(in_zone) and in_zone[j + 1]:
            j += 1

        t0 = t[idx0]
        eta0 = wrap_center(theta[idx0] - np.pi, np.pi)          # wrapped theta-pi at entry
        thd0 = float(thetadot[idx0])                  # slope at entry

        seg_t = t[idx0:j+1]
        eta_ref = eta0 + thd0 * (seg_t - t0)

        th_ref_used[idx0:j+1] = np.pi + eta_ref
        thd_ref_used[idx0:j+1] = thd0

    return th_ref_used, thd_ref_used




def plot_states_with_refs(base, cfg, meta_title="Scenario E"):
    
    """
    5 stacked subplots:
      1) u
      2) phi and phi_ref
      3) theta-pi and theta_ref-pi
      4) phidot and phidot_ref
      5) thetadot and thetadot_ref

      - Shared x-axis (zoom/pan sync).
      - Y-axis autoscale per subplot to the visible time window.
      - Shaded regions where singular zone is active (base['SING']).
    """

    t = base["t"]
    X = base["X"]
    U = base["U"]
    PHREF = base["PHREF"]
    PHDREF = base["PHDREF"]
    omega = float(base["omega"][0])
 
    # singular zone mask (optional)
    sing = base.get("SING", None)
    if sing is not None:
        sing_mask = np.asarray(sing, float)[:t.size] > 0.5
        sing_segments = _segments_from_mask(t, sing_mask)
    else:
        sing_segments = []

    amp = float(cfg["theta_ref"]["amp"])
    phase = float(cfg["theta_ref"]["phase"])

    th_ref = np.pi + amp * np.sin(omega*t + phase)
    thd_ref = amp * omega * np.cos(omega*t + phase)
    
    # singular mask (if available)
    sing = base.get("SING", None)
    if sing is not None:
        sing_mask = (np.asarray(sing, float)[:t.size] > 0.5)
    else:
        sing_mask = np.zeros_like(t, dtype=bool)

    # reconstruct piecewise refs (linear in singular zone)
    th_ref_used, thd_ref_used = _theta_piecewise_refs_from_sing(
        t=t,
        theta=X[:, 1],
        thetadot=X[:, 3],
        sing_mask=sing_mask,
        omega=omega,
        amp=amp,
        phase=phase
    )



    fig, axes = plt.subplots(
        5, 1, figsize=(11, 10),
        sharex=True,
        constrained_layout=True
    )

    # ---- helper: shade singular segments on an axis ----
    def shade(ax):
        for (a, b) in sing_segments:
            ax.axvspan(a, b, color="gray", alpha=0.15, lw=0)

    # 1) u
    axes[0].plot(t, U, lw=1.0)
    axes[0].set_ylabel("u [Nm]")
    axes[0].grid(True, alpha=0.3)
    shade(axes[0])

    # 2) phi
    axes[1].plot(t, X[:, 0], lw=1.2, label="phi")
    axes[1].plot(t, PHREF, lw=1.0, ls="--", label="phi_ref")
    axes[1].set_ylabel("phi [rad]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    shade(axes[1])

    # 3) theta-pi
    axes[2].plot(t, theta_dev(X[:, 1]), lw=1.2, label="theta-pi")
    axes[2].plot(t, theta_dev(th_ref), lw=1.0, ls="--", alpha=0.45, label="theta_ref-pi")
    axes[2].plot(t, theta_dev(th_ref_used), lw=1.2, ls="--", label="theta_ref_used (piecewise)")
    axes[2].axhline(+amp, ls=":", alpha=0.5)
    axes[2].axhline(-amp, ls=":", alpha=0.5)
    axes[2].set_ylabel("theta-pi [rad]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    shade(axes[2])


    # 4) phidot
    axes[3].plot(t, X[:, 2], lw=1.2, label="phidot")
    axes[3].plot(t, PHDREF, lw=1.0, ls="--", label="phidot_ref")
    axes[3].set_ylabel("phidot [rad/s]")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    shade(axes[3])

    # 5) thetadot
    axes[4].plot(t, X[:, 3], lw=1.2, label="thetadot")
    axes[4].plot(t, thd_ref, lw=1.0, ls="--", alpha=0.45, label="thetadot_ref")
    axes[4].plot(t, thd_ref_used, lw=1.2, ls="--", label="thetadot_ref_used (piecewise)")
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("thetadot [rad/s]")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    shade(axes[4])

    fig.suptitle(meta_title, y=1.02)

    # ---- Initial y autoscale (full range) ----
    for ax in axes:
        ax.relim()
        ax.autoscale_view()

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
