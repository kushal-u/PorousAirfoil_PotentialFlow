"""
plotter.py

Simple plotting utilities for the fixed porous-model runner.

Outputs only:
1. Airfoil with porous network
2. Cp distribution with XFOIL comparison
3. AoA sweep: CL, CD, CM vs AoA
4. Velocity contour comparison
5. Pressure contour comparison
6. Difference contour
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sorted_side_ids(geom, side: str) -> np.ndarray:
    ids = geom.upper_panel_ids if side == "upper" else geom.lower_panel_ids
    ids = np.asarray(ids, dtype=int)
    return ids[np.argsort(geom.XC[ids])]


# =============================================================================
# AIRFOIL WITH POROUS NETWORK
# =============================================================================
def plot_airfoil_with_porous_network(
    geom,
    network,
    coupled_result,
    output_dir: str | Path,
    title: str = "Airfoil with Porous Network",
) -> None:
    """
    Plot the airfoil and all independent porous passages.

    This works with the IndependentPassageNetwork used in run_porous_models.py:
        network.passages
        passage.pore1
        passage.pore2
        passage.pore_surface_point(...)
    """
    out = _prepare_output_dir(output_dir)
    save_path = out / "airfoil_with_porous_network.png"

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill(
        geom.XB / geom.chord,
        geom.YB / geom.chord,
        color="lightgray",
        alpha=0.35,
        label="Airfoil interior",
    )
    ax.plot(
        geom.XB / geom.chord,
        geom.YB / geom.chord,
        "k-",
        lw=1.4,
        label="Airfoil surface",
    )

    for i, passage in enumerate(network.passages):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        x1 = sp1.x / geom.chord
        y1 = sp1.y / geom.chord
        x2 = sp2.x / geom.chord
        y2 = sp2.y / geom.chord

        state = coupled_result.passage_states[i]
        q = float(state.Q)

        line = ax.plot(
            [x1, x2],
            [y1, y2],
            "-",
            lw=1.4,
            alpha=0.95,
            label=passage.name,
        )[0]

        color = line.get_color()

        # Pore markers on surface
        ax.plot(x1, y1, "o", ms=5, color=color)
        ax.plot(x2, y2, "s", ms=5, color=color)

        # Flow direction arrow
        if abs(q) > 1e-30:
            if q >= 0.0:
                xa, ya, xb, yb = x1, y1, x2, y2
            else:
                xa, ya, xb, yb = x2, y2, x1, y1

            dx = xb - xa
            dy = yb - ya

            ax.annotate(
                "",
                xy=(xa + 0.62 * dx, ya + 0.62 * dy),
                xytext=(xa + 0.38 * dx, ya + 0.38 * dy),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color),
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if len(network.passages) <= 18:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot Saved] {save_path}")


# =============================================================================
# CP DISTRIBUTION WITH XFOIL
# =============================================================================
def plot_cp_distribution_with_xfoil(
    geom,
    porous_result,
    solid_result,
    output_dir: str | Path,
    xfoil_cp_data: dict | None = None,
) -> None:
    """
    Surface Cp comparison:
        - porous panel method
        - solid panel method
        - XFOIL solid airfoil, when available
    """
    out = _prepare_output_dir(output_dir)
    save_path = out / "cp_distribution_with_xfoil.png"

    upper = _sorted_side_ids(geom, "upper")
    lower = _sorted_side_ids(geom, "lower")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        geom.XC[upper] / geom.chord,
        porous_result.Cp[upper],
        "ro-",
        ms=3,
        lw=1.0,
        label="Porous panel upper",
    )
    ax.plot(
        geom.XC[lower] / geom.chord,
        porous_result.Cp[lower],
        "bo-",
        ms=3,
        lw=1.0,
        label="Porous panel lower",
    )

    ax.plot(
        geom.XC[upper] / geom.chord,
        solid_result.Cp[upper],
        "k--",
        lw=1.3,
        label="Solid panel upper",
    )
    ax.plot(
        geom.XC[lower] / geom.chord,
        solid_result.Cp[lower],
        "g--",
        lw=1.3,
        label="Solid panel lower",
    )

    if xfoil_cp_data is not None:
        if "x_upper" in xfoil_cp_data and "cp_upper" in xfoil_cp_data:
            ax.plot(
                xfoil_cp_data["x_upper"],
                xfoil_cp_data["cp_upper"],
                "m-.",
                lw=1.2,
                label="XFOIL upper",
            )

        if "x_lower" in xfoil_cp_data and "cp_lower" in xfoil_cp_data:
            ax.plot(
                xfoil_cp_data["x_lower"],
                xfoil_cp_data["cp_lower"],
                "c-.",
                lw=1.2,
                label="XFOIL lower",
            )

    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title("Cp Distribution: Porous Panel vs Solid Panel vs XFOIL")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot Saved] {save_path}")


# =============================================================================
# AOA SWEEP
# =============================================================================
def plot_aoa_sweep_comparison(
    sweep_df,
    output_dir: str | Path,
) -> None:
    """
    AoA sweep figure:
        - CL vs AoA
        - CD vs AoA
        - CM vs AoA

    The user wrote CD twice, so the third subplot is CM, matching the
    original aerodynamic output variables.
    """
    out = _prepare_output_dir(output_dir)
    save_path = out / "aoa_sweep_CL_CD_CM.png"

    required = {
        "alpha_deg",
        "porous_CL",
        "porous_CD",
        "porous_CM",
        "solid_panel_CL",
        "solid_panel_CD",
        "solid_panel_CM",
    }
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep_df missing columns: {sorted(missing)}")

    has_xfoil = all(
        c in sweep_df.columns
        for c in ("xfoil_CL", "xfoil_CD", "xfoil_CM")
    )

    alpha = sweep_df["alpha_deg"].to_numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, key in zip(axs, ["CL", "CD", "CM"]):
        ax.plot(
            alpha,
            sweep_df[f"porous_{key}"],
            "o-",
            lw=1.5,
            ms=4,
            label="Porous panel",
        )
        ax.plot(
            alpha,
            sweep_df[f"solid_panel_{key}"],
            "s--",
            lw=1.5,
            ms=4,
            label="Solid panel",
        )

        if has_xfoil:
            ax.plot(
                alpha,
                sweep_df[f"xfoil_{key}"],
                "d-.",
                lw=1.5,
                ms=4,
                label="XFOIL",
            )

        ax.set_xlabel("AoA [deg]")
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs AoA")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot Saved] {save_path}")


# =============================================================================
# CONTOUR HELPERS
# =============================================================================
def _build_field_grid(solver, xlim, ylim, nx, ny):
    chord = solver.geom.chord

    x = np.linspace(xlim[0] * chord, xlim[1] * chord, nx)
    y = np.linspace(ylim[0] * chord, ylim[1] * chord, ny)

    XX, YY = np.meshgrid(x, y)
    XXn = XX / chord
    YYn = YY / chord

    return XX, YY, XXn, YYn


def _velocity_magnitude_for_display(solver, result, XX, YY):
    inside = solver.build_inside_mask(XX, YY)

    # Evaluate without interior NaNs to avoid contour artifacts.
    blank = np.zeros_like(XX, dtype=bool)
    U, V, _, _ = solver.velocity_field(XX, YY, result, inside=blank)

    Vmag = np.sqrt(U**2 + V**2)
    Vmag[inside] = np.nan

    return Vmag, U, V, inside


def _pressure_for_display(solver, result, XX, YY):
    inside = solver.build_inside_mask(XX, YY)

    blank = np.zeros_like(XX, dtype=bool)
    U, V, _, _ = solver.velocity_field(XX, YY, result, inside=blank)

    P = (
        solver.flow.p_inf
        + 0.5
        * solver.flow.rho_inf
        * (solver.flow.v_inf**2 - (U**2 + V**2))
    )
    P[inside] = np.nan

    return P, U, V, inside


# =============================================================================
# VELOCITY CONTOUR
# =============================================================================
def plot_velocity_contours_comparison(
    porous_solver,
    porous_result,
    solid_solver,
    solid_result,
    output_dir: str | Path,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    nx: int = 1000,
    ny: int = 1000,
) -> None:
    """
    Save side-by-side velocity-magnitude contours for porous and solid airfoils.

    This version restores the streamline integration code from the previous
    plotter.py exactly.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator
    from matplotlib.path import Path as MplPath

    out = _prepare_output_dir(output_dir)
    save_path = out / "velocity_contours_comparison.png"

    chord = porous_solver.geom.chord
    X = np.linspace(xlim[0] * chord, xlim[1] * chord, nx)
    Y = np.linspace(ylim[0] * chord, ylim[1] * chord, ny)
    XX, YY = np.meshgrid(X, Y)

    # Evaluate velocity field without interior masking so RegularGridInterpolator
    # stencils near the surface are never poisoned by NaN neighbours.
    blank_mask = np.zeros_like(XX, dtype=bool)

    inside_p = porous_solver.build_inside_mask(XX, YY)
    Up, Vp, _, _ = porous_solver.velocity_field(XX, YY, porous_result, inside=blank_mask)
    Vmag_p = np.sqrt(Up**2 + Vp**2)
    Vmag_p_disp = Vmag_p.copy()
    Vmag_p_disp[inside_p] = np.nan

    inside_s = solid_solver.build_inside_mask(XX, YY)
    Us, Vs, _, _ = solid_solver.velocity_field(XX, YY, solid_result, inside=blank_mask)
    Vmag_s = np.sqrt(Us**2 + Vs**2)
    Vmag_s_disp = Vmag_s.copy()
    Vmag_s_disp[inside_s] = np.nan

    finite_vals = np.concatenate(
        [Vmag_p_disp[np.isfinite(Vmag_p_disp)], Vmag_s_disp[np.isfinite(Vmag_s_disp)]]
    )
    vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))

    XXn, YYn = XX / chord, YY / chord

    # Streamline integration (arc-length parameterised, body-terminating)
    def integrate_streamlines(U, V, geom):
        fu = RegularGridInterpolator((Y, X), U, bounds_error=False, fill_value=0.0, method="linear")
        fv = RegularGridInterpolator((Y, X), V, bounds_error=False, fill_value=0.0, method="linear")
        poly = MplPath(np.column_stack([geom.XB, geom.YB]))

        def rhs(t, state):
            x, y = state
            u = float(fu([[y, x]])[0])
            v = float(fv([[y, x]])[0])
            speed = np.hypot(u, v)
            return ([0.0, 0.0] if speed < 1e-12 else [u / speed, v / speed])

        def hit_right(t, s): return s[0] - xlim[1] * chord
        hit_right.terminal = True; hit_right.direction = 1

        def hit_top(t, s): return s[1] - ylim[1] * chord
        hit_top.terminal = True; hit_top.direction = 1

        def hit_bottom(t, s): return s[1] - ylim[0] * chord
        hit_bottom.terminal = True; hit_bottom.direction = -1

        def hit_body(t, s):
            return -1.0 if poly.contains_points([[s[0], s[1]]])[0] else 1.0
        hit_body.terminal = True; hit_body.direction = -1

        events = [hit_right, hit_top, hit_bottom, hit_body]
        y_starts = np.linspace(ylim[0] * chord + 0.005 * chord,
                               ylim[1] * chord - 0.005 * chord, 50)
        x_start = xlim[0] * chord + 0.01 * chord
        s_max = 3.0 * np.hypot((xlim[1] - xlim[0]) * chord, (ylim[1] - ylim[0]) * chord)

        lines = []
        for y0 in y_starts:
            sol = solve_ivp(rhs, (0.0, s_max), [x_start, y0],
                            events=events, max_step=0.005 * chord,
                            rtol=1e-6, atol=1e-8, dense_output=False)
            if sol.y.shape[1] >= 2:
                lines.append(sol.y.copy())
        return lines

    lines_p = integrate_streamlines(Up, Vp, porous_solver.geom)
    lines_s = integrate_streamlines(Us, Vs, solid_solver.geom)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    # ---- Porous subplot ----
    axes[0].contourf(XXn, YYn, Vmag_p_disp, levels=20, vmin=vmin, vmax=vmax)
    axes[0].fill(porous_solver.geom.XB / chord, porous_solver.geom.YB / chord, "k", zorder=3)
    for line in lines_p:
        axes[0].plot(line[0] / chord, line[1] / chord, "-", color="white", lw=0.8, zorder=2)

    axes[0].set_title("Velocity Magnitude & Streamlines — Porous Airfoil")
    axes[0].set_xlabel("x/c")
    axes[0].set_ylabel("y/c")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_aspect("equal", adjustable="box")

    # ---- Solid subplot ----
    im1 = axes[1].contourf(XXn, YYn, Vmag_s_disp, levels=20, vmin=vmin, vmax=vmax)
    axes[1].fill(solid_solver.geom.XB / chord, solid_solver.geom.YB / chord, "k", zorder=3)
    for line in lines_s:
        axes[1].plot(line[0] / chord, line[1] / chord, "-", color="white", lw=0.8, zorder=2)

    axes[1].set_title("Velocity Magnitude & Streamlines — Solid Airfoil")
    axes[1].set_xlabel("x/c")
    axes[1].set_ylabel("y/c")
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Velocity magnitude [m/s]")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


# =============================================================================
# PRESSURE CONTOUR
# =============================================================================
def plot_pressure_contours_comparison(
    porous_solver,
    porous_result,
    solid_solver,
    solid_result,
    output_dir: str | Path,
    xlim: tuple[float, float] = (-0.2, 1.2),
    ylim: tuple[float, float] = (-0.35, 0.35),
    nx: int = 1000,
    ny: int = 1000,
) -> None:
    out = _prepare_output_dir(output_dir)
    save_path = out / "pressure_contours_comparison.png"

    XX, YY, XXn, YYn = _build_field_grid(porous_solver, xlim, ylim, nx, ny)

    Pp, _, _, _ = _pressure_for_display(
        porous_solver,
        porous_result,
        XX,
        YY,
    )
    Ps, _, _, _ = _pressure_for_display(
        solid_solver,
        solid_result,
        XX,
        YY,
    )

    finite = np.concatenate([Pp[np.isfinite(Pp)], Ps[np.isfinite(Ps)]])
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))

    chord = porous_solver.geom.chord

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im0 = axes[0].contourf(
        XXn,
        YYn,
        Pp,
        levels=40,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].fill(
        porous_solver.geom.XB / chord,
        porous_solver.geom.YB / chord,
        "k",
    )
    axes[0].set_title("Static Pressure: Porous")
    axes[0].set_xlabel("x/c")
    axes[0].set_ylabel("y/c")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].contourf(
        XXn,
        YYn,
        Ps,
        levels=40,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].fill(
        solid_solver.geom.XB / chord,
        solid_solver.geom.YB / chord,
        "k",
    )
    axes[1].set_title("Static Pressure: Solid")
    axes[1].set_xlabel("x/c")
    axes[1].set_ylabel("y/c")
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Pressure [Pa]")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot Saved] {save_path}")


# =============================================================================
# DIFFERENCE CONTOUR
# =============================================================================
def plot_difference_contours(
    porous_solver,
    porous_result,
    solid_solver,
    solid_result,
    output_dir: str | Path,
    xlim: tuple[float, float] = (-0.2, 1.2),
    ylim: tuple[float, float] = (-0.35, 0.35),
    nx: int = 350,
    ny: int = 350,
) -> None:
    out = _prepare_output_dir(output_dir)
    save_path = out / "difference_contours.png"

    XX, YY, XXn, YYn = _build_field_grid(porous_solver, xlim, ylim, nx, ny)

    Vmag_p, Up, Vp, inside_p = _velocity_magnitude_for_display(
        porous_solver,
        porous_result,
        XX,
        YY,
    )
    Vmag_s, Us, Vs, inside_s = _velocity_magnitude_for_display(
        solid_solver,
        solid_result,
        XX,
        YY,
    )

    inside = inside_p | inside_s

    dV = Vmag_p - Vmag_s
    dV[inside] = np.nan

    Pp = (
        porous_solver.flow.p_inf
        + 0.5
        * porous_solver.flow.rho_inf
        * (porous_solver.flow.v_inf**2 - (Up**2 + Vp**2))
    )
    Ps = (
        solid_solver.flow.p_inf
        + 0.5
        * solid_solver.flow.rho_inf
        * (solid_solver.flow.v_inf**2 - (Us**2 + Vs**2))
    )

    dP = Pp - Ps
    dP[inside] = np.nan

    def sym_limit(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 1.0
        return float(np.max(np.abs(finite)))

    vlim_v = sym_limit(dV)
    vlim_p = sym_limit(dP)

    chord = porous_solver.geom.chord

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im0 = axes[0].contourf(
        XXn,
        YYn,
        dV,
        levels=40,
        vmin=-vlim_v,
        vmax=vlim_v,
        cmap="RdBu_r",
    )
    axes[0].fill(
        porous_solver.geom.XB / chord,
        porous_solver.geom.YB / chord,
        "white",
        zorder=3,
    )
    axes[0].plot(
        porous_solver.geom.XB / chord,
        porous_solver.geom.YB / chord,
        "k-",
        lw=0.8,
        zorder=4,
    )
    axes[0].set_title("Velocity Difference: Porous - Solid")
    axes[0].set_xlabel("x/c")
    axes[0].set_ylabel("y/c")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_aspect("equal", adjustable="box")

    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.95)
    cbar0.set_label("Δ|V| [m/s]")

    im1 = axes[1].contourf(
        XXn,
        YYn,
        dP,
        levels=40,
        vmin=-vlim_p,
        vmax=vlim_p,
        cmap="RdBu_r",
    )
    axes[1].fill(
        porous_solver.geom.XB / chord,
        porous_solver.geom.YB / chord,
        "white",
        zorder=3,
    )
    axes[1].plot(
        porous_solver.geom.XB / chord,
        porous_solver.geom.YB / chord,
        "k-",
        lw=0.8,
        zorder=4,
    )
    axes[1].set_title("Pressure Difference: Porous - Solid")
    axes[1].set_xlabel("x/c")
    axes[1].set_ylabel("y/c")
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_aspect("equal", adjustable="box")

    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.95)
    cbar1.set_label("ΔP [Pa]")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot Saved] {save_path}")