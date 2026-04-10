"""
plotter.py

Plotting and reporting utilities for the lumped porous-connection model.

All figures are saved to disk instead of being displayed.
Geometry-based plots are normalised by chord so axes remain x/c and y/c.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from porous_network_optimisation import TwoPoreOneChamberNetwork
from solver import PanelGeometry, SPVPResult, SourceVortexPanelMethod
from xfoil import XFOILPolarPoint


def _prepare_output_dir(output_dir: str | Path) -> Path:
    """Create the output directory if it does not already exist."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# def plot_airfoil_with_pores(geom: PanelGeometry, network: TwoPoreOneChamberNetwork, output_dir: str | Path) -> None:
#     """Save the airfoil geometry with the two opening locations and internal link."""
#     out = _prepare_output_dir(output_dir)
#     save_path = out / "airfoil_with_porous_network.png"

#     p1 = network.pore_surface_point(geom, network.pore1)
#     p2 = network.pore_surface_point(geom, network.pore2)

#     plt.figure(figsize=(10, 4))
#     plt.plot(geom.XB / geom.chord, geom.YB / geom.chord, "k-", lw=1.4, label="Airfoil")
#     plt.plot([p1.x / geom.chord, p2.x / geom.chord], [p1.y / geom.chord, p2.y / geom.chord], "--", lw=1.4, label="Equivalent internal link")
#     plt.plot(p1.x / geom.chord, p1.y / geom.chord, "ro", ms=8, label=f"Opening 1 ({network.pore1.side})")
#     plt.plot(p2.x / geom.chord, p2.y / geom.chord, "bo", ms=8, label=f"Opening 2 ({network.pore2.side})")
#     plt.axis("equal")
#     plt.xlabel("x/c")
#     plt.ylabel("y/c")
#     plt.title("Airfoil with Lumped Porous Connection")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"[Plot Saved] {save_path}")

def plot_airfoil_with_pores(
    geom: PanelGeometry,
    network: TwoPoreOneChamberNetwork,
    output_dir: str | Path,
    coupled_result=None,
) -> None:
    """
    Save the airfoil geometry with pore locations and the internal link.

    If coupled_result is provided, also draw an arrow showing the internal
    flow direction based on the sign of Q:
    - Q > 0 : pore1 -> pore2
    - Q < 0 : pore2 -> pore1
    """
    out = _prepare_output_dir(output_dir)
    save_path = out / "airfoil_with_porous_network.png"

    p1 = network.pore_surface_point(geom, network.pore1)
    p2 = network.pore_surface_point(geom, network.pore2)

    plt.figure(figsize=(10, 4))
    plt.plot(geom.XB / geom.chord, geom.YB / geom.chord, "k-", lw=1.4, label="Airfoil")

    # dashed link between pores
    plt.plot(
        [p1.x / geom.chord, p2.x / geom.chord],
        [p1.y / geom.chord, p2.y / geom.chord],
        "--",
        lw=1.4,
        color="gray",
        label="Internal link",
    )

    # pore markers
    plt.plot(
        p1.x / geom.chord,
        p1.y / geom.chord,
        "ro",
        ms=8,
        label=f"Pore 1 ({network.pore1.side})",
    )
    plt.plot(
        p2.x / geom.chord,
        p2.y / geom.chord,
        "bo",
        ms=8,
        label=f"Pore 2 ({network.pore2.side})",
    )

    # optional flow-direction arrow
    if coupled_result is not None:
        q = coupled_result.network_state.Q

        if q >= 0.0:
            xa, ya = p1.x / geom.chord, p1.y / geom.chord
            xb, yb = p2.x / geom.chord, p2.y / geom.chord
            flow_label = "Internal flow: pore1 → pore2"
        else:
            xa, ya = p2.x / geom.chord, p2.y / geom.chord
            xb, yb = p1.x / geom.chord, p1.y / geom.chord
            flow_label = "Internal flow: pore2 → pore1"

        dx = xb - xa
        dy = yb - ya

        # draw arrow over the dashed link
        plt.arrow(
            xa,
            ya,
            0.75 * dx,
            0.75 * dy,
            length_includes_head=True,
            head_width=0.01,
            head_length=0.02,
            linewidth=2.0,
            color="green",
            alpha=0.9,
        )

        xm = 0.5 * (xa + xb)
        ym = 0.5 * (ya + yb)
        plt.text(
            xm,
            ym,
            f"{flow_label}\nQ = {q:.3e} m$^3$/s",
            fontsize=9,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
        )

    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil with Porous Network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Plot Saved] {save_path}")

def plot_cp_distribution(geom: PanelGeometry, porous_result: SPVPResult, output_dir: str | Path, xfoil_cp_data: dict | None = None) -> None:
    """Save the surface Cp distribution plot and optionally overlay XFOIL Cp."""
    out = _prepare_output_dir(output_dir)
    save_path = out / "cp_distribution_comparison.png"

    plt.figure(figsize=(10, 5))
    porous_upper = geom.upper_panel_ids
    porous_lower = geom.lower_panel_ids

    plt.plot(geom.XC[porous_upper] / geom.chord, porous_result.Cp[porous_upper], "ro-", ms=3, lw=1.0, label="Porous Cp (upper)")
    plt.plot(geom.XC[porous_lower] / geom.chord, porous_result.Cp[porous_lower], "bo-", ms=3, lw=1.0, label="Porous Cp (lower)")

    if xfoil_cp_data is not None:
        if "x_upper" in xfoil_cp_data and "cp_upper" in xfoil_cp_data:
            plt.plot(xfoil_cp_data["x_upper"], xfoil_cp_data["cp_upper"], "k--", lw=1.3, label="XFOIL solid Cp (upper)")
        if "x_lower" in xfoil_cp_data and "cp_lower" in xfoil_cp_data:
            plt.plot(xfoil_cp_data["x_lower"], xfoil_cp_data["cp_lower"], "g--", lw=1.3, label="XFOIL solid Cp (lower)")

    plt.gca().invert_yaxis()
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title("Surface Pressure Coefficient: Porous vs Solid XFOIL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_velocity_contours_comparison(
    porous_solver: SourceVortexPanelMethod,
    porous_result: SPVPResult,
    solid_solver: SourceVortexPanelMethod,
    solid_result: SPVPResult,
    output_dir: str | Path,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    nx: int = 1000,
    ny: int = 1000,
) -> None:
    """Save side-by-side velocity-magnitude contours for porous and solid airfoils."""
    out = _prepare_output_dir(output_dir)
    save_path = out / "velocity_contours_comparison.png"

    chord = porous_solver.geom.chord
    X = np.linspace(xlim[0] * chord, xlim[1] * chord, nx)
    Y = np.linspace(ylim[0] * chord, ylim[1] * chord, ny)
    XX, YY = np.meshgrid(X, Y)

    inside_p = porous_solver.build_inside_mask(XX, YY)
    Up, Vp, _, inside_p = porous_solver.velocity_field(XX, YY, porous_result, inside=inside_p)
    Vmag_p = np.sqrt(Up**2 + Vp**2)
    Vmag_p[inside_p] = np.nan

    inside_s = solid_solver.build_inside_mask(XX, YY)
    Us, Vs, _, inside_s = solid_solver.velocity_field(XX, YY, solid_result, inside=inside_s)
    Vmag_s = np.sqrt(Us**2 + Vs**2)
    Vmag_s[inside_s] = np.nan

    finite_vals = np.concatenate([Vmag_p[np.isfinite(Vmag_p)], Vmag_s[np.isfinite(Vmag_s)]])
    vmin = np.min(finite_vals)
    vmax = np.max(finite_vals)

    XXn = XX / chord
    YYn = YY / chord
    x_norm = X / chord
    y_norm = Y / chord

    Up_masked = np.ma.array(Up, mask=inside_p)
    Vp_masked = np.ma.array(Vp, mask=inside_p)
    Us_masked = np.ma.array(Us, mask=inside_s)
    Vs_masked = np.ma.array(Vs, mask=inside_s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    im0 = axes[0].contourf(XXn, YYn, Vmag_p, levels=40, vmin=vmin, vmax=vmax)
    axes[0].fill(porous_solver.geom.XB / chord, porous_solver.geom.YB / chord, "k", zorder=3)
    axes[0].streamplot(x_norm, y_norm, Up_masked, Vp_masked, color=(1.0, 1.0, 1.0, 0.6), linewidth=0.8, density=1.2, arrowsize=1.0, zorder=2)
    axes[0].set_title("Velocity Magnitude & Streamlines - Porous Airfoil")
    axes[0].set_xlabel("x/c")
    axes[0].set_ylabel("y/c")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_aspect("equal", adjustable="box")

    im1 = axes[1].contourf(XXn, YYn, Vmag_s, levels=40, vmin=vmin, vmax=vmax)
    axes[1].fill(solid_solver.geom.XB / chord, solid_solver.geom.YB / chord, "k", zorder=3)
    axes[1].streamplot(x_norm, y_norm, Us_masked, Vs_masked, color=(1.0, 1.0, 1.0, 0.6), linewidth=0.8, density=1.2, arrowsize=1.0, zorder=2)
    axes[1].set_title("Velocity Magnitude & Streamlines - Solid Airfoil")
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


def plot_pressure_contours_comparison(
    porous_solver: SourceVortexPanelMethod,
    porous_result: SPVPResult,
    solid_solver: SourceVortexPanelMethod,
    solid_result: SPVPResult,
    output_dir: str | Path,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    nx: int = 1000,
    ny: int = 1000,
) -> None:
    """Save side-by-side static-pressure contours for porous and solid airfoils."""
    out = _prepare_output_dir(output_dir)
    save_path = out / "pressure_contours_comparison.png"

    chord = porous_solver.geom.chord
    X = np.linspace(xlim[0] * chord, xlim[1] * chord, nx)
    Y = np.linspace(ylim[0] * chord, ylim[1] * chord, ny)
    XX, YY = np.meshgrid(X, Y)

    inside_p = porous_solver.build_inside_mask(XX, YY)
    Up, Vp, _, inside_p = porous_solver.velocity_field(XX, YY, porous_result, inside=inside_p)
    Pp = porous_solver.flow.p_inf + 0.5 * porous_solver.flow.rho_inf * (porous_solver.flow.v_inf**2 - (Up**2 + Vp**2))
    Pp[inside_p] = np.nan

    inside_s = solid_solver.build_inside_mask(XX, YY)
    Us, Vs, _, inside_s = solid_solver.velocity_field(XX, YY, solid_result, inside=inside_s)
    Ps = solid_solver.flow.p_inf + 0.5 * solid_solver.flow.rho_inf * (solid_solver.flow.v_inf**2 - (Us**2 + Vs**2))
    Ps[inside_s] = np.nan

    finite_vals = np.concatenate([Pp[np.isfinite(Pp)], Ps[np.isfinite(Ps)]])
    vmin = np.min(finite_vals)
    vmax = np.max(finite_vals)

    XXn = XX / chord
    YYn = YY / chord
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    im0 = axes[0].contourf(XXn, YYn, Pp, levels=40, vmin=vmin, vmax=vmax)
    axes[0].fill(porous_solver.geom.XB / chord, porous_solver.geom.YB / chord, "k")
    axes[0].set_title("Pressure Contour - Porous Airfoil")
    axes[0].set_xlabel("x/c")
    axes[0].set_ylabel("y/c")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_aspect("equal", adjustable="box")

    im1 = axes[1].contourf(XXn, YYn, Ps, levels=40, vmin=vmin, vmax=vmax)
    axes[1].fill(solid_solver.geom.XB / chord, solid_solver.geom.YB / chord, "k")
    axes[1].set_title("Pressure Contour - Solid Airfoil")
    axes[1].set_xlabel("x/c")
    axes[1].set_ylabel("y/c")
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Pressure [Pa]")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")

def plot_internal_flow_direction(
    geom: PanelGeometry,
    network: TwoPoreOneChamberNetwork,
    coupled_result,
    output_dir: str | Path,
) -> None:
    """
    Save a dedicated plot showing only the airfoil, pores, and internal flow direction.
    """
    out = _prepare_output_dir(output_dir)
    save_path = out / "internal_flow_direction.png"

    p1 = network.pore_surface_point(geom, network.pore1)
    p2 = network.pore_surface_point(geom, network.pore2)

    x1 = p1.x / geom.chord
    y1 = p1.y / geom.chord
    x2 = p2.x / geom.chord
    y2 = p2.y / geom.chord

    q = float(coupled_result.network_state.Q)

    plt.figure(figsize=(10, 4))
    plt.plot(geom.XB / geom.chord, geom.YB / geom.chord, "k-", lw=1.4)

    plt.plot([x1, x2], [y1, y2], "--", lw=1.6)
    plt.plot(x1, y1, "ro", ms=8, label=f"Pore 1 ({network.pore1.side})")
    plt.plot(x2, y2, "bo", ms=8, label=f"Pore 2 ({network.pore2.side})")

    if q > 0.0:
        xa, ya, xb, yb = x1, y1, x2, y2
        flow_text = f"Q = {q:.3e} m^3/s (pore1 → pore2)"
    elif q < 0.0:
        xa, ya, xb, yb = x2, y2, x1, y1
        flow_text = f"Q = {q:.3e} m^3/s (pore2 → pore1)"
    else:
        xa = ya = xb = yb = None
        flow_text = "Q = 0"

    if xa is not None:
        dx = xb - xa
        dy = yb - ya
        plt.arrow(
            xa,
            ya,
            0.75 * dx,
            0.75 * dy,
            length_includes_head=True,
            head_width=0.01,
            head_length=0.02,
            lw=2.0,
        )

    plt.text(
        0.5 * (x1 + x2),
        0.5 * (y1 + y2) + 0.03,
        flow_text,
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
    )

    plt.axis("equal")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Internal Flow Direction for Best Porous Design")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Plot Saved] {save_path}")

def plot_aoa_sweep_comparison(
    sweep_df,
    output_dir: str | Path,
) -> None:
    """
    Save CL, CD, and CM comparison plots versus angle of attack.

    Parameters
    ----------
    sweep_df : pandas.DataFrame
        Sweep comparison table.

        Required columns
        ----------------
        - alpha_deg
        - porous_CL, porous_CD, porous_CM
        - solid_panel_CL, solid_panel_CD, solid_panel_CM

        Optional columns
        ----------------
        - xfoil_CL, xfoil_CD, xfoil_CM

    output_dir : str | Path
        Directory where the figure will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function plots up to three curves in each subplot:
    - Porous
    - Solid Panel Method
    - XFOIL (if available)
    """
    out = _prepare_output_dir(output_dir)
    save_path = out / "aoa_sweep_comparison.png"

    required_cols = {
        "alpha_deg",
        "porous_CL", "porous_CD", "porous_CM",
        "solid_panel_CL", "solid_panel_CD", "solid_panel_CM",
    }
    missing = required_cols - set(sweep_df.columns)
    if missing:
        raise ValueError(
            f"sweep_df is missing required columns: {sorted(missing)}"
        )

    alpha = sweep_df["alpha_deg"].to_numpy()

    has_xfoil = all(
        col in sweep_df.columns
        for col in ("xfoil_CL", "xfoil_CD", "xfoil_CM")
    )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # ------------------------------------------------------------------
    # CL
    # ------------------------------------------------------------------
    axs[0].plot(
        alpha,
        sweep_df["porous_CL"],
        "o-",
        lw=1.5,
        ms=4,
        label="Porous",
    )
    axs[0].plot(
        alpha,
        sweep_df["solid_panel_CL"],
        "s--",
        lw=1.5,
        ms=4,
        label="Solid Panel Method",
    )
    if has_xfoil:
        axs[0].plot(
            alpha,
            sweep_df["xfoil_CL"],
            "d-.",
            lw=1.5,
            ms=4,
            label="XFOIL",
        )

    axs[0].set_xlabel("Angle of attack [deg]")
    axs[0].set_ylabel("CL")
    axs[0].set_title("AoA Sweep Comparison: CL")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # ------------------------------------------------------------------
    # CD
    # ------------------------------------------------------------------
    axs[1].plot(
        alpha,
        sweep_df["porous_CD"],
        "o-",
        lw=1.5,
        ms=4,
        label="Porous",
    )
    axs[1].plot(
        alpha,
        sweep_df["solid_panel_CD"],
        "s--",
        lw=1.5,
        ms=4,
        label="Solid Panel Method",
    )
    if has_xfoil:
        axs[1].plot(
            alpha,
            sweep_df["xfoil_CD"],
            "d-.",
            lw=1.5,
            ms=4,
            label="XFOIL",
        )

    axs[1].set_xlabel("Angle of attack [deg]")
    axs[1].set_ylabel("CD")
    axs[1].set_title("AoA Sweep Comparison: CD")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # ------------------------------------------------------------------
    # CM
    # ------------------------------------------------------------------
    axs[2].plot(
        alpha,
        sweep_df["porous_CM"],
        "o-",
        lw=1.5,
        ms=4,
        label="Porous",
    )
    axs[2].plot(
        alpha,
        sweep_df["solid_panel_CM"],
        "s--",
        lw=1.5,
        ms=4,
        label="Solid Panel Method",
    )
    if has_xfoil:
        axs[2].plot(
            alpha,
            sweep_df["xfoil_CM"],
            "d-.",
            lw=1.5,
            ms=4,
            label="XFOIL",
        )

    axs[2].set_xlabel("Angle of attack [deg]")
    axs[2].set_ylabel("CM")
    axs[2].set_title("AoA Sweep Comparison: CM")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot Saved] {save_path}")


def print_xfoil_comparison(result: SPVPResult, xfoil_point: XFOILPolarPoint | None) -> None:
    """Print porous-airfoil results and the solid-airfoil XFOIL comparison point."""
    print("=" * 72)
    print("POROUS AIRFOIL RESULT")
    print("=" * 72)
    print(f"CL : {result.CL:.8f}")
    print(f"CD : {result.CD:.8e}")
    print(f"CM : {result.CM:.8f}")

    if xfoil_point is not None:
        print()
        print("=" * 72)
        print("SOLID AIRFOIL XFOIL BASELINE")
        print("=" * 72)
        print(f"alpha_xfoil : {xfoil_point.alpha:.3f} deg")
        print(f"CL_xfoil    : {xfoil_point.CL:.8f}")
        print(f"CD_xfoil    : {xfoil_point.CD:.8e}")
        print(f"CM_xfoil    : {xfoil_point.CM:.8f}")
        print()
        print(f"Delta CL    : {result.CL - xfoil_point.CL:.8f}")
        print(f"Delta CD    : {result.CD - xfoil_point.CD:.8e}")
        print(f"Delta CM    : {result.CM - xfoil_point.CM:.8f}")


def print_best_design_summary(best_result, best_network: TwoPoreOneChamberNetwork, geom: PanelGeometry) -> None:
    """Print a summary of the best porous-network design."""
    coupled = best_result.coupled_result
    state = coupled.network_state
    aero = coupled.aero_result

    print("=" * 72)
    print("BEST POROUS NETWORK DESIGN")
    print("=" * 72)
    print(f"Topology                : ({best_result.topology.pore1_side}, {best_result.topology.pore2_side})")
    print(f"x1, x2 [x/c]            : {best_network.pore1.x_frac:.6f}, {best_network.pore2.x_frac:.6f}")
    print(
        f"x1, x2 [m]              : {geom.x_from_fraction(best_network.pore1.x_frac):.6f}, "
        f"{geom.x_from_fraction(best_network.pore2.x_frac):.6f}"
    )
    print(f"d1, d2 [m]              : {best_network.pore1.diameter:.6e}, {best_network.pore2.diameter:.6e}")
    print(f"Effective Dh [m]        : {best_network.chamber.hydraulic_diameter:.6e}")
    print(f"Effective Area [m^2]    : {best_network.chamber.area:.6e}")
    print(f"CL                      : {aero.CL:.8f}")
    print(f"CD                      : {aero.CD:.8e}")
    print(f"CM                      : {aero.CM:.8f}")
    print(f"Q [m^3/s]               : {state.Q:.8e}")
    print(f"Rs [Pa·s/m^3]           : {state.Rs:.8e}")
    print(f"p1 [Pa]                 : {state.p1:.3f}")
    print(f"p2 [Pa]                 : {state.p2:.3f}")
    print(f"p_internal_1 [Pa]       : {state.p_internal_1:.3f}")
    print(f"p_internal_2 [Pa]       : {state.p_internal_2:.3f}")
    print(f"dp_total [Pa]           : {state.dp_total:.3f}")
    print(f"Re_equivalent           : {state.reynolds_equivalent:.3f}")
    print(f"Coupling converged      : {coupled.converged}")
    print(f"Coupling iterations     : {coupled.iterations}")
    print(f"max|vn| [m/s]            : {coupled.max_vn:.8e}")
    print(f"max|vn| / Vinf           : {coupled.max_vn_over_vinf:.8e}")
