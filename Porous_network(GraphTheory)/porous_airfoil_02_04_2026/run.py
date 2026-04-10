"""
run.py

Main executable for porous-airfoil optimisation using a lumped hydraulic
resistance model.

Workflow
--------
1. Define the operating point using Reynolds number.
2. Build the airfoil geometry and aerodynamic solver.
3. Optimise the lumped porous connection over all four opening topologies.
4. Select the best converged design.
5. Compare against a solid-airfoil XFOIL baseline.
6. Export CSV files and save plots.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from solver import (
    AirfoilConfig,
    CouplingConfig,
    FlowConfig,
    PanelGeometry,
    ReferenceConfig,
    SourceVortexPanelMethod,
    generate_naca4,
)
from porous_network_optimisation import (
    CoupledPorousAirfoilSolver,
    OptimizationConfig,
    PorousNetworkOptimizer,
)
from plotter import (
    plot_airfoil_with_pores,
    plot_aoa_sweep_comparison,
    plot_cp_distribution,
    plot_pressure_contours_comparison,
    plot_velocity_contours_comparison,
    print_best_design_summary,
    print_xfoil_comparison,
    plot_internal_flow_direction,
)
from xfoil import (
    load_xfoil_cp,
    load_xfoil_polar_dataframe,
    run_and_load_xfoil_point,
    run_xfoil_cp,
    run_xfoil_polar_sweep,
)


def reynolds_to_velocity(reynolds: float, rho: float, mu: float, chord: float) -> float:
    """
    Convert Reynolds number to freestream velocity using:

        Re = rho * V * c / mu
    """
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if rho <= 0.0:
        raise ValueError("Density must be positive.")
    if mu <= 0.0:
        raise ValueError("Dynamic viscosity must be positive.")
    if chord <= 0.0:
        raise ValueError("Chord must be positive.")

    return reynolds * mu / (rho * chord)


def add_xfoil_to_sweep_comparison(
    sweep_df: pd.DataFrame,
    xfoil_sweep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge XFOIL AoA sweep data into an existing comparison dataframe.
    """
    xfoil_df = xfoil_sweep_df.copy()
    xfoil_df.columns = [str(c).strip() for c in xfoil_df.columns]

    rename_map: dict[str, str] = {}

    for col in xfoil_df.columns:
        c = col.strip().lower()

        if c in {"alpha", "alpha_deg", "aoa", "aoa_deg"}:
            rename_map[col] = "alpha_deg"
        elif c == "cl":
            rename_map[col] = "xfoil_CL"
        elif c == "cd":
            rename_map[col] = "xfoil_CD"
        elif c == "cm":
            rename_map[col] = "xfoil_CM"

    xfoil_df = xfoil_df.rename(columns=rename_map)

    required = {"alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM"}
    missing = required - set(xfoil_df.columns)
    if missing:
        raise ValueError(
            f"XFOIL sweep dataframe is missing columns after normalization: {sorted(missing)}. "
            f"Available columns: {list(xfoil_sweep_df.columns)}"
        )

    return sweep_df.merge(
        xfoil_df[["alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM"]],
        on="alpha_deg",
        how="left",
    )


def export_results_to_csv(
    output_dir: str | Path,
    geom: PanelGeometry,
    best_result,
    best_network,
    reynolds: float,
    xfoil_point=None,
    xfoil_cp_data: dict | None = None,
    sweep_df: pd.DataFrame | None = None,
) -> None:
    """Export porous-airfoil results and optional XFOIL outputs to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coupled = best_result.coupled_result
    aero = coupled.aero_result
    state = coupled.network_state

    summary = {
        "topology_pore1_side": best_result.topology.pore1_side,
        "topology_pore2_side": best_result.topology.pore2_side,
        "reynolds": reynolds,
        "x1_frac": best_network.pore1.x_frac,
        "x2_frac": best_network.pore2.x_frac,
        "x1_abs_m": geom.x_from_fraction(best_network.pore1.x_frac),
        "x2_abs_m": geom.x_from_fraction(best_network.pore2.x_frac),
        "d1_m": best_network.pore1.diameter,
        "d2_m": best_network.pore2.diameter,
        "effective_hydraulic_diameter_m": best_network.chamber.hydraulic_diameter,
        "friction_scale": best_network.chamber.friction_scale,
        "CL": aero.CL,
        "CD": aero.CD,
        "CM": aero.CM,
        "CL_kj": aero.CL_kj,
        "Cx": aero.Cx,
        "Cy": aero.Cy,
        "Q_m3_s": state.Q,
        "Rs_Pa_s_per_m3": state.Rs,
        "p1_Pa": state.p1,
        "p2_Pa": state.p2,
        "p_internal_1_Pa": state.p_internal_1,
        "p_internal_2_Pa": state.p_internal_2,
        "dp_total_Pa": state.dp_total,
        "Re_equivalent": state.reynolds_equivalent,
        "coupling_converged": coupled.converged,
        "coupling_iterations": coupled.iterations,
        "system_residual_norm": aero.system_residual_norm,
        "kutta_residual": aero.kutta_residual,
        "normal_bc_residual_max": aero.normal_bc_residual_max,
        "source_sum": aero.source_sum,
        "max_vn_m_s": coupled.max_vn,
        "max_vn_over_vinf": coupled.max_vn_over_vinf,
    }

    if xfoil_point is not None:
        summary["xfoil_alpha_deg"] = xfoil_point.alpha
        summary["xfoil_solid_CL"] = xfoil_point.CL
        summary["xfoil_solid_CD"] = xfoil_point.CD
        summary["xfoil_solid_CM"] = xfoil_point.CM
        summary["delta_CL_vs_solid_xfoil"] = aero.CL - xfoil_point.CL
        summary["delta_CD_vs_solid_xfoil"] = aero.CD - xfoil_point.CD
        summary["delta_CM_vs_solid_xfoil"] = aero.CM - xfoil_point.CM

    summary_path = output_dir / "best_design_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[CSV Saved] {summary_path}")

    sp1 = best_network.pore_surface_point(geom, best_network.pore1)
    sp2 = best_network.pore_surface_point(geom, best_network.pore2)

    ids1 = best_network._covered_panels(
        geom=geom,
        center_panel_id=sp1.panel_id,
        side=best_network.pore1.side,
        opening_diameter=best_network.pore1.diameter,
        min_panels_per_pore=2,
    )
    ids2 = best_network._covered_panels(
        geom=geom,
        center_panel_id=sp2.panel_id,
        side=best_network.pore2.side,
        opening_diameter=best_network.pore2.diameter,
        min_panels_per_pore=2,
    )

    vn = coupled.normal_transpiration
    pore1_mask = np.zeros(geom.num_pan, dtype=int)
    pore2_mask = np.zeros(geom.num_pan, dtype=int)
    pore1_mask[ids1] = 1
    pore2_mask[ids2] = 1

    df_surface = pd.DataFrame(
        {
            "panel_id": np.arange(geom.num_pan),
            "XB_start_m": geom.XB[:-1],
            "YB_start_m": geom.YB[:-1],
            "XB_end_m": geom.XB[1:],
            "YB_end_m": geom.YB[1:],
            "XC_m": geom.XC,
            "YC_m": geom.YC,
            "XC_over_c": geom.XC / geom.chord,
            "YC_over_c": geom.YC / geom.chord,
            "panel_length_m": geom.S,
            "tx": geom.tx,
            "ty": geom.ty,
            "nx": geom.nx,
            "ny": geom.ny,
            "Vt_m_s": aero.Vt,
            "Cp": aero.Cp,
            "normal_transpiration_m_s": vn,
            "is_pore1_panel": pore1_mask,
            "is_pore2_panel": pore2_mask,
        }
    )
    surface_path = output_dir / "surface_data.csv"
    df_surface.to_csv(surface_path, index=False)
    print(f"[CSV Saved] {surface_path}")

    if xfoil_cp_data is not None:
        xfoil_cp_path = output_dir / "xfoil_cp_distribution.csv"
        pd.DataFrame(
            {
                "x_over_c": xfoil_cp_data["x"],
                "y_over_c": xfoil_cp_data["y"],
                "cp": xfoil_cp_data["cp"],
            }
        ).to_csv(xfoil_cp_path, index=False)
        print(f"[CSV Saved] {xfoil_cp_path}")

    if sweep_df is not None:
        sweep_path = output_dir / "aoa_sweep_comparison.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"[CSV Saved] {sweep_path}")


def run_porous_aoa_sweep(
    airfoil_name: str,
    chord: float,
    n_panels: int,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    reynolds: float,
    best_network,
    coupling: CouplingConfig,
    alpha_start: float = -5.0,
    alpha_end: float = 12.0,
    alpha_step: float = 1.0,
) -> pd.DataFrame:
    """
    Run an angle-of-attack sweep for the optimised porous network and also
    compute the solid-airfoil panel-method baseline at each angle.
    """
    rows: list[dict[str, float]] = []
    v_inf = reynolds_to_velocity(reynolds, rho_inf, mu_inf, chord)

    for alpha in np.arange(alpha_start, alpha_end + 0.1, alpha_step):
        flow_i = FlowConfig(
            aoa_deg=float(alpha),
            v_inf=v_inf,
            rho_inf=rho_inf,
            mu_inf=mu_inf,
            p_inf=p_inf,
        )

        ref_i = ReferenceConfig(x_ref=0.25 * chord, y_ref=0.0)
        XB_i, YB_i = generate_naca4(airfoil_name, n_panels, chord=chord)
        geom_i = PanelGeometry(XB_i, YB_i, flow_i.aoa_deg)
        aero_solver_i = SourceVortexPanelMethod(geom_i, flow_i, ref_i)

        solid_i = aero_solver_i.solve()

        coupled_i = CoupledPorousAirfoilSolver(
            aero_solver=aero_solver_i,
            network=best_network,
            coupling=coupling,
            opt=None,
        ).solve(verbose=False)

        rows.append(
            {
                "alpha_deg": float(alpha),
                "porous_CL": float(coupled_i.aero_result.CL),
                "porous_CD": float(coupled_i.aero_result.CD),
                "porous_CM": float(coupled_i.aero_result.CM),
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
            }
        )

    return pd.DataFrame(rows)


def build_gain_shortlist_dataframe(
    results,
    optimizer,
    geom: PanelGeometry,
    min_gain_percent: float = 5.0,
) -> pd.DataFrame:
    """
    Build a table of all converged designs whose lift gain exceeds the threshold.
    """
    rows: list[dict] = []
    solid_cl = float(optimizer.solid_cl)

    for r in results:
        if r.coupled_result is None:
            continue
        if not r.coupled_result.converged:
            continue

        cr = r.coupled_result
        aero = cr.aero_result
        state = cr.network_state
        network = optimizer.design_to_network(r.x_opt, r.topology)

        gain_percent = 100.0 * (aero.CL / solid_cl - 1.0)
        if gain_percent < min_gain_percent:
            continue

        rows.append(
            {
                "topology_pore1_side": r.topology.pore1_side,
                "topology_pore2_side": r.topology.pore2_side,
                "success": r.success,
                "message": r.message,
                "best_fun": r.best_fun,
                "gain_percent": gain_percent,
                "CL": aero.CL,
                "CD": aero.CD,
                "CM": aero.CM,
                "Q_m3_s": state.Q,
                "Rs_Pa_s_per_m3": state.Rs,
                "dp_total_Pa": state.dp_total,
                "Re_equivalent": state.reynolds_equivalent,
                "max_vn_m_s": cr.max_vn,
                "max_vn_over_vinf": cr.max_vn_over_vinf,
                "coupling_iterations": cr.iterations,
                "x1_frac": network.pore1.x_frac,
                "x2_frac": network.pore2.x_frac,
                "x1_abs_m": geom.x_from_fraction(network.pore1.x_frac),
                "x2_abs_m": geom.x_from_fraction(network.pore2.x_frac),
                "d1_m": network.pore1.diameter,
                "d2_m": network.pore2.diameter,
                "effective_hydraulic_diameter_m": network.chamber.hydraulic_diameter,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="gain_percent", ascending=False).reset_index(drop=True)
    return df


def run() -> None:
    """Main project driver."""
    print("=" * 72)
    print("STARTING POROUS AIRFOIL OPTIMIZATION")
    print("=" * 72)

    xfoil_exe_path = r"C:\Users\kusha\OneDrive\Desktop\airfoil_rewrite\xfoil.exe"
    xfoil_mach = 0.0
    xfoil_timeout = 60.0

    reynolds = 5.0e5
    aoa_deg = 4.0
    rho_inf = 1.225
    mu_inf = 1.8e-5
    p_inf = 101325.0

    airfoil_name = "0018"
    chord = 1.0
    n_panels = 1000

    alpha_sweep_start = -5.0
    alpha_sweep_end = 12.0
    alpha_sweep_step = 1.0

    v_inf = reynolds_to_velocity(reynolds=reynolds, rho=rho_inf, mu=mu_inf, chord=chord)
    print(f"[Setup] Reynolds number input      : {reynolds:.6e}")
    print(f"[Setup] Computed freestream speed  : {v_inf:.6f} m/s")
    print(f"[Setup] XFOIL executable path      : {xfoil_exe_path}")

    flow = FlowConfig(
        aoa_deg=aoa_deg,
        v_inf=v_inf,
        rho_inf=rho_inf,
        mu_inf=mu_inf,
        p_inf=p_inf,
    )
    airfoil = AirfoilConfig(
        airfoil_name=airfoil_name,
        chord=chord,
        n_panels=n_panels,
    )
    ref = ReferenceConfig(x_ref=0.25 * chord, y_ref=0.0)
    coupling = CouplingConfig(max_iter=60, tol_vn=1e-6, tol_q=1e-11, relaxation=0.5)
    opt = OptimizationConfig()

    print("[Setup] Generating airfoil geometry...")
    XB, YB = generate_naca4(airfoil.airfoil_name, airfoil.n_panels, chord=airfoil.chord)
    geom = PanelGeometry(XB, YB, flow.aoa_deg)

    print("[Setup] Building panel solver...")
    aero_solver = SourceVortexPanelMethod(geom, flow, ref)

    print("[Setup] Solving solid-airfoil panel baseline...")
    solid_result = aero_solver.solve()
    print(f"[Setup] Solid-airfoil baseline CL   : {solid_result.CL:.8f}")
    print("[Setup] Objective uses gain + actuation + drag penalties (no hard 10% target penalty).")

    print("[Setup] Building optimizer...")
    optimizer = PorousNetworkOptimizer(geom, aero_solver, coupling, opt)

    plot_dir = Path.cwd() / "plots"
    csv_dir = Path.cwd() / "csv_results"
    xfoil_dir = Path.cwd() / "xfoil_output"

    print(f"[Setup] Plot output folder         : {plot_dir}")
    print(f"[Setup] CSV output folder          : {csv_dir}")
    print(f"[Setup] XFOIL output folder        : {xfoil_dir}")

    t0 = time.perf_counter()
    print("[Run] Starting Differential Evolution across all topologies...")
    all_results = optimizer.optimize_all_topologies()

    csv_dir.mkdir(parents=True, exist_ok=True)

    all_above_5_df = pd.DataFrame(optimizer.saved_design_rows)
    if not all_above_5_df.empty:
        all_above_5_df = (
            all_above_5_df
            .sort_values(by="gain_percent", ascending=False)
            .drop_duplicates(
                subset=[
                    "topology_pore1_side",
                    "topology_pore2_side",
                    "x1_frac",
                    "x2_frac",
                    "d1_m",
                    "d2_m",
                    "effective_hydraulic_diameter_m",
                ]
            )
            .reset_index(drop=True)
        )

    all_above_5_path = csv_dir / "all_designs_above_5_percent_lift_gain.csv"
    all_above_5_df.to_csv(all_above_5_path, index=False)
    print(f"[CSV Saved] {all_above_5_path}")

    feasible = [
        r for r in all_results
        if r.coupled_result is not None and r.coupled_result.converged
    ]
    if not feasible:
        raise RuntimeError("No converged feasible porous-network design found.")

    gain_shortlist_df = build_gain_shortlist_dataframe(
        results=all_results,
        optimizer=optimizer,
        geom=geom,
        min_gain_percent=5.0,
    )

    if not gain_shortlist_df.empty:
        top_row = gain_shortlist_df.iloc[0]

        def matches_top_row(r) -> bool:
            if r.coupled_result is None or not r.coupled_result.converged:
                return False
            cr = r.coupled_result
            return (
                r.topology.pore1_side == top_row["topology_pore1_side"]
                and r.topology.pore2_side == top_row["topology_pore2_side"]
                and abs(cr.aero_result.CL - top_row["CL"]) < 1e-12
                and abs(cr.network_state.Q - top_row["Q_m3_s"]) < 1e-12
            )

        matched = [r for r in all_results if matches_top_row(r)]
        best = matched[0] if matched else max(feasible, key=lambda r: r.coupled_result.aero_result.CL)
    else:
        best = max(feasible, key=lambda r: r.coupled_result.aero_result.CL)

    best_network = optimizer.design_to_network(best.x_opt, best.topology)
    t1 = time.perf_counter()

    print()
    print("=" * 72)
    print("OPTIMIZATION SUMMARY")
    print("=" * 72)
    for r in all_results:
        converged = r.coupled_result.converged if r.coupled_result is not None else False
        print(
            f"Topology ({r.topology.pore1_side}, {r.topology.pore2_side}) | "
            f"success={r.success} | converged={converged} | "
            f"fun={r.best_fun:.8f} | message={r.message}"
        )

    print()
    print("=" * 72)
    print("DESIGNS WITH LIFT GAIN ABOVE 5%")
    print("=" * 72)

    if gain_shortlist_df.empty:
        print("No converged designs achieved more than 5% lift gain.")
    else:
        for i, row in gain_shortlist_df.iterrows():
            print(
                f"[{i+1}] Topology ({row['topology_pore1_side']}, {row['topology_pore2_side']}) | "
                f"CL={row['CL']:.8f} | "
                f"CD={row['CD']:.8e} | "
                f"gain={row['gain_percent']:.3f}% | "
                f"Q={row['Q_m3_s']:.8e} | "
                f"max_vn={row['max_vn_m_s']:.6e} m/s | "
                f"max_vn/Vinf={row['max_vn_over_vinf']:.6f}"
            )

    print()
    print(f"[Run] Total elapsed time = {t1 - t0:.3f} s")
    print(
        f"[Run] Best lift gain over solid baseline: "
        f"{100.0 * (best.coupled_result.aero_result.CL / solid_result.CL - 1.0):.3f}%"
    )
    print()

    print_best_design_summary(best, best_network, geom)

    xfoil_point = None
    xfoil_cp_data = None

    try:
        print()
        print("[XFOIL] Running XFOIL solid-airfoil baseline...")
        polar_path, xfoil_point = run_and_load_xfoil_point(
            xfoil_exe_path=xfoil_exe_path,
            airfoil_name=airfoil_name,
            reynolds=reynolds,
            aoa_deg=aoa_deg,
            output_dir=xfoil_dir,
            mach=xfoil_mach,
            timeout=xfoil_timeout,
        )
        print(f"[XFOIL] Polar file created: {polar_path}")
        print("[XFOIL] Comparison point loaded successfully.")
    except Exception as exc:
        print(f"[XFOIL] Automatic XFOIL baseline run failed: {exc}")

    try:
        print("[XFOIL] Running XFOIL Cp export...")
        cp_path = run_xfoil_cp(
            xfoil_exe_path=xfoil_exe_path,
            airfoil_name=airfoil_name,
            aoa_deg=aoa_deg,
            output_dir=xfoil_dir,
            mach=xfoil_mach,
            timeout=xfoil_timeout,
        )
        xfoil_cp_data = load_xfoil_cp(cp_path)
        print(f"[XFOIL] Cp file created: {cp_path}")
    except Exception as exc:
        print(f"[XFOIL] Cp export failed: {exc}")

    print()
    print_xfoil_comparison(best.coupled_result.aero_result, xfoil_point)

    sweep_df = None
    try:
        print()
        print("[Sweep] Running porous + solid panel AoA sweep...")
        sweep_df = run_porous_aoa_sweep(
            airfoil_name=airfoil_name,
            chord=chord,
            n_panels=n_panels,
            rho_inf=rho_inf,
            mu_inf=mu_inf,
            p_inf=p_inf,
            reynolds=reynolds,
            best_network=best_network,
            coupling=coupling,
            alpha_start=alpha_sweep_start,
            alpha_end=alpha_sweep_end,
            alpha_step=alpha_sweep_step,
        )

        print("[Sweep] Panel-only sweep columns:", list(sweep_df.columns))

        print("[Sweep] Running XFOIL AoA sweep...")
        xfoil_sweep_path = run_xfoil_polar_sweep(
            xfoil_exe_path=xfoil_exe_path,
            airfoil_name=airfoil_name,
            reynolds=reynolds,
            alpha_start=alpha_sweep_start,
            alpha_end=alpha_sweep_end,
            alpha_step=alpha_sweep_step,
            output_dir=xfoil_dir,
            mach=xfoil_mach,
            timeout=120.0,
        )

        xfoil_sweep_df = load_xfoil_polar_dataframe(xfoil_sweep_path)
        print("[Sweep] XFOIL columns:", list(xfoil_sweep_df.columns))

        sweep_df = add_xfoil_to_sweep_comparison(sweep_df, xfoil_sweep_df)

        print("[Sweep] Combined sweep columns:", list(sweep_df.columns))
        print("[Sweep] AoA sweep comparison table built successfully.")

    except Exception as exc:
        print(f"[Sweep] AoA sweep comparison failed: {exc}")

    print()
    print("[CSV] Exporting result tables...")

    gain_shortlist_path = csv_dir / "designs_above_5_percent_lift_gain.csv"
    gain_shortlist_df.to_csv(gain_shortlist_path, index=False)
    print(f"[CSV Saved] {gain_shortlist_path}")

    export_results_to_csv(
        output_dir=csv_dir,
        geom=geom,
        best_result=best,
        best_network=best_network,
        reynolds=reynolds,
        xfoil_point=xfoil_point,
        xfoil_cp_data=xfoil_cp_data,
        sweep_df=sweep_df,
    )

    print()
    print("[Plot] Saving airfoil with porous network...")
    plot_airfoil_with_pores(
        geom=geom,
        network=best_network,
        output_dir=plot_dir,
        coupled_result=best.coupled_result,
    )

    print("[Plot] Saving Cp comparison plot...")
    plot_cp_distribution(
        geom=geom,
        porous_result=best.coupled_result.aero_result,
        output_dir=plot_dir,
        xfoil_cp_data=xfoil_cp_data,
    )

    print("[Plot] Saving velocity contour comparison...")
    plot_velocity_contours_comparison(
        porous_solver=aero_solver,
        porous_result=best.coupled_result.aero_result,
        solid_solver=aero_solver,
        solid_result=solid_result,
        output_dir=plot_dir,
    )

    print("[Plot] Saving pressure contour comparison...")
    plot_pressure_contours_comparison(
        porous_solver=aero_solver,
        porous_result=best.coupled_result.aero_result,
        solid_solver=aero_solver,
        solid_result=solid_result,
        output_dir=plot_dir,
    )

    print("[Plot] Saving internal flow direction plot...")
    plot_internal_flow_direction(
        geom=geom,
        network=best_network,
        coupled_result=best.coupled_result,
        output_dir=plot_dir,
    )

    if sweep_df is not None:
        print("[Plot] Saving AoA sweep comparison plots...")
        plot_aoa_sweep_comparison(sweep_df, plot_dir)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    run()