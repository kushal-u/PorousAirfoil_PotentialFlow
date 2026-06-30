"""
run_test.py

Run a fixed set of saved porous-network test cases and compare each one against:
1. the solid airfoil solved by the in-house source-vortex panel method
2. optionally, the solid airfoil solved by XFOIL

This script is intended for validation of shortlisted porous designs rather than
for optimisation.

What this script does
---------------------
For each saved case:
- rebuild the porous network from fixed parameters
- solve the coupled porous-airfoil problem
- solve the solid airfoil using the panel method
- compare porous vs solid panel-method results
- optionally run XFOIL point and Cp comparisons
- run porous / solid / XFOIL AoA sweeps
- save plots and CSV outputs

Outputs per case
----------------
Each case gets its own folder containing:
- Cp comparison plot
- velocity contour comparison
- pressure contour comparison
- airfoil geometry with pores
- AoA sweep CSV
- AoA sweep plot
- per-case summary CSV

A combined CSV for all cases is also saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solver import (
    CouplingConfig,
    FlowConfig,
    PanelGeometry,
    ReferenceConfig,
    SourceVortexPanelMethod,
    generate_naca4,
)
from porous_network_optimisation import (
    Chamber,
    CoupledPorousAirfoilSolver,
    Pore,
    Topology,
    TwoPoreOneChamberNetwork,
)
from plotter import (
    plot_airfoil_with_pores,
    plot_aoa_sweep_comparison,
    plot_cp_distribution,
    plot_pressure_contours_comparison,
    plot_velocity_contours_comparison,
    print_best_design_summary,
    print_xfoil_comparison,
)
from xfoil import (
    load_xfoil_cp,
    load_xfoil_polar_dataframe,
    run_and_load_xfoil_point,
    run_xfoil_cp,
    run_xfoil_polar_sweep,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================
XFOIL_EXE_PATH = r"C:\Users\kusha\OneDrive\Desktop\airfoil_rewrite\xfoil.exe"
USE_XFOIL = True

AIRFOIL_NAME = "0018"
CHORD = 1.0
N_PANELS = 3000

REYNOLDS_EXTERNAL = 5.0e5
AOA_DEG = 4.0

RHO_INF = 1.225
MU_INF = 1.8e-5
P_INF = 101325.0

ALPHA_SWEEP_START = -5.0
ALPHA_SWEEP_END = 12.0
ALPHA_SWEEP_STEP = 1.0

SPAN_WIDTH = 0.02

COUPLING = CouplingConfig(
    max_iter=60,
    tol_vn=1e-6,
    tol_q=1e-11,
    relaxation=0.5,
)


# =============================================================================
# SAVED TEST CASES
# =============================================================================
@dataclass
class SavedCase:
    """
    Fixed porous-network design to re-run.

    Parameters
    ----------
    case_name : str
        Folder-friendly case name.
    topology : Topology
        Pore-side topology.
    x1_frac : float
        Chordwise location of pore 1 as x/c.
    x2_frac : float
        Chordwise location of pore 2 as x/c.
    d1_m : float
        Pore 1 diameter in metres.
    d2_m : float
        Pore 2 diameter in metres.
    dh_m : float
        Effective internal hydraulic diameter in metres.
    q_gain_percent_ref : float
        Reference gain taken from a previous optimisation result or CSV.
    """

    case_name: str
    topology: Topology
    x1_frac: float
    x2_frac: float
    d1_m: float
    d2_m: float
    dh_m: float
    q_gain_percent_ref: float


CASES: list[SavedCase] = [
    # SavedCase(
    #     case_name="case_01_lower_lower_gain_5p8",
    #     topology=Topology("lower", "lower"),
    #     x1_frac=0.0481177545426265,
    #     x2_frac=0.9381729053922714,
    #     d1_m=0.018567805729273,
    #     d2_m=0.0191053393230789, 
    #     dh_m=0.0181438476021222,
    #     q_gain_percent_ref=5.8,
    # ),
    SavedCase(
        case_name="case_02_upper_upper_gain_8p1",
        topology=Topology("upper", "upper"),
        x1_frac=0.0100298966699871,
        x2_frac=0.9445559176683432,
        d1_m=0.0196404343706552,
        d2_m=0.0195645943848706,
        dh_m=0.0195645943848706,
        q_gain_percent_ref=8.050602808199624,
    ),
    # SavedCase(
    #     case_name="case_03_lower_upper_gain_7p4",
    #     topology=Topology("lower", "upper"),
    #     x1_frac=0.70932901647652,
    #     x2_frac=0.939033402877522,
    #     d1_m=0.0186527084426661,
    #     d2_m=0.0184883649705853,
    #     dh_m=0.010570396062818,
    #     q_gain_percent_ref=7.4,
    # ),
]


# =============================================================================
# HELPERS
# =============================================================================
def reynolds_to_velocity(reynolds: float, rho: float, mu: float, chord: float) -> float:
    """
    Convert Reynolds number to freestream velocity.

    Parameters
    ----------
    reynolds : float
        Reynolds number based on chord.
    rho : float
        Fluid density [kg/m^3].
    mu : float
        Dynamic viscosity [Pa·s].
    chord : float
        Chord length [m].

    Returns
    -------
    float
        Freestream velocity [m/s].

    Notes
    -----
    Uses:
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


def build_network_from_case(case: SavedCase, span_width: float) -> TwoPoreOneChamberNetwork:
    """
    Rebuild a porous network from one saved case definition.

    Parameters
    ----------
    case : SavedCase
        Saved porous-network case.
    span_width : float
        Spanwise width used to distribute transpiration.

    Returns
    -------
    TwoPoreOneChamberNetwork
        Reconstructed porous network.
    """
    pore1 = Pore(
        x_frac=case.x1_frac,
        side=case.topology.pore1_side,
        diameter=case.d1_m,
    )
    pore2 = Pore(
        x_frac=case.x2_frac,
        side=case.topology.pore2_side,
        diameter=case.d2_m,
    )

    chamber = Chamber(
        hydraulic_diameter=case.dh_m,
        area=0.25 * np.pi * case.dh_m**2,
        friction_scale=1.0,
        geometry_type="any",
    )

    return TwoPoreOneChamberNetwork(
        pore1=pore1,
        pore2=pore2,
        chamber=chamber,
        span_width=span_width,
    )


def build_base_problem(
    airfoil_name: str,
    chord: float,
    n_panels: int,
    aoa_deg: float,
    reynolds_external: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
) -> tuple[FlowConfig, PanelGeometry, ReferenceConfig, SourceVortexPanelMethod]:
    """
    Build the common solid-airfoil aerodynamic problem.

    Parameters
    ----------
    airfoil_name : str
        NACA 4-digit airfoil identifier.
    chord : float
        Chord length [m].
    n_panels : int
        Number of panels on the airfoil boundary.
    aoa_deg : float
        Angle of attack [deg].
    reynolds_external : float
        External Reynolds number based on chord.
    rho_inf : float
        Freestream density [kg/m^3].
    mu_inf : float
        Freestream dynamic viscosity [Pa·s].
    p_inf : float
        Freestream static pressure [Pa].

    Returns
    -------
    tuple[FlowConfig, PanelGeometry, ReferenceConfig, SourceVortexPanelMethod]
        Flow configuration, geometry, reference configuration, and solver.
    """
    v_inf = reynolds_to_velocity(reynolds_external, rho_inf, mu_inf, chord)

    flow = FlowConfig(
        aoa_deg=aoa_deg,
        v_inf=v_inf,
        rho_inf=rho_inf,
        mu_inf=mu_inf,
        p_inf=p_inf,
    )

    XB, YB = generate_naca4(airfoil_name, n_panels, chord=chord)
    geom = PanelGeometry(XB, YB, aoa_deg)

    ref = ReferenceConfig(
        x_ref=0.25 * chord,
        y_ref=0.0,
    )

    aero_solver = SourceVortexPanelMethod(geom, flow, ref)
    return flow, geom, ref, aero_solver


def run_porous_aoa_sweep_for_case(
    airfoil_name: str,
    chord: float,
    n_panels: int,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    reynolds_external: float,
    network: TwoPoreOneChamberNetwork,
    coupling: CouplingConfig,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
) -> pd.DataFrame:
    """
    Run an AoA sweep for one porous design and compute both porous and solid
    panel-method results.

    Parameters
    ----------
    airfoil_name : str
        NACA 4-digit airfoil identifier.
    chord : float
        Chord length [m].
    n_panels : int
        Number of panels.
    rho_inf : float
        Freestream density [kg/m^3].
    mu_inf : float
        Freestream dynamic viscosity [Pa·s].
    p_inf : float
        Freestream static pressure [Pa].
    reynolds_external : float
        External Reynolds number based on chord.
    network : TwoPoreOneChamberNetwork
        Porous network to evaluate.
    coupling : CouplingConfig
        Coupling solver settings.
    alpha_start : float
        Start angle of attack [deg].
    alpha_end : float
        End angle of attack [deg].
    alpha_step : float
        AoA step size [deg].

    Returns
    -------
    pandas.DataFrame
        Table with columns:
        - alpha_deg
        - porous_CL, porous_CD, porous_CM
        - solid_panel_CL, solid_panel_CD, solid_panel_CM
    """
    rows: list[dict[str, float]] = []
    v_inf = reynolds_to_velocity(reynolds_external, rho_inf, mu_inf, chord)

    for alpha in np.arange(alpha_start, alpha_end + 0.1, alpha_step):
        flow_i = FlowConfig(
            aoa_deg=float(alpha),
            v_inf=v_inf,
            rho_inf=rho_inf,
            mu_inf=mu_inf,
            p_inf=p_inf,
        )

        ref_i = ReferenceConfig(
            x_ref=0.25 * chord,
            y_ref=0.0,
        )

        XB_i, YB_i = generate_naca4(airfoil_name, n_panels, chord=chord)
        geom_i = PanelGeometry(XB_i, YB_i, flow_i.aoa_deg)
        aero_solver_i = SourceVortexPanelMethod(geom_i, flow_i, ref_i)

        solid_i = aero_solver_i.solve()

        coupled_i = CoupledPorousAirfoilSolver(
            aero_solver=aero_solver_i,
            network=network,
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


def add_xfoil_to_sweep_comparison(
    sweep_df: pd.DataFrame,
    xfoil_sweep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge XFOIL AoA sweep data into an existing comparison dataframe.

    Accepted input column styles include variants such as:
    - alpha_deg / alpha / Alpha
    - CL / cl
    - CD / cd
    - CM / cm
    """
    xfoil_df = xfoil_sweep_df.copy()

    # normalize raw column names
    xfoil_df.columns = [str(c).strip() for c in xfoil_df.columns]

    rename_map = {}

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


def build_case_summary_row(
    case: SavedCase,
    reynolds_external: float,
    flow: FlowConfig,
    solid_result,
    coupled_result,
) -> dict[str, Any]:
    """
    Build one summary CSV row for one test case.

    Parameters
    ----------
    case : SavedCase
        Saved porous-network case.
    reynolds_external : float
        External Reynolds number.
    flow : FlowConfig
        Freestream flow configuration.
    solid_result
        Solid-airfoil panel-method result.
    coupled_result
        Coupled porous-airfoil result.

    Returns
    -------
    dict[str, Any]
        One summary row for CSV export.
    """
    aero = coupled_result.aero_result
    state = coupled_result.network_state

    return {
        "case_name": case.case_name,
        "topology_pore1_side": case.topology.pore1_side,
        "topology_pore2_side": case.topology.pore2_side,
        "reynolds_external": float(reynolds_external),
        "v_inf_m_s": float(flow.v_inf),
        "aoa_deg": float(flow.aoa_deg),
        "x1_frac": float(case.x1_frac),
        "x2_frac": float(case.x2_frac),
        "d1_m": float(case.d1_m),
        "d2_m": float(case.d2_m),
        "effective_hydraulic_diameter_m": float(case.dh_m),
        "reference_gain_percent_from_csv": float(case.q_gain_percent_ref),
        "solid_panel_CL": float(solid_result.CL),
        "solid_panel_CD": float(solid_result.CD),
        "solid_panel_CM": float(solid_result.CM),
        "porous_CL": float(aero.CL),
        "porous_CD": float(aero.CD),
        "porous_CM": float(aero.CM),
        "gain_percent_vs_solid_panel": float(100.0 * (aero.CL / solid_result.CL - 1.0)),
        "delta_CL_vs_solid_panel": float(aero.CL - solid_result.CL),
        "delta_CD_vs_solid_panel": float(aero.CD - solid_result.CD),
        "delta_CM_vs_solid_panel": float(aero.CM - solid_result.CM),
        "Q_m3_s": float(state.Q),
        "Rs_Pa_s_per_m3": float(state.Rs),
        "dp_total_Pa": float(state.dp_total),
        "reynolds_internal_equivalent": float(state.reynolds_equivalent),
        "max_vn_m_s": float(coupled_result.max_vn),
        "max_vn_over_vinf": float(coupled_result.max_vn_over_vinf),
        "coupling_converged": bool(coupled_result.converged),
        "coupling_iterations": int(coupled_result.iterations),
    }


# =============================================================================
# MAIN CASE RUNNER
# =============================================================================
def run_one_case(
    case: SavedCase,
    geom: PanelGeometry,
    flow: FlowConfig,
    solid_result,
    aero_solver: SourceVortexPanelMethod,
    coupling: CouplingConfig,
    reynolds_external: float,
    output_root: Path,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    use_xfoil: bool,
    xfoil_exe_path: str,
) -> dict[str, Any]:
    """
    Run one saved porous design and save all outputs.

    Parameters
    ----------
    case : SavedCase
        Saved porous-network case.
    geom : PanelGeometry
        Base airfoil geometry.
    flow : FlowConfig
        Base freestream configuration.
    solid_result
        Solid-airfoil panel-method baseline result.
    aero_solver : SourceVortexPanelMethod
        Base aerodynamic solver.
    coupling : CouplingConfig
        Coupling settings.
    reynolds_external : float
        External Reynolds number.
    output_root : Path
        Root directory for per-case outputs.
    airfoil_name : str
        NACA airfoil name.
    chord : float
        Chord length [m].
    n_panels : int
        Number of panels.
    use_xfoil : bool
        Whether to run optional XFOIL comparisons.
    xfoil_exe_path : str
        Full path to XFOIL executable.

    Returns
    -------
    dict[str, Any]
        Per-case summary row.
    """
    case_dir = output_root / case.case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    network = build_network_from_case(case, span_width=SPAN_WIDTH)

    coupled = CoupledPorousAirfoilSolver(
        aero_solver=aero_solver,
        network=network,
        coupling=coupling,
        opt=None,
    ).solve(verbose=True)

    print()
    print("=" * 72)
    print(f"RUNNING {case.case_name}")
    print("=" * 72)
    print(f"Reynolds number          : {reynolds_external:.6e}")
    print(f"Freestream velocity [m/s]: {flow.v_inf:.6f}")
    print(f"Solid panel-method CL    : {solid_result.CL:.8f}")
    print(f"Lift gain [%]            : {100.0 * (coupled.aero_result.CL / solid_result.CL - 1.0):.6f}")
    print(f"Internal Re_eq           : {coupled.network_state.reynolds_equivalent:.6f}")

    temp_best = type(
        "TempBest",
        (),
        {
            "topology": case.topology,
            "coupled_result": coupled,
        },
    )()

    print_best_design_summary(temp_best, network, geom)

    xfoil_point = None
    xfoil_cp_data = None

    # ------------------------------------------------------------------
    # XFOIL comparison for this case (optional)
    # ------------------------------------------------------------------
    if use_xfoil:
        try:
            print("[XFOIL] Running solid-airfoil point comparison...")
            polar_path, xfoil_point = run_and_load_xfoil_point(
                xfoil_exe_path=xfoil_exe_path,
                airfoil_name=airfoil_name,
                reynolds=reynolds_external,
                aoa_deg=flow.aoa_deg,
                output_dir=case_dir,
                mach=0.0,
                timeout=60.0,
            )
            print(f"[XFOIL] Polar file created: {polar_path}")
        except Exception as exc:
            print(f"[XFOIL] Point comparison failed: {exc}")

        try:
            print("[XFOIL] Running Cp export...")
            cp_path = run_xfoil_cp(
                xfoil_exe_path=xfoil_exe_path,
                airfoil_name=airfoil_name,
                aoa_deg=flow.aoa_deg,
                output_dir=case_dir,
                mach=0.0,
                timeout=60.0,
            )
            xfoil_cp_data = load_xfoil_cp(cp_path)
            print(f"[XFOIL] Cp file created: {cp_path}")
        except Exception as exc:
            print(f"[XFOIL] Cp export failed: {exc}")

        print()
        print_xfoil_comparison(coupled.aero_result, xfoil_point)

    # ------------------------------------------------------------------
    # Per-case plots
    # ------------------------------------------------------------------
    plot_airfoil_with_pores(
        geom=geom,
        network=network,
        output_dir=case_dir,
        coupled_result=coupled,
    )

    plot_cp_distribution(
        geom=geom,
        porous_result=coupled.aero_result,
        output_dir=case_dir,
        xfoil_cp_data=xfoil_cp_data,
    )

    plot_velocity_contours_comparison(
        porous_solver=aero_solver,
        porous_result=coupled.aero_result,
        solid_solver=aero_solver,
        solid_result=solid_result,
        output_dir=case_dir,
    )

    plot_pressure_contours_comparison(
        porous_solver=aero_solver,
        porous_result=coupled.aero_result,
        solid_solver=aero_solver,
        solid_result=solid_result,
        output_dir=case_dir,
    )

    # ------------------------------------------------------------------
    # AoA sweep: porous + solid panel
    # ------------------------------------------------------------------
    print("[Sweep] Running porous vs solid panel-method AoA sweep...")
    sweep_df = run_porous_aoa_sweep_for_case(
        airfoil_name=airfoil_name,
        chord=chord,
        n_panels=n_panels,
        rho_inf=flow.rho_inf,
        mu_inf=flow.mu_inf,
        p_inf=flow.p_inf,
        reynolds_external=reynolds_external,
        network=network,
        coupling=coupling,
        alpha_start=ALPHA_SWEEP_START,
        alpha_end=ALPHA_SWEEP_END,
        alpha_step=ALPHA_SWEEP_STEP,
    )

    # ------------------------------------------------------------------
    # Optional XFOIL AoA sweep
    # ------------------------------------------------------------------
    if use_xfoil:
        try:
            print("[Sweep] Running XFOIL AoA sweep...")
            xfoil_sweep_path = run_xfoil_polar_sweep(
                xfoil_exe_path=xfoil_exe_path,
                airfoil_name=airfoil_name,
                reynolds=reynolds_external,
                alpha_start=ALPHA_SWEEP_START,
                alpha_end=ALPHA_SWEEP_END,
                alpha_step=ALPHA_SWEEP_STEP,
                output_dir=case_dir,
                mach=0.0,
                timeout=120.0,
            )

            xfoil_sweep_df = load_xfoil_polar_dataframe(xfoil_sweep_path)
            sweep_df = add_xfoil_to_sweep_comparison(sweep_df, xfoil_sweep_df)

            xfoil_sweep_csv = case_dir / "aoa_sweep_xfoil.csv"
            xfoil_sweep_df.to_csv(xfoil_sweep_csv, index=False)
            print(f"[CSV Saved] {xfoil_sweep_csv}")

        except Exception as exc:
            print(f"[Sweep] XFOIL AoA sweep failed: {exc}")

    sweep_csv = case_dir / "aoa_sweep_comparison.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"[CSV Saved] {sweep_csv}")

    plot_aoa_sweep_comparison(sweep_df, case_dir)

    # ------------------------------------------------------------------
    # Per-case summary CSV
    # ------------------------------------------------------------------
    row = build_case_summary_row(
        case=case,
        reynolds_external=reynolds_external,
        flow=flow,
        solid_result=solid_result,
        coupled_result=coupled,
    )

    if xfoil_point is not None:
        row["xfoil_alpha_deg"] = float(xfoil_point.alpha)
        row["xfoil_CL"] = float(xfoil_point.CL)
        row["xfoil_CD"] = float(xfoil_point.CD)
        row["xfoil_CM"] = float(xfoil_point.CM)
        row["delta_CL_vs_xfoil"] = float(coupled.aero_result.CL - xfoil_point.CL)
        row["delta_CD_vs_xfoil"] = float(coupled.aero_result.CD - xfoil_point.CD)
        row["delta_CM_vs_xfoil"] = float(coupled.aero_result.CM - xfoil_point.CM)

    pd.DataFrame([row]).to_csv(case_dir / "case_summary.csv", index=False)
    print(f"[CSV Saved] {case_dir / 'case_summary.csv'}")

    return row


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """
    Run all saved validation cases.

    Returns
    -------
    None
    """
    output_root = Path.cwd() / "run_test_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    flow, geom, _, aero_solver = build_base_problem(
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=N_PANELS,
        aoa_deg=AOA_DEG,
        reynolds_external=REYNOLDS_EXTERNAL,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
    )

    solid_result = aero_solver.solve()

    all_rows: list[dict[str, Any]] = []

    for case in CASES:
        row = run_one_case(
            case=case,
            geom=geom,
            flow=flow,
            solid_result=solid_result,
            aero_solver=aero_solver,
            coupling=COUPLING,
            reynolds_external=REYNOLDS_EXTERNAL,
            output_root=output_root,
            airfoil_name=AIRFOIL_NAME,
            chord=CHORD,
            n_panels=N_PANELS,
            use_xfoil=USE_XFOIL,
            xfoil_exe_path=XFOIL_EXE_PATH,
        )
        all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(output_root / "all_cases_summary.csv", index=False)

    print()
    print(f"[CSV Saved] {output_root / 'all_cases_summary.csv'}")
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()