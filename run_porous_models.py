"""Command-line entry point for the fixed porous-airfoil model study.

This script reads all user-editable settings from porous_config.py, builds the
solid-airfoil panel-method baseline, runs the selected porous model(s), and
writes CSV, plot, XFOIL-comparison, and ParaView outputs.

Typical usage:
    python run_porous_models.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from porous_config import (
    AIRFOIL_NAME,
    AOA_DEG,
    CHORD,
    CONTOUR_NX,
    CONTOUR_NY,
    MAKE_CONTOUR_PLOTS,
    MU_INF,
    N_PANELS,
    OUTPUT_ROOT,
    P_INF,
    PORE_DIAMETER,
    REYNOLDS_EXTERNAL,
    RHO_INF,
    SELECTED_MODEL,
    USE_XFOIL,
)
from paraview_export import export_solid_reference_paraview_files
from porous_config import PorousModelSpec
from porous_core import build_base_problem, build_model_specs, run_one_model

# Valid model names accepted by porous_config.SELECTED_MODEL.
# Keep this set synchronized with build_model_specs() in porous_core.py.
VALID_MODEL_NAMES = {
    "all",
    "model_1_9_chordwise",
    "model_2_9_perpendicular",
    "model_3_combined_independent",
    "model_4_saved_case_1",
}


def select_models(models: list[PorousModelSpec]) -> list[PorousModelSpec]:
    """Return the model list requested by porous_config.SELECTED_MODEL.

    SELECTED_MODEL is intentionally read only from porous_config.py. This keeps
    the run script predictable for GitHub users: changing one configuration value
    controls whether all models or a single named model is executed.
    """
    if SELECTED_MODEL not in VALID_MODEL_NAMES:
        raise ValueError(
            f"Invalid SELECTED_MODEL={SELECTED_MODEL!r}. "
            f"Choose one of: {sorted(VALID_MODEL_NAMES)}"
        )

    if SELECTED_MODEL == "all":
        return models

    selected = [model for model in models if model.name == SELECTED_MODEL]

    if not selected:
        available = [model.name for model in models]
        raise RuntimeError(
            f"No model matched SELECTED_MODEL={SELECTED_MODEL!r}. "
            f"Available models: {available}"
        )

    return selected


def main() -> None:
    """Run the fixed porous-airfoil cases using values from porous_config.py."""
    n_panels = int(N_PANELS)
    contour_nx = int(CONTOUR_NX)
    contour_ny = int(CONTOUR_NY)
    make_contours = bool(MAKE_CONTOUR_PLOTS)

    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RUNNING FIXED POROUS MODELS — PANEL METHOD + XFOIL")
    print("=" * 80)
    print(f"Airfoil                  : NACA {AIRFOIL_NAME}")
    print(f"AoA [deg]                : {AOA_DEG}")
    print(f"Reynolds number          : {REYNOLDS_EXTERNAL:.6e}")
    print(f"Panels                   : {n_panels}")
    print(f"Selected model           : {SELECTED_MODEL}")
    print(f"Common pore diameter [m] : {PORE_DIAMETER:.6e}")
    print(f"Output root              : {output_root}")
    print(f"Contour plots            : {make_contours}")
    print(f"Contour grid             : {contour_nx} x {contour_ny}")
    print(f"XFOIL enabled            : {USE_XFOIL}")

    flow, geom, _, aero_solver = build_base_problem(
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=n_panels,
        aoa_deg=AOA_DEG,
        reynolds_external=REYNOLDS_EXTERNAL,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
    )

    print(f"Freestream speed [m/s]   : {flow.v_inf:.8f}")
    print("[Setup] Solving solid-airfoil baseline...")

    solid_result = aero_solver.solve()

    print(f"[Setup] Solid CL          : {solid_result.CL:.8f}")
    print(f"[Setup] Solid CD          : {solid_result.CD:.8e}")
    print(f"[Setup] Solid CM          : {solid_result.CM:.8f}")

    export_solid_reference_paraview_files(
        output_root=output_root,
        geom=geom,
        aero_solver=aero_solver,
        solid_result=solid_result,
        field_nx=contour_nx,
        field_ny=contour_ny,
    )

    models = select_models(
        build_model_specs(
            geom=geom,
            diameter_m=PORE_DIAMETER,
        )
    )

    rows = []

    for model in models:
        row = run_one_model(
            model=model,
            geom=geom,
            flow=flow,
            aero_solver=aero_solver,
            solid_result=solid_result,
            output_root=output_root,
            make_contours=make_contours,
            contour_nx=contour_nx,
            contour_ny=contour_ny,
        )
        rows.append(row)

    combined_path = output_root / "all_models_summary.csv"
    pd.DataFrame(rows).to_csv(combined_path, index=False)

    print()
    print(f"[CSV Saved] {combined_path}")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
