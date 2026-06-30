# Python file reference

This document describes the purpose of each Python file, the main classes/functions it contains, and where a new user should make changes.

## `run.py`

**Role:** Convenience entry point.

This file is intentionally small. It imports `main()` from `run_porous_models.py` and executes it when the file is run directly.

Use it when you want the conventional command:

```bash
python run.py
```

Important contents:

- `main`: imported from `run_porous_models.py`.
- `if __name__ == "__main__"`: calls `main()`.

## `run_porous_models.py`

**Role:** Main executable for the current fixed porous-model workflow.

This script reads user settings from `porous_config.py`, builds the solid-airfoil baseline, selects one or more porous model definitions, runs each model, and saves a combined summary table.

Important contents:

- `VALID_MODEL_NAMES`: allowed strings for `porous_config.SELECTED_MODEL`.
- `select_models(models)`: filters the full model list to either all models or one selected model.
- `main()`: orchestrates the full run:
  - reads panel count, contour resolution, and output folder;
  - builds the aerodynamic problem using `build_base_problem()`;
  - solves the solid-airfoil baseline;
  - exports the solid ParaView reference;
  - builds porous model definitions;
  - runs each selected model with `run_one_model()`;
  - writes `all_models_summary.csv`.

GitHub-readiness change made here:

- `SELECTED_MODEL` is no longer hard-coded in this runner. It is read from `porous_config.py`, so changing the config file now controls which model runs.

## `porous_config.py`

**Role:** User-editable configuration plus shared dataclasses.

This file should be the first place users edit values. It contains physical settings, solver/coupling settings, output settings, XFOIL setup, and compact data classes shared by the porous model code.

Main settings:

- XFOIL environment handling:
  - `USE_XFOIL`
  - `XFOIL_EXE_PATH`
  - `XFOIL_FOLDER`
  - `XFOIL_OUTPUT_ROOT`
- Run settings:
  - `OUTPUT_ROOT`
  - `SELECTED_MODEL`
  - `AIRFOIL_NAME`
  - `CHORD`
  - `N_PANELS`
- Flow settings:
  - `REYNOLDS_EXTERNAL`
  - `AOA_DEG`
  - `RHO_INF`
  - `MU_INF`
  - `P_INF`
- Porous-channel settings:
  - `SPAN_WIDTH`
  - `PORE_DIAMETER`
  - `DEFAULT_PORE_DIAMETER`
- Coupling and plotting settings:
  - `COUPLING`
  - `MAKE_CONTOUR_PLOTS`
  - `CONTOUR_NX`
  - `CONTOUR_NY`
  - `AOA_SWEEP_START_DEG`
  - `AOA_SWEEP_END_DEG`
  - `AOA_SWEEP_STEP_DEG`
  - `EXPORT_AOA_SWEEP_PARAVIEW`

Important dataclasses:

- `GeometryLimits`: geometric feasibility limits for pores and internal channels.
- `Pore`: one surface pore location, side, and diameter.
- `Chamber`: hydraulic diameter and cross-sectional area of a passage.
- `PassageState`: solved internal-flow state for one passage.
- `PassageSpec`: model-level definition of one passage.
- `PorousModelSpec`: named porous model containing multiple passages.
- `MultiPassageResult`: coupled solver result for one porous model.

Important helper functions:

- `find_xfoil_executable()`: finds XFOIL from `XFOIL_EXE`, PATH, or `XFOIL_FOLDER`.
- `reynolds_to_velocity()`: converts chord Reynolds number to freestream velocity.
- `circular_laminar_resistance()`: computes circular-channel Poiseuille resistance and area.

## `porous_core.py`

**Role:** Porous-network physics, model construction, coupling, CSV export, and AoA sweep logic.

This is the main physics/control module for the porous part of the fixed-model study.

Main classes:

- `IndependentPassage`: represents one internal hydraulic passage connecting two surface pores.
  - Maps pore chord fractions to panel surface points.
  - Computes passage length.
  - Validates geometry.
  - Computes pressure-driven flow rate using hydraulic resistance.
  - Converts flow rate into panel normal transpiration velocity.
- `IndependentPassageNetwork`: collection of independent passages.
  - Validates pore spacing and overlap.
  - Solves all passage flow states.
  - Sums normal transpiration from all passages.
- `MultiPassageCoupledSolver`: fixed-point coupling between panel pressures and internal passage flows.
  - Starts with zero transpiration.
  - Solves the aerodynamic panel problem.
  - Computes panel pressures.
  - Solves internal channel flow rates.
  - Updates transpiration with relaxation.
  - Stops when both transpiration and flow-rate changes meet tolerances.

Main model-builder functions:

- `model1_le_te_surface_pores()`: creates the LE/TE pore bands for Model 1.
- `build_model1_chordwise()`: creates 9 LE-to-TE channels.
- `build_vertical_passages()`: creates lower-to-upper perpendicular channels.
- `build_model4_saved_case1()`: creates the saved two-passage design.
- `build_model_specs()`: returns all supported `PorousModelSpec` objects.

Main run/export functions:

- `build_base_problem()`: builds flow, geometry, reference point, and aerodynamic solver.
- `build_network_from_model()`: converts a model specification into an executable passage network.
- `run_one_model()`: runs one fixed-AoA model and writes plots, CSVs, XFOIL comparison, AoA sweep, and ParaView files.
- `run_porous_aoa_sweep_for_model()`: computes porous and solid aerodynamic coefficients over the AoA range.
- `build_passage_dataframe()`: creates passage-level CSV data.
- `export_surface_data()`: writes panel-level surface CSV data.
- `build_model_summary_row()`: builds one row for `model_summary.csv` and `all_models_summary.csv`.

## `solver.py`

**Role:** Core aerodynamic geometry and source-vortex panel method.

This file is independent of the porous-network definitions. It provides the aerodynamic baseline used by both solid and porous cases.

Main dataclasses:

- `FlowConfig`: angle of attack, velocity, density, viscosity, and reference pressure.
- `AirfoilConfig`: airfoil name, chord, and panel count.
- `ReferenceConfig`: moment reference point.
- `CouplingConfig`: max iterations, convergence tolerances, and relaxation factor.
- `SurfacePoint`: surface location mapped to a panel.
- `SPVPResult`: solved aerodynamic quantities.

Main functions/classes:

- `generate_naca4()`: generates a closed NACA 4-digit airfoil boundary.
- `ensure_clockwise_boundary()`: ensures clockwise panel ordering.
- `PanelGeometry`: computes control points, panel lengths, tangents, normals, side mapping, and surface interpolation.
- `compute_IJKL_vectorized()`: computes the source/vortex influence matrices.
- `SourceVortexPanelMethod`: builds and solves the aerodynamic linear system.
  - `solve()`: returns `SPVPResult`.
  - `panel_pressures()`: converts Cp to absolute pressure.
  - `velocity_field()`: computes off-body velocity and Cp fields for plotting/export.

Performance note:

- If `numba` is installed, the velocity-field kernel uses JIT compilation and parallel loops.
- If `numba` is not installed, the NumPy fallback runs automatically.

## `plotter.py`

**Role:** Matplotlib figure generation for the fixed-model runner.

Main functions:

- `plot_airfoil_with_porous_network()`: airfoil outline, pore locations, internal channels, and flow-direction arrows.
- `plot_cp_distribution_with_xfoil()`: surface Cp comparison between porous panel, solid panel, and optional XFOIL.
- `plot_aoa_sweep_comparison()`: CL, CD, and CM versus angle of attack.
- `plot_velocity_contours_comparison()`: side-by-side porous/solid velocity magnitude and streamlines.
- `plot_pressure_contours_comparison()`: side-by-side porous/solid static pressure contours.
- `plot_difference_contours()`: porous-minus-solid velocity and pressure difference contours.

Important internal helpers:

- `_prepare_output_dir()`: creates plot output folders.
- `_sorted_side_ids()`: sorts upper/lower panels by x-position.
- `_build_field_grid()`: constructs the contour grid.
- `_velocity_magnitude_for_display()`: computes display velocity magnitude while masking the airfoil interior.
- `_pressure_for_display()`: computes pressure fields for contour plotting.

## `paraview_export.py`

**Role:** Writes ParaView-compatible VTK XML files without requiring the `vtk` Python package.

Main single-case exports:

- `paraview_airfoil_surface.vtp`: airfoil surface panels with Cp, velocity, and transpiration fields.
- `paraview_solid_airfoil_surface.vtp`: solid surface reference.
- `paraview_porous_network.vtp`: internal porous channels with pressure, flow rate, Reynolds number, and direction vectors.
- `paraview_flow_field_porous.vts`: porous velocity/pressure/Cp field.
- `paraview_flow_field_solid.vts`: optional solid field.
- `paraview_flow_field_delta.vts`: porous-minus-solid difference field.
- `paraview_results.pvd`: ParaView collection file.

Main functions:

- `export_airfoil_surface_vtp()`
- `export_solid_airfoil_surface_vtp()`
- `export_porous_network_vtp()`
- `export_flow_field_vts()`
- `export_flow_field_delta_vts()`
- `export_paraview_collection()`
- `export_solid_reference_paraview_files()`
- `export_paraview_files()`
- `export_aoa_sweep_collections()`

Important implementation notes:

- The writer uses ASCII VTK XML, which is easy to inspect and portable.
- AoA sweep `.pvd` files store angle of attack as the timestep, so ParaView can step through alpha values.

## `xfoil.py`

**Role:** Optional XFOIL automation and parser.

The repository can run without XFOIL. This file is used only when `USE_XFOIL = True` and an executable is found.

Main dataclass:

- `XFOILPolarPoint`: one parsed polar row containing alpha, CL, CD, and CM.

Main functions:

- `run_xfoil_polar()`: runs XFOIL at one angle of attack and saves a polar file.
- `load_xfoil_polar()`: parses an XFOIL polar file into `XFOILPolarPoint` objects.
- `nearest_xfoil_point()`: finds the polar point closest to a requested AoA.
- `run_and_load_xfoil_point()`: convenience wrapper for one polar point.
- `run_xfoil_cp()`: runs XFOIL and writes surface Cp data.
- `load_xfoil_cp()`: parses surface Cp data and separates upper/lower surfaces.
- `run_xfoil_polar_sweep()`: runs an AoA sweep with XFOIL.
- `load_xfoil_polar_dataframe()`: returns the sweep as a pandas DataFrame.

Important XFOIL assumptions:

- The provided commands are configured for inviscid, incompressible XFOIL comparison.
- `VISC` and `MACH` are intentionally not used in the command block.


