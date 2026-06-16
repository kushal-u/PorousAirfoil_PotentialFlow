# Configuration variable reference

All normal user edits should be made in `porous_config.py`.

## XFOIL variables

| Variable | Type | Meaning |
|---|---:|---|
| `USE_XFOIL` | `bool` | Enables optional XFOIL comparison. It can be disabled with environment variable `USE_XFOIL=0`. |
| `XFOIL_FOLDER` | `Path` | Optional folder to search when XFOIL is not found from `XFOIL_EXE` or PATH. Set with environment variable `XFOIL_FOLDER`. |
| `XFOIL_OUTPUT_ROOT` | `Path` | Folder for XFOIL Cp and polar text outputs. Set with `XFOIL_OUTPUT_ROOT` or defaults to `xfoil_outputs/`. |
| `XFOIL_MACH` | `float` | Mach number placeholder. Current XFOIL commands are inviscid/incompressible and do not issue `MACH`. |
| `XFOIL_TIMEOUT` | `float` | Maximum allowed XFOIL runtime per command, in seconds. |
| `XFOIL_EXE_PATH` | `Path | None` | Resolved XFOIL executable path after automatic search. |

## Run-selection variables

| Variable | Type | Meaning |
|---|---:|---|
| `OUTPUT_ROOT` | `Path` | Main output folder for all CSV, plot, and ParaView results. |
| `SELECTED_MODEL` | `str` | Model selector. Use `"all"` or one specific model name. |

Valid model names:

```text
all
model_1_9_chordwise
model_2_9_perpendicular
model_3_combined_independent
model_4_saved_case_1
```

## Geometry variables

| Variable | Type | Units | Meaning |
|---|---:|---:|---|
| `AIRFOIL_NAME` | `str` | — | NACA 4-digit code, for example `"0018"`. |
| `CHORD` | `float` | m | Airfoil chord length. |
| `N_PANELS` | `int` | — | Number of boundary panels. Higher values improve resolution but increase runtime. |

## External-flow variables

| Variable | Type | Units | Meaning |
|---|---:|---:|---|
| `REYNOLDS_EXTERNAL` | `float` | — | Reynolds number based on chord. |
| `AOA_DEG` | `float` | deg | Fixed-case angle of attack. |
| `RHO_INF` | `float` | kg/m³ | Freestream density. |
| `MU_INF` | `float` | Pa·s | Dynamic viscosity. |
| `P_INF` | `float` | Pa | Reference freestream static pressure. |

The code computes freestream speed from Reynolds number:

```text
V_inf = Re * mu / (rho * chord)
```

## Porous-channel variables

| Variable | Type | Units | Meaning |
|---|---:|---:|---|
| `SPAN_WIDTH` | `float` | m | Out-of-plane width used to convert channel flow rate `Q` into panel-normal transpiration velocity `vn`. |
| `PORE_DIAMETER` | `float` | m | Common pore/channel diameter for Models 1-3. |
| `DEFAULT_PORE_DIAMETER` | `float` | m | Default value copied from `PORE_DIAMETER` for dataclass defaults. |

Model 4 uses its own fixed diameters inside `build_model4_saved_case1()` in `porous_core.py`.

## Coupling variables

The `COUPLING` object is a `CouplingConfig` dataclass from `solver.py`.

| Field | Type | Meaning |
|---|---:|---|
| `max_iter` | `int` | Maximum fixed-point coupling iterations. |
| `tol_vn` | `float` | Convergence tolerance for the change in normal transpiration velocity. |
| `tol_q` | `float` | Convergence tolerance for the change in internal flow rate. |
| `relaxation` | `float` | Under-relaxation factor. Use a smaller value for difficult cases. |

Recommended convergence adjustments:

- If the solution oscillates, reduce `relaxation`.
- If the solution changes slowly but consistently, increase `max_iter`.
- If large diameters fail to converge, reduce `PORE_DIAMETER` or use stronger under-relaxation.

## Plot variables

| Variable | Type | Meaning |
|---|---:|---|
| `MAKE_CONTOUR_PLOTS` | `bool` | Enables velocity, pressure, and difference contour plots. Disable for faster debugging. |
| `CONTOUR_NX` | `int` | Number of grid points in x for contour and ParaView field outputs. |
| `CONTOUR_NY` | `int` | Number of grid points in y for contour and ParaView field outputs. |

## Angle-of-attack sweep variables

| Variable | Type | Units | Meaning |
|---|---:|---:|---|
| `AOA_SWEEP_START_DEG` | `float` | deg | First angle in the AoA sweep. |
| `AOA_SWEEP_END_DEG` | `float` | deg | Last angle in the AoA sweep. |
| `AOA_SWEEP_STEP_DEG` | `float` | deg | Step size between sweep angles. |
| `EXPORT_AOA_SWEEP_PARAVIEW` | `bool` | — | Writes one ParaView case per AoA and parent `.pvd` sweep collections. |

## Dataclass definitions

### `GeometryLimits`

Controls geometric feasibility checks for pores and internal passages.

| Field | Meaning |
|---|---|
| `x_min_frac`, `x_max_frac` | Allowed pore x/c range. |
| `thickness_fraction_limit` | Maximum allowed pore diameter as a fraction of local airfoil thickness. |
| `min_gap` | Minimum spacing between pore centers. |
| `fit_fraction` | Maximum channel/chamber hydraulic diameter as a fraction of minimum local thickness. |
| `endpoint_tol` | Tolerance for detecting exact leading/trailing-edge endpoints. |
| `endpoint_fit_margin` | Margin used to avoid zero-thickness endpoint checks. |

### `Pore`

One surface pore.

| Field | Meaning |
|---|---|
| `x_frac` | Pore chordwise location, x/c in `[0, 1]`. |
| `side` | Surface side: `"upper"` or `"lower"`. |
| `diameter` | Pore diameter in metres. |

### `Chamber`

Equivalent internal channel/chamber geometry.

| Field | Meaning |
|---|---|
| `hydraulic_diameter` | Hydraulic diameter in metres. |
| `area` | Cross-sectional area in m². |

### `PassageState`

Solved state for one internal passage.

| Field | Meaning |
|---|---|
| `Q` | Volumetric flow rate in m³/s. Positive means pore1 to pore2. |
| `p1`, `p2` | Surface pressures at pore1 and pore2. |
| `dp_total` | Absolute pressure difference. |
| `Rs` | Hydraulic resistance in Pa·s/m³. |
| `reynolds_equivalent` | Equivalent internal Reynolds number. |

### `PassageSpec`

Static model definition for one passage.

| Field | Meaning |
|---|---|
| `name` | Human-readable passage name. |
| `x1_frac`, `side1` | First pore location and side. |
| `x2_frac`, `side2` | Second pore location and side. |
| `diameter_m` | Passage diameter. |
| `layout_kind` | Label used for plotting/ParaView grouping. |

### `PorousModelSpec`

Static model definition for a full porous model.

| Field | Meaning |
|---|---|
| `name` | Model name used in output folder names and `SELECTED_MODEL`. |
| `description` | Human-readable model explanation. |
| `passages` | Tuple of `PassageSpec` definitions. |

### `MultiPassageResult`

Coupled result for a porous model.

| Field | Meaning |
|---|---|
| `aero_result` | Panel-method aerodynamic result after coupling. |
| `passage_states` | List of solved `PassageState` objects. |
| `normal_transpiration` | Panel-normal transpiration velocity array. |
| `converged` | Whether the fixed-point iteration met tolerances. |
| `iterations` | Number of coupling iterations used. |
| `max_vn` | Maximum absolute normal transpiration velocity. |
| `max_vn_over_vinf` | Maximum normal transpiration velocity normalized by freestream speed. |
