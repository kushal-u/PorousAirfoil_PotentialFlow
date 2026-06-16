# Porous Airfoil Panel Method

A Python research code for comparing **solid** and **porous** NACA 4-digit airfoils using a source-vortex panel method coupled to simplified internal hydraulic-resistance channels.

The current supported workflow runs fixed porous-network layouts, compares them against a solid-airfoil panel-method baseline, optionally compares against XFOIL, and exports CSV tables, Matplotlib plots, and ParaView-compatible VTK files.

## Main features

- NACA 4-digit airfoil generation with cosine-spaced panels.
- Source-vortex panel-method solver with Kutta condition.
- Fixed porous-channel models:
  - `model_1_9_chordwise`: 9 leading-edge to trailing-edge internal channels.
  - `model_2_9_perpendicular`: 9 lower-to-upper channels.
  - `model_3_combined_independent`: Models 1 and 2 combined as independent passages.
  - `model_4_saved_case_1`: saved two-passage validation/design case.
- Fixed-point coupling between surface pressure, internal flow rate, and surface transpiration velocity.
- Optional XFOIL Cp and angle-of-attack sweep comparison.
- CSV outputs for surface data, passage data, model summaries, and AoA sweeps.
- Matplotlib plots for airfoil layout, Cp distribution, aerodynamic sweep, and contour comparisons.
- ParaView `.vtp`, `.vts`, and `.pvd` exports for surface, porous-network, flow-field, and AoA sweep visualisation.

## Repository layout

```text
porous_airfoil_panel_method/
├── run.py                         # simple GitHub-style entry point
├── run_porous_models.py           # main fixed-model runner
├── porous_config.py               # user-editable simulation settings and dataclasses
├── porous_core.py                 # porous-network model definitions, coupling, CSV/sweep logic
├── solver.py                      # NACA geometry and source-vortex panel solver
├── plotter.py                     # Matplotlib plotting utilities
├── paraview_export.py             # ASCII VTK/ParaView export utilities
├── xfoil.py                       # optional XFOIL runner and parser
├── requirements.txt               # required Python dependencies
├── requirements-optional.txt      # optional acceleration dependencies
├── pyproject.toml                 # project metadata and tooling config
├── .gitignore                     # ignores generated outputs and local files
├── docs/
│   ├── FILE_REFERENCE.md          # detailed Python file descriptions
│   └── CONFIG_REFERENCE.md        # detailed configuration variable definitions

```

## Quick start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Optional acceleration for large contour grids:

```bash
pip install -r requirements-optional.txt
```

### 3. Configure the case

Open `porous_config.py` and edit the user settings. The most common variables are:

```python
SELECTED_MODEL = "all"          # or one model name
AIRFOIL_NAME = "0018"           # NACA 4-digit airfoil
N_PANELS = 1000                 # panel discretisation
REYNOLDS_EXTERNAL = 5.0e5       # chord Reynolds number
AOA_DEG = 4.0                   # fixed-case angle of attack
PORE_DIAMETER = 0.0040          # common diameter for Models 1-3 [m]
MAKE_CONTOUR_PLOTS = True       # set False for a faster run
```

To run only the saved two-passage case:

```python
SELECTED_MODEL = "all"
```

### 4. Run

```bash
python run.py
```

Equivalent:

```bash
python run_porous_models.py
```

Results are written to:

```text
run_porous_models_outputs/
```

## XFOIL setup, optional

The panel-method code does not require XFOIL. If XFOIL is unavailable, the program skips XFOIL Cp and polar comparisons and still runs the panel/porous model.

Recommended setup:

Windows PowerShell:

```powershell
$env:XFOIL_EXE="C:\path\to\xfoil.exe"
```

macOS/Linux:

```bash
export XFOIL_EXE=/path/to/xfoil
```

You can also disable XFOIL explicitly:

Windows PowerShell:

```powershell
$env:USE_XFOIL="0"
```

macOS/Linux:

```bash
export USE_XFOIL=0
```

## Output files

For each selected model, the runner creates a folder such as:

```text
run_porous_models_outputs/model_1_9_chordwise/
```

Typical outputs include:

- `passage_summary.csv`: pore positions, channel lengths, pressure drops, flow rates, hydraulic resistance, and equivalent internal Reynolds numbers.
- `surface_data.csv`: panel-level Cp, tangential velocity, solid/porous differences, and normal transpiration velocity.
- `model_summary.csv`: lift, drag, moment, convergence, and runtime summary for one model.
- `aoa_sweep_comparison.csv`: porous and solid panel-method aerodynamic coefficients over the configured AoA range.
- `airfoil_with_porous_network.png`: airfoil layout with pores/channels and flow-direction arrows.
- `cp_distribution_with_xfoil.png`: porous, solid, and optional XFOIL Cp comparison.
- `aoa_sweep_CL_CD_CM.png`: CL, CD, and CM versus angle of attack.
- `paraview_results.pvd` plus `.vtp` and `.vts` files: ParaView visualisation outputs.

The repository root output also includes:

- `all_models_summary.csv`: one-row summary per selected model.
- `solid_reference_paraview/`: ParaView files for the solid-airfoil baseline.

## ParaView usage

Open the `.pvd` files in ParaView:

- Single fixed model: `model_name/paraview_results.pvd`
- Solid reference: `solid_reference_paraview/paraview_solid_reference.pvd`
- AoA sweep: `model_name/aoa_sweep_paraview/paraview_aoa_sweep_all_parts.pvd`

For AoA sweeps, the angle of attack is stored as the ParaView timestep.

## Important numerical notes

- The internal-channel model uses laminar circular Poiseuille resistance:

  ```text
  Rs = 128 μ L / (π D⁴)
  Q  = Δp / Rs
  ```

- Channel diameter strongly affects coupling strength because hydraulic resistance scales with `D^-4`. Increasing diameter can dramatically increase `Q` and surface transpiration velocity, which can make the fixed-point coupling harder to converge.
- For difficult cases, reduce `COUPLING.relaxation`, reduce pore diameter, or increase `COUPLING.max_iter` in `porous_config.py`.
- Very high contour resolutions, for example `1000 x 1000`, can be slow. Set `MAKE_CONTOUR_PLOTS = False` or lower `CONTOUR_NX` and `CONTOUR_NY` for quick checks.

## Detailed file and variable documentation

See:

- [`docs/FILE_REFERENCE.md`](docs/FILE_REFERENCE.md)
- [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md)



## License

No license has been selected in this prepared version. Before publishing publicly on GitHub, add a `LICENSE` file that matches how you want others to use the code.
