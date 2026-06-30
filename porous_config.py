"""User-editable configuration and lightweight data containers.

Edit this file to change the airfoil, Reynolds number, selected porous model,
pore diameter, coupling tolerances, plotting resolution, and XFOIL settings.
The numerical implementation lives in solver.py and porous_core.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil

import numpy as np

from solver import CouplingConfig

# =============================================================================
# XFOIL CONFIGURATION
# =============================================================================
# XFOIL is optional. The panel-method solver runs without it.
#
# Recommended GitHub-friendly setup:
#   1. Put xfoil or xfoil.exe on your PATH, OR
#   2. Set the environment variable XFOIL_EXE to the executable path.
#
# Examples:
#   Windows PowerShell:  $env:XFOIL_EXE="C:\path\to\xfoil.exe"
#   macOS/Linux shell:   export XFOIL_EXE=/path/to/xfoil
USE_XFOIL = os.getenv("USE_XFOIL", "1").strip().lower() not in {"0", "false", "no"}

# Optional search folder used only when XFOIL is not found on PATH or by XFOIL_EXE.
# Recursive scanning is enabled only when this environment variable is set.
_XFOIL_FOLDER_ENV = os.getenv("XFOIL_FOLDER", "").strip()
XFOIL_FOLDER = (
    Path(_XFOIL_FOLDER_ENV).expanduser().resolve()
    if _XFOIL_FOLDER_ENV
    else Path.cwd()
)

# Folder for XFOIL polar/Cp text outputs. Kept outside the source tree by default.
XFOIL_OUTPUT_ROOT = Path(
    os.getenv("XFOIL_OUTPUT_ROOT", str(Path.cwd() / "xfoil_outputs"))
).expanduser()

# Inviscid/incompressible comparison settings.
XFOIL_MACH = 0.0
XFOIL_TIMEOUT = 120.0


def find_xfoil_executable() -> Path | None:
    """Find an XFOIL executable in a portable, GitHub-friendly way.

    Search order:
        1. XFOIL_EXE environment variable
        2. xfoil or xfoil.exe on system PATH
        3. Common subfolders inside XFOIL_FOLDER
        4. Recursive search inside XFOIL_FOLDER

    Returns
    -------
    Path | None
        Absolute executable path when found; otherwise None.
    """
    env_xfoil = os.getenv("XFOIL_EXE", "").strip()
    if env_xfoil:
        env_path = Path(env_xfoil).expanduser()
        if env_path.exists():
            return env_path.resolve()

    for executable_name in ("xfoil", "xfoil.exe"):
        system_xfoil = shutil.which(executable_name)
        if system_xfoil is not None:
            return Path(system_xfoil).resolve()

    candidates = [
        XFOIL_FOLDER / "xfoil",
        XFOIL_FOLDER / "xfoil.exe",
        XFOIL_FOLDER / "bin" / "xfoil",
        XFOIL_FOLDER / "bin" / "xfoil.exe",
        XFOIL_FOLDER / "runs" / "xfoil",
        XFOIL_FOLDER / "runs" / "xfoil.exe",
        XFOIL_FOLDER / "src" / "xfoil",
        XFOIL_FOLDER / "src" / "xfoil.exe",
        XFOIL_FOLDER / "Xfoil" / "xfoil",
        XFOIL_FOLDER / "Xfoil" / "xfoil.exe",
        XFOIL_FOLDER / "Xfoil" / "bin" / "xfoil",
        XFOIL_FOLDER / "Xfoil" / "bin" / "xfoil.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Avoid an expensive recursive scan of the whole repository unless the
    # user explicitly configured XFOIL_FOLDER. This keeps imports fast on fresh clones.
    if _XFOIL_FOLDER_ENV and XFOIL_FOLDER.exists():
        for pattern in ("xfoil", "xfoil.exe"):
            matches = list(XFOIL_FOLDER.rglob(pattern))
            if matches:
                return matches[0].resolve()

    return None


XFOIL_EXE_PATH = find_xfoil_executable()

if USE_XFOIL and XFOIL_EXE_PATH is None:
    print("=" * 80)
    print("[XFOIL WARNING]")
    print("Could not find xfoil/xfoil.exe. XFOIL comparisons will be skipped.")
    print("Set XFOIL_EXE or XFOIL_FOLDER to enable XFOIL on this machine.")
    print(f"Checked folder: {XFOIL_FOLDER}")
    print("=" * 80)
    USE_XFOIL = False
elif USE_XFOIL:
    print(f"[XFOIL] Using executable: {XFOIL_EXE_PATH}")

# =============================================================================
# USER SETTINGS
# =============================================================================
# Run with: python run_porous_models.py
# Edit the values below instead of passing command-line arguments.
OUTPUT_ROOT = Path.cwd() / "run_porous_models_outputs"

# Choose which fixed porous model to run.
# Valid values:
#   "all"
#   "model_1_9_chordwise"
#   "model_2_9_perpendicular"
#   "model_3_combined_independent"
#   "model_4_saved_case_1"
SELECTED_MODEL = "all"

# Geometry and discretisation.
AIRFOIL_NAME = "0018"       # NACA 4-digit airfoil code, e.g. "0018".
CHORD = 1.0                  # Chord length [m].
N_PANELS = 1000              # Number of source/vortex panels on the closed body.

# External flow condition.
REYNOLDS_EXTERNAL = 5.0e5    # Reynolds number based on chord.
AOA_DEG = 4.0                # Angle of attack [degrees].

# Fluid properties.
RHO_INF = 1.225              # Freestream density [kg/m^3].
MU_INF = 1.8e-5              # Dynamic viscosity [Pa s].
P_INF = 101325.0             # Reference static pressure [Pa].

# Porous-channel dimensions.
SPAN_WIDTH = 0.02            # Out-of-plane model width used to convert Q to vn [m].
PORE_DIAMETER = 0.0040       # Common pore/channel diameter for Models 1-3 [m].
DEFAULT_PORE_DIAMETER = PORE_DIAMETER

# Fixed-point coupling controls.
# Decrease relaxation for difficult/high-transpiration cases; increase max_iter
# when the residual decreases slowly but steadily.
COUPLING = CouplingConfig(
    max_iter=500,      # Maximum panel/porous coupling iterations.
    tol_vn=1e-6,      # Convergence tolerance for normal transpiration [m/s].
    tol_q=1e-8,       # Convergence tolerance for channel flow-rate change [m^3/s].
    relaxation=0.5,   # Under-relaxation factor in (0, 1].
)

# Plot resolution. 1000 x 1000 looks good but can be slow.
MAKE_CONTOUR_PLOTS = True
CONTOUR_NX = 1000
CONTOUR_NY = 1000

# AoA sweep settings used for both the CSV/plot comparison and ParaView export.
AOA_SWEEP_START_DEG = -5.0
AOA_SWEEP_END_DEG = 15.0
AOA_SWEEP_STEP_DEG = 1.0

# When True, the AoA sweep also writes one ParaView case per angle plus
# parent .pvd collection files that can be opened as an AoA/time series.
EXPORT_AOA_SWEEP_PARAVIEW = True


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass(frozen=True)
class GeometryLimits:
    # Allow exact LE and TE for the middle Model 1 channel.
    x_min_frac: float = 0.0
    x_max_frac: float = 1.0

    thickness_fraction_limit: float = 0.85
    min_gap: float = 0.0020
    fit_fraction: float = 0.70


    # Exact endpoint handling.
    endpoint_tol: float = 1e-8

    # Skip the zero-thickness endpoint region when checking if the internal
    # channel fits inside the airfoil body.
    endpoint_fit_margin: float = 0.02


@dataclass(frozen=True)
class Pore:
    x_frac: float
    side: str
    diameter: float

    def __post_init__(self) -> None:
        side = self.side.lower()
        object.__setattr__(self, "side", side)

        if side not in ("upper", "lower"):
            raise ValueError("Pore side must be 'upper' or 'lower'.")
        if not (0.0 <= self.x_frac <= 1.0):
            raise ValueError("x_frac must be in [0, 1].")
        if self.diameter <= 0.0:
            raise ValueError("Pore diameter must be positive.")


@dataclass(frozen=True)
class Chamber:
    hydraulic_diameter: float
    area: float

    def __post_init__(self) -> None:
        if self.hydraulic_diameter <= 0.0:
            raise ValueError("hydraulic_diameter must be positive.")
        if self.area <= 0.0:
            raise ValueError("area must be positive.")


@dataclass
class PassageState:
    Q: float
    p1: float
    p2: float
    dp_total: float
    Rs: float
    reynolds_equivalent: float


@dataclass(frozen=True)
class PassageSpec:
    name: str
    x1_frac: float
    side1: str
    x2_frac: float
    side2: str
    diameter_m: float = DEFAULT_PORE_DIAMETER
    layout_kind: str = "surface_pair"


@dataclass(frozen=True)
class PorousModelSpec:
    name: str
    description: str
    passages: tuple[PassageSpec, ...]


@dataclass
class MultiPassageResult:
    aero_result: object
    passage_states: list[PassageState]
    normal_transpiration: np.ndarray
    converged: bool
    iterations: int
    max_vn: float
    max_vn_over_vinf: float


# =============================================================================
# BASIC HELPERS
# =============================================================================
def reynolds_to_velocity(
    reynolds: float,
    rho: float,
    mu: float,
    chord: float,
) -> float:
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if rho <= 0.0:
        raise ValueError("Density must be positive.")
    if mu <= 0.0:
        raise ValueError("Dynamic viscosity must be positive.")
    if chord <= 0.0:
        raise ValueError("Chord must be positive.")

    return reynolds * mu / (rho * chord)


def circular_laminar_resistance(
    mu: float,
    length: float,
    diameter: float,
) -> tuple[float, float]:
    """
    Poiseuille resistance for a circular internal channel.

        Rs = 128 mu L / (pi D^4)
        Q  = dp / Rs
    """
    area = 0.25 * np.pi * diameter**2
    Rs = 128.0 * mu * length / max(np.pi * diameter**4, 1e-30)
    return float(Rs), float(area)
