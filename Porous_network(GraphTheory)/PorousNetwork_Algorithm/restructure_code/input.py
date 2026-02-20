# input.py
import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for physics and geometry."""
    # File / Output
    AIRFOIL_NAME: str = "0018"
    OUTPUT_DIR: str = "porous_airfoil_results"

    # Geometry
    N_PANELS: int = 3000
    CHORD: float = 1.0

    # Porous Network Settings
    PORE_RADIUS_INLET: float = 10000e-6
    PORE_RADIUS_OUTLET: float = 10000e-6
    N_INLETS: int = 1
    N_OUTLETS: int = 1

    # Physics
    REYNOLDS_NUM: float = 250000
    ANGLE_OF_ATTACK: float = 6.0
    RHO: float = 1.225
    MU: float = 1.78e-5
    P_INF: float = 0.0

    # Solver
    CONVERGENCE_TOL: float = 1e-8

    # --- Anderson coupling (replaces Newton–Krylov) ---
    ANDERSON_M: int = 5
    ANDERSON_MAXITER: int = 60
    ANDERSON_BETA: float = 0.005   # relaxation (0.2–0.6 typical)
    ANDERSON_DAMPING: float = 1e-10

    # --- Numba switches ---
    USE_NUMBA: bool = True
    NUMBA_PARALLEL: bool = True

    # --- Plot quality / resolution ---
    FIG_DPI: int = 300
    FLOW_NX: int = 100
    FLOW_NY: int = 100
    CONTOUR_LEVELS: int = 60
    STREAM_DENSITY: float = 2.0

    NETWORK_TOPOLOGY = "pressure_web" # Options: "spine", "suction_web", "pressure_web"

    SUCTION_PORE_X_MAX = 0.85
    FORCE_PORE_AT_XMAX = True
    XMAX_TARGET_TOL = 0.02

    N_SUCTION_LOWEST = 3
    N_SUCTION_BINS = 3

    MIN_PORE_SPACING = 0.03
    MIN_PORE_PANEL_GAP = 2

    N_SPINE_NODES = 6
    SPINE_Y = 0.0
    SPINE_X_PAD_LO = 0.15
    SPINE_X_PAD_HI = 0.95

    PORE_RADIUS_WEB = 0.005
    PORE_RADIUS_SPINE = 0.005
    PORE_RADIUS_WEB_TO_SPAR = 0.005

    @property
    def V_INF(self):
        return (self.REYNOLDS_NUM * self.MU) / (self.RHO * self.CHORD)


class AirfoilGenerator:
    @staticmethod
    def generate_naca4(number: str, n_panels: int = 160):
        """Generates NACA 4-digit airfoil coordinates."""
        m = int(number[0]) / 100.0
        p = int(number[1]) / 10.0
        t = int(number[2:]) / 100.0

        beta = np.linspace(0, np.pi, n_panels // 2 + 1)
        x = (1 - np.cos(beta)) / 2
        yt = 5 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )

        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        if m != 0:
            yc[x <= p] = m / p**2 * (2 * p * x[x <= p] - x[x <= p] ** 2)
            yc[x > p] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[x > p] - x[x > p] ** 2)
            dyc_dx[x <= p] = 2 * m / p**2 * (p - x[x <= p])
            dyc_dx[x > p] = 2 * m / (1 - p) ** 2 * (p - x[x > p])

        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # enforce TE closure
        xu[-1], yu[-1] = 1.0, 0.0
        xl[-1], yl[-1] = 1.0, 0.0

        X = np.concatenate((xu[::-1], xl[1:]))
        Y = np.concatenate((yu[::-1], yl[1:]))

        return X, Y
