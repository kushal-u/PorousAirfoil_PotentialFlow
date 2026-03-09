#!/usr/bin/env python3
"""
iter2.py — Modular Porous-Airfoil Panel + Internal Network Coupling
===================================================================

Run:
    python3 iter2.py

What’s new vs iter1.py
----------------------
1) Modular architecture with clear swap points:
   - Geometry factory
   - Aero solver (panel method) with cached A, cached LU, cached influences
   - Network builder (plenum + spine) using your fixed design radii
   - Internal solver (sparse LU cached)
   - Coupling model (relax + clip)
   - Reporter (CSV + optional plots)

2) Fixed design parameters (radii):
   Design: {
      'r_branch_in': 0.0019590275837019517,
      'r_branch_out': 0.005243981588587115,
      'r_spine': 0.008
   }

3) AoA sweep from -5 to +10 degrees (inclusive).
   Saves:
     - output/sweep_summary.csv
     - output/aoa_XXX/ (optional figures + per-AoA CSV)

Notes
-----
- The “spine” radius is used by adding a second internal node, connected to the
  plenum via a spine edge (conductance uses r_spine).
- Inlets connect to spine with r_branch_in.
- Outlets connect to plenum with r_branch_out.

Dependencies: numpy, scipy, networkx, matplotlib
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")
# Only generate heavy figures + Cp-compare plots for these AoAs
PLOT_AOAS = {-5, 0, 10}


# ==============================================================================
# 1) CONFIG
# ==============================================================================

@dataclass(frozen=True)
class AirfoilConfig:
    name: str = "0018"
    n_panels: int = 320  # closed loop => N points = N+1


@dataclass(frozen=True)
class FlowConfig:
    reynolds: float = 250000
    alpha_deg: float = 6.0  # not used in sweep; kept for single-run fallback
    rho: float = 1.225
    mu: float = 1.78e-5
    p_inf: float = 0.0
    chord: float = 1.0

    def v_inf(self) -> float:
        # V = Re*mu / (rho*chord)
        return (self.reynolds * self.mu) / (self.rho * self.chord)


@dataclass(frozen=True)
class PorousConfig:
    # Leakage conversion area (surface pore equivalent)
    surface_pore_radius: float = 5000e-6  # 5mm radius (kept from your iter1)

    # Candidate regions (used in builder selection)
    inlet_x_min: float = 0.02
    inlet_x_max: float = 0.20
    outlet_x_min: float = 0.85

    # Counts
    n_inlets: int = 40
    n_outlets: int = 15


@dataclass(frozen=True)
class DesignRadii:
    r_branch_in: float = 0.0019590275837019517
    r_branch_out: float = 0.005243981588587115
    r_spine: float = 0.008


@dataclass(frozen=True)
class CouplingConfig:
    max_iter: int = 100
    relax: float = 0.01
    tol: float = 1e-8
    clip: float = 80.0
    min_iters: int = 6


@dataclass(frozen=True)
class PlotConfig:
    make_figures: bool = True
    flowfield_res: int = 100
    internal_grid_res: int = 200


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "porous_airfoil_results_iter2"


@dataclass(frozen=True)
class SweepConfig:
    do_sweep: bool = True
    alpha_start: int = -5
    alpha_end: int = 10
    alpha_step: int = 1


@dataclass(frozen=True)
class SimConfig:
    airfoil: AirfoilConfig = AirfoilConfig()
    flow: FlowConfig = FlowConfig()
    porous: PorousConfig = PorousConfig()
    design: DesignRadii = DesignRadii()
    coupling: CouplingConfig = CouplingConfig()
    plot: PlotConfig = PlotConfig()
    output: OutputConfig = OutputConfig()
    sweep: SweepConfig = SweepConfig()


# ==============================================================================
# 2) GEOMETRY
# ==============================================================================

@dataclass
class AirfoilGeometry:
    X: np.ndarray
    Y: np.ndarray
    XC: np.ndarray
    YC: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    L: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    nx: np.ndarray
    ny: np.ndarray


def naca4(number: str, n_panels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed airfoil coords for NACA 4-digit.
    Uses cosine clustering and closed trailing edge thickness coefficient 0.1036.
    """
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0

    beta = np.linspace(0.0, np.pi, n_panels // 2 + 1)
    x = (1.0 - np.cos(beta)) / 2.0

    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m != 0.0:
        mask1 = x <= p
        mask2 = ~mask1
        yc[mask1] = m / p**2 * (2 * p * x[mask1] - x[mask1] ** 2)
        yc[mask2] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[mask2] - x[mask2] ** 2)

        dyc_dx[mask1] = 2 * m / p**2 * (p - x[mask1])
        dyc_dx[mask2] = 2 * m / (1 - p) ** 2 * (p - x[mask2])

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Force trailing edge closed
    xu[-1], yu[-1] = 1.0, 0.0
    xl[-1], yl[-1] = 1.0, 0.0

    # TE upper -> LE -> TE lower
    X = np.concatenate((xu[::-1], xl[1:]))
    Y = np.concatenate((yu[::-1], yl[1:]))
    return X, Y


def make_geometry(cfg: SimConfig) -> AirfoilGeometry:
    X, Y = naca4(cfg.airfoil.name, cfg.airfoil.n_panels)

    # Panel geometry
    XC = (X[:-1] + X[1:]) / 2.0
    YC = (Y[:-1] + Y[1:]) / 2.0
    dx = X[1:] - X[:-1]
    dy = Y[1:] - Y[:-1]
    L = np.sqrt(dx**2 + dy**2)

    tx = dx / L
    ty = dy / L
    nxv = dy / L
    nyv = -dx / L

    return AirfoilGeometry(
        X=np.asarray(X),
        Y=np.asarray(Y),
        XC=np.asarray(XC),
        YC=np.asarray(YC),
        dx=np.asarray(dx),
        dy=np.asarray(dy),
        L=np.asarray(L),
        tx=np.asarray(tx),
        ty=np.asarray(ty),
        nx=np.asarray(nxv),
        ny=np.asarray(nyv),
    )


# ==============================================================================
# 3) AERO SOLVER (Panel Method, cached influences + cached LU)
# ==============================================================================

class PanelMethodAero:
    """
    Source+Vortex panel method:
      - Influence matrices built once per geometry
      - System matrix A built once per geometry
      - LU cached
      - Alpha can be updated cheaply (only freestream projections change)

    solve(V_leakage_normal) returns Cp
    """

    def __init__(self, geom: AirfoilGeometry, flow: FlowConfig):
        self.geom = geom
        self.flow = flow

        self.X = geom.X
        self.Y = geom.Y
        self.XC = geom.XC
        self.YC = geom.YC
        self.L = geom.L
        self.tx = geom.tx
        self.ty = geom.ty
        self.nx = geom.nx
        self.ny = geom.ny

        self.N = len(self.X) - 1
        self.V_INF = flow.v_inf()

        # Cached influences (geometry-only)
        self.Is_n = np.zeros((self.N, self.N))
        self.Iv_n = np.zeros((self.N, self.N))
        self.Is_t = np.zeros((self.N, self.N))
        self.Iv_t = np.zeros((self.N, self.N))
        self._build_influence_matrices()

        # Cached A, LU (geometry-only)
        self.A = None
        self.lu = None
        self.piv = None
        self._build_and_factorize_A()

        # State
        self.alpha = 0.0
        self.Vinf_x = 0.0
        self.Vinf_y = 0.0
        self.Vinf_n = np.zeros(self.N)
        self.Vinf_t = np.zeros(self.N)

        self.q = np.zeros(self.N)
        self.gamma = 0.0

        # Initialize to flow.alpha_deg (if used)
        self.set_alpha_deg(flow.alpha_deg)

    def set_alpha_deg(self, alpha_deg: float) -> None:
        self.alpha = np.radians(alpha_deg)
        self.Vinf_x = self.V_INF * np.cos(self.alpha)
        self.Vinf_y = self.V_INF * np.sin(self.alpha)
        self.Vinf_n = self.Vinf_x * self.nx + self.Vinf_y * self.ny
        self.Vinf_t = self.Vinf_x * self.tx + self.Vinf_y * self.ty

    def _build_influence_matrices(self) -> None:
        N = self.N
        for i in range(N):
            for j in range(N):
                if i == j:
                    self.Is_n[i, j] = 0.5 * np.pi
                    self.Is_t[i, j] = 0.0
                    self.Iv_n[i, j] = 0.0
                    self.Iv_t[i, j] = 0.5 * np.pi
                    continue

                dx = self.XC[i] - self.X[j]
                dy = self.YC[i] - self.Y[j]

                x_local = dx * self.tx[j] + dy * self.ty[j]
                y_local = -dx * self.ty[j] + dy * self.tx[j]

                r1_sq = x_local**2 + y_local**2
                r2_sq = (x_local - self.L[j])**2 + y_local**2

                theta1 = np.arctan2(y_local, x_local)
                theta2 = np.arctan2(y_local, x_local - self.L[j])
                dtheta = theta2 - theta1

                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi

                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc = 1.0 / np.pi * dtheta

                uv_loc, vv_loc = -vs_loc, us_loc

                us_glob = us_loc * self.tx[j] - vs_loc * self.ty[j]
                vs_glob = us_loc * self.ty[j] + vs_loc * self.tx[j]
                uv_glob = uv_loc * self.tx[j] - vv_loc * self.ty[j]
                vv_glob = uv_loc * self.ty[j] + vv_loc * self.tx[j]

                self.Is_n[i, j] = us_glob * self.nx[i] + vs_glob * self.ny[i]
                self.Is_t[i, j] = us_glob * self.tx[i] + vs_glob * self.ty[i]
                self.Iv_n[i, j] = uv_glob * self.nx[i] + vv_glob * self.ny[i]
                self.Iv_t[i, j] = uv_glob * self.tx[i] + vv_glob * self.ty[i]

    def _build_and_factorize_A(self) -> None:
        N = self.N
        A = np.zeros((N + 1, N + 1))

        A[:N, :N] = self.Is_n
        A[:N, N] = np.sum(self.Iv_n, axis=1)

        A[N, :N] = self.Is_t[0, :] + self.Is_t[N - 1, :]
        A[N, N] = np.sum(self.Iv_t[0, :] + self.Iv_t[N - 1, :])

        self.A = A
        self.lu, self.piv = scipy.linalg.lu_factor(A)

    def solve(self, V_leakage_normal: np.ndarray) -> np.ndarray:
        N = self.N
        V_leakage_normal = np.asarray(V_leakage_normal)
        if V_leakage_normal.shape[0] != N:
            raise ValueError(f"V_leakage_normal must have length {N}")

        b = np.zeros(N + 1)
        b[:N] = V_leakage_normal - self.Vinf_n
        b[N] = -(self.Vinf_t[0] + self.Vinf_t[N - 1])

        x = scipy.linalg.lu_solve((self.lu, self.piv), b)
        self.q = x[:N]
        self.gamma = x[N]

        Vt = self.Vinf_t + self.Is_t @ self.q + self.gamma * np.sum(self.Iv_t, axis=1)
        Cp = 1.0 - (Vt / self.V_INF) ** 2
        return Cp

    def velocity_field(self, X_grid: np.ndarray, Y_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(X_grid) + self.Vinf_x
        v = np.zeros_like(Y_grid) + self.Vinf_y

        for j in range(self.N):
            dx = X_grid - self.X[j]
            dy = Y_grid - self.Y[j]
            x_loc = dx * self.tx[j] + dy * self.ty[j]
            y_loc = -dx * self.ty[j] + dy * self.tx[j]

            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - self.L[j])**2 + y_loc**2

            theta1 = np.arctan2(y_loc, x_loc)
            theta2 = np.arctan2(y_loc, x_loc - self.L[j])
            dtheta = theta2 - theta1
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

            us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
            vs_loc = 1.0 / np.pi * dtheta
            uv_loc, vv_loc = -vs_loc, us_loc

            u_ind = (us_loc * self.q[j] + uv_loc * self.gamma) * self.tx[j] - \
                    (vs_loc * self.q[j] + vv_loc * self.gamma) * self.ty[j]
            v_ind = (us_loc * self.q[j] + uv_loc * self.gamma) * self.ty[j] + \
                    (vs_loc * self.q[j] + vv_loc * self.gamma) * self.tx[j]

            u += u_ind
            v += v_ind

        return u, v


# ==============================================================================
# 4) NETWORK BUILDER (uses fixed Design radii)
# ==============================================================================

@dataclass
class NetworkBuildResult:
    G: nx.Graph
    porous_panels: List[int]


class CrossFlowPlenumWithSpineBuilder:
    """
    Builds a small internal network:
      - spine_node (internal) at (0.45, 0.0)
      - plenum_node (internal) at (0.60, 0.0)
      - spine edge between them uses r_spine
      - inlet boundary nodes connect to spine with r_branch_in
      - outlet boundary nodes connect to plenum with r_branch_out

    Inlet selection:
      - bottom surface (y < 0) and x in [inlet_x_min, inlet_x_max]
      - choose highest pressure (Pa) among candidates

    Outlet selection:
      - top surface (y > 0) and x >= outlet_x_min
      - choose largest x (closest to TE)
    """

    def __init__(self, porous: PorousConfig, design: DesignRadii, mu: float):
        self.porous = porous
        self.design = design
        self.mu = mu

    @staticmethod
    def _conductance(radius: float, length: float, mu: float) -> float:
        # Hagen–Poiseuille conductance: Q = cond * dP
        return (np.pi * radius**4) / (8.0 * mu * (length + 1e-15))

    def build(self, aero: PanelMethodAero, pressure_baseline: np.ndarray) -> NetworkBuildResult:
        xc, yc = aero.XC, aero.YC
        G = nx.Graph()

        spine_id = 90001
        plenum_id = 90002
        spine_pos = np.array([0.45, 0.0])
        plenum_pos = np.array([0.60, 0.0])

        G.add_node(spine_id, pos=tuple(spine_pos), type="internal")
        G.add_node(plenum_id, pos=tuple(plenum_pos), type="internal")

        # Spine edge
        Lsp = float(np.linalg.norm(plenum_pos - spine_pos)) + 1e-15
        csp = self._conductance(self.design.r_spine, Lsp, self.mu)
        G.add_edge(spine_id, plenum_id, length=Lsp, cond=csp, radius=self.design.r_spine, type="spine")

        # Candidate panels
        inlet_candidates = [i for i in range(len(xc))
                            if (yc[i] < 0.0 and self.porous.inlet_x_min <= xc[i] <= self.porous.inlet_x_max)]
        outlet_candidates = [i for i in range(len(xc))
                             if (yc[i] > 0.0 and xc[i] >= self.porous.outlet_x_min)]

        # Select inlets by highest pressure (Pa)
        inlet_scores = [{"id": i, "p": float(pressure_baseline[i])} for i in inlet_candidates]
        inlet_scores.sort(key=lambda d: d["p"], reverse=True)
        selected_inlets = [d["id"] for d in inlet_scores[: self.porous.n_inlets]]

        # Select outlets by max x (closest to TE)
        outlet_scores = [{"id": i, "x": float(xc[i])} for i in outlet_candidates]
        outlet_scores.sort(key=lambda d: d["x"], reverse=True)
        selected_outlets = [d["id"] for d in outlet_scores[: self.porous.n_outlets]]

        # Add inlet boundary nodes -> spine
        for pid in selected_inlets:
            if not G.has_node(pid):
                G.add_node(pid, pos=(float(xc[pid]), float(yc[pid])), type="boundary", panel_idx=int(pid))
            node_pos = np.array([xc[pid], yc[pid]])
            length = float(np.linalg.norm(node_pos - spine_pos)) + 1e-15
            cond = self._conductance(self.design.r_branch_in, length, self.mu)
            G.add_edge(pid, spine_id, length=length, cond=cond, radius=self.design.r_branch_in, type="branch_in")

        # Add plenum -> outlet boundary nodes
        for pid in selected_outlets:
            if not G.has_node(pid):
                G.add_node(pid, pos=(float(xc[pid]), float(yc[pid])), type="boundary", panel_idx=int(pid))
            node_pos = np.array([xc[pid], yc[pid]])
            length = float(np.linalg.norm(node_pos - plenum_pos)) + 1e-15
            cond = self._conductance(self.design.r_branch_out, length, self.mu)
            G.add_edge(plenum_id, pid, length=length, cond=cond, radius=self.design.r_branch_out, type="branch_out")

        porous_pids = selected_inlets + selected_outlets
        print(f"   -> Network: {len(selected_inlets)} inlets -> {len(selected_outlets)} outlets; nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        return NetworkBuildResult(G=G, porous_panels=porous_pids)


# ==============================================================================
# 5) INTERNAL SOLVER (cached sparse LU)
# ==============================================================================

class InternalFlowSolverSparseLU:
    """
    Resistive network:
      Q_ij = cond_ij * (P_i - P_j)

    Boundary nodes have prescribed pressure each iteration (via panel pressure).
    Internal node pressures are solved via sparse LU of internal-only system.

    Leakage velocity per porous panel:
      v_leak(pid) = - Q_net(node_pid) / A_surface(pid)

    A_surface(pid) is constant here, based on cfg.porous.surface_pore_radius,
    but it’s isolated so you can replace it easily later.
    """

    def __init__(self, G: nx.Graph, mu: float, surface_pore_radius: float):
        self.G = G
        self.mu = mu
        self.surface_pore_radius = surface_pore_radius

        self.nodes = list(G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.nodes)}

        self.boundary_nodes = [n for n in self.nodes if G.nodes[n].get("type") == "boundary"]
        self.internal_nodes = [n for n in self.nodes if G.nodes[n].get("type") != "boundary"]

        self.int_index = {n: k for k, n in enumerate(self.internal_nodes)}
        self.n_int = len(self.internal_nodes)

        self.A_int = None
        self.lu_int = None
        self._build_factorization()

    def _build_factorization(self) -> None:
        if self.n_int == 0:
            self.A_int = sp.csc_matrix((0, 0))
            self.lu_int = None
            return

        rows, cols, data = [], [], []

        for n in self.internal_nodes:
            k = self.int_index[n]
            diag = 0.0

            for nbr in self.G.neighbors(n):
                c = float(self.G[n][nbr]["cond"])
                diag += c
                if nbr in self.int_index:
                    kk = self.int_index[nbr]
                    rows.append(k); cols.append(kk); data.append(-c)

            rows.append(k); cols.append(k); data.append(diag)

        A_int = sp.csc_matrix((data, (rows, cols)), shape=(self.n_int, self.n_int))
        self.A_int = A_int
        self.lu_int = spla.splu(A_int)

    def area_surface(self, panel_id: int) -> float:
        # Swap point: replace with panel-length scaling / porosity model later
        return float(np.pi * self.surface_pore_radius**2)

    def solve(self, P_boundary_by_panel: Dict[int, float]) -> Tuple[Dict[int, float], np.ndarray]:
        # Boundary node pressures
        P_bnode: Dict[int, float] = {}
        for bn in self.boundary_nodes:
            pid = self.G.nodes[bn].get("panel_idx", None)
            P_bnode[bn] = float(P_boundary_by_panel.get(pid, 0.0))

        # RHS for internal system
        rhs = np.zeros(self.n_int)
        if self.n_int > 0:
            for n in self.internal_nodes:
                k = self.int_index[n]
                s = 0.0
                for nbr in self.G.neighbors(n):
                    if nbr in self.int_index:
                        continue
                    c = float(self.G[n][nbr]["cond"])
                    s += c * P_bnode.get(nbr, 0.0)
                rhs[k] = s

            P_int = self.lu_int.solve(rhs)
        else:
            P_int = np.array([])

        # Full pressure vector in stable ordering
        P_all = np.zeros(len(self.nodes))
        for n in self.internal_nodes:
            P_all[self.node_index[n]] = P_int[self.int_index[n]]
        for n in self.boundary_nodes:
            P_all[self.node_index[n]] = P_bnode[n]

        # Leakage velocity per boundary panel
        velocities_by_panel: Dict[int, float] = {}
        for bn in self.boundary_nodes:
            pid = int(self.G.nodes[bn]["panel_idx"])
            i = self.node_index[bn]
            P_i = float(P_all[i])

            Q_net = 0.0
            for nbr in self.G.neighbors(bn):
                c = float(self.G[bn][nbr]["cond"])
                j = self.node_index[nbr]
                Q_net += c * (P_i - float(P_all[j]))

            A = self.area_surface(pid)
            velocities_by_panel[pid] = -Q_net / (A + 1e-30)

        return velocities_by_panel, P_all


# ==============================================================================
# 6) COUPLING MODEL
# ==============================================================================

class RelaxedClippedCoupling:
    def __init__(self, relax: float, clip: float):
        self.relax = float(relax)
        self.clip = float(clip)

    @staticmethod
    def external_pressure(Cp: np.ndarray, q_inf: float, p_inf: float) -> np.ndarray:
        return p_inf + q_inf * Cp

    @staticmethod
    def boundary_map(P_ext: np.ndarray, porous_pids: List[int]) -> Dict[int, float]:
        return {int(pid): float(P_ext[pid]) for pid in porous_pids}

    def update_leakage(
        self,
        V_old: np.ndarray,
        V_new_dict: Dict[int, float],
        porous_pids: List[int],
    ) -> Tuple[np.ndarray, float]:
        V = V_old.copy()
        max_diff = 0.0

        for pid in porous_pids:
            pid = int(pid)
            v_calc = float(V_new_dict.get(pid, 0.0))
            v_old = float(V[pid])

            v_new = self.relax * v_calc + (1.0 - self.relax) * v_old
            v_new = max(min(v_new, self.clip), -self.clip)

            diff = abs(v_new - v_old)
            if diff > max_diff:
                max_diff = diff

            V[pid] = v_new

        return V, max_diff


# ==============================================================================
# 7) POST: forces + reporting
# ==============================================================================

def compute_forces_like_iter1(Cp: np.ndarray, aero: PanelMethodAero) -> Tuple[float, float, float, float]:
    # Same normalization convention as your iter1 (Cp * panel normals * panel length)
    fx_elem = -Cp * aero.nx * aero.L
    fy_elem = -Cp * aero.ny * aero.L
    Fx = float(np.sum(fx_elem))
    Fy = float(np.sum(fy_elem))

    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    return CL, CD, Fx, Fy


class Reporter:
    def __init__(self, base_out_dir: str, plot_cfg: PlotConfig):
        self.base_out_dir = base_out_dir
        self.plot_cfg = plot_cfg
        os.makedirs(self.base_out_dir, exist_ok=True)

    def save_sweep_summary(self, rows: List[Dict[str, float]]) -> str:
        path = os.path.join(self.base_out_dir, "sweep_summary.csv")
        with open(path, "w") as f:
            headers = list(rows[0].keys()) if rows else ["alpha_deg", "CL_solid", "CL_porous", "CD_solid", "CD_porous"]
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
        return path

    def save_case_csv(
        self,
        case_dir: str,
        aero: PanelMethodAero,
        Cp_solid: np.ndarray,
        Cp: np.ndarray,
        V_leakage: np.ndarray,
        CL_solid: float,
        CL: float,
        CD_solid: float,
        CD: float,
    ) -> str:
        os.makedirs(case_dir, exist_ok=True)
        csv_path = os.path.join(case_dir, "simulation_data.csv")

        cl_change = ((CL - CL_solid) / (abs(CL_solid) + 1e-9)) * 100.0
        cd_change = ((CD - CD_solid) / (abs(CD_solid) + 1e-9)) * 100.0

        with open(csv_path, "w") as f:
            f.write("--- GLOBAL RESULTS ---\n")
            f.write("Metric,Solid_Baseline,Porous_Result,Change_Percent\n")
            f.write(f"CL,{CL_solid:.6f},{CL:.6f},{cl_change:.2f}%\n")
            f.write(f"CD,{CD_solid:.6f},{CD:.6f},{cd_change:.2f}%\n\n")

            f.write("--- PANEL DISTRIBUTION DATA ---\n")
            f.write("Panel_ID,XC,YC,Cp_Solid,Cp_Porous,V_leakage\n")
            for i in range(aero.N):
                f.write(
                    f"{i},{aero.XC[i]:.6f},{aero.YC[i]:.6f},"
                    f"{Cp_solid[i]:.6f},{Cp[i]:.6f},{V_leakage[i]:.6f}\n"
                )
        return csv_path

    def plot_case(
        self,
        case_dir: str,
        aero: PanelMethodAero,
        Cp_solid: np.ndarray,
        Cp: np.ndarray,
        V_leakage: np.ndarray,
        CL_solid: float,
        CL: float,
        G: nx.Graph,
        P_nodes: np.ndarray,
        node_order: List[int],
    ) -> None:
        if not self.plot_cfg.make_figures:
            return

        pos = nx.get_node_attributes(G, "pos")

        # Fig 1: geometry + Cp
        fig1 = plt.figure(figsize=(12, 12))
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])

        ax1 = fig1.add_subplot(gs1[0])
        ax1.plot(aero.X, aero.Y, linewidth=2, color="black", label="Airfoil")
        ax1.fill(aero.X, aero.Y, "whitesmoke")
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="cyan", alpha=0.6, width=1.5)
        nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), ax=ax1, node_size=30, node_color="black")
        ax1.set_title("Geometry & Porous Network", fontsize=14)
        ax1.axis("equal")

        ax2 = fig1.add_subplot(gs1[1])
        ax2.plot(aero.XC, Cp_solid, "k--", label=f"Solid ($C_L = {CL_solid:.3f}$)")
        ax2.plot(aero.XC, Cp, "b-", label=f"Porous ($C_L = {CL:.3f}$)")
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title("Pressure Coefficient ($C_p$)", fontsize=14)

        fig1.savefig(os.path.join(case_dir, "01_Geometry_and_Cp.png"), dpi=250, bbox_inches="tight")
        plt.close(fig1)

        # Fig 3: surface vectors (pressure)
        fig3 = plt.figure(figsize=(10, 6))
        ax5 = fig3.add_subplot(111)
        ax5.plot(aero.X, aero.Y, linewidth=2, color="black")
        ax5.fill(aero.X, aero.Y, "whitesmoke")

        U_p = -Cp * aero.nx * 0.15
        V_p = -Cp * aero.ny * 0.15
        ax5.quiver(aero.XC, aero.YC, U_p, V_p, Cp, cmap="coolwarm_r", scale=1, scale_units="xy", width=0.004)
        ax5.set_title("Surface Pressure & Leakage", fontsize=14)
        ax5.axis("equal")

        fig3.savefig(os.path.join(case_dir, "03_Pressure_Vectors.png"), dpi=250, bbox_inches="tight")
        plt.close(fig3)

        # Fig 5: external flowfield
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.6, 0.6
        Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, self.plot_cfg.flowfield_res),
                             np.linspace(y_min, y_max, self.plot_cfg.flowfield_res))
        Ug, Vg = aero.velocity_field(Xg, Yg)
        vel_mag = np.sqrt(Ug**2 + Vg**2)

        fig5 = plt.figure(figsize=(12, 7))
        ax9 = fig5.add_subplot(111)
        ax9.contourf(Xg, Yg, vel_mag, levels=50, cmap="viridis", extend="both", alpha=0.9)

        if np.max(vel_mag) > 1e-6:
            seed_points = np.column_stack((np.ones(40) * x_min, np.linspace(y_min, y_max, 40)))
            ax9.streamplot(Xg, Yg, Ug, Vg, color="white", linewidth=0.8, density=2, start_points=seed_points)

        ax9.fill(aero.X, aero.Y, color="black", zorder=3)
        ax9.set_title("External Flow Field", fontsize=14)
        ax9.axis("equal")

        fig5.savefig(os.path.join(case_dir, "05_Flow_Field.png"), dpi=250, bbox_inches="tight")
        plt.close(fig5)

        # Fig 6: internal velocity contour
        node_map = {node: i for i, node in enumerate(node_order)}
        points = []
        values = []
        max_v_found = 0.0

        for u, v, data in G.edges(data=True):
            idx_u = node_map[u]
            idx_v = node_map[v]
            Pu = float(P_nodes[idx_u])
            Pv = float(P_nodes[idx_v])
            dP = Pu - Pv

            cond = float(data["cond"])
            Q = cond * dP

            R_edge = float(data.get("radius", 1e-6))
            A_pipe = np.pi * (R_edge**2)
            v_mag = abs(Q) / (A_pipe + 1e-30)
            max_v_found = max(max_v_found, v_mag)

            pos_u = np.array(pos[u])
            pos_v = np.array(pos[v])
            for t in np.linspace(0, 1, 5):
                pt = pos_u + t * (pos_v - pos_u)
                points.append(pt)
                values.append(v_mag)

        points = np.array(points) if points else np.zeros((0, 2))
        values = np.array(values) if values else np.zeros((0,))

        fig6 = plt.figure(figsize=(14, 8))
        ax_int = fig6.add_subplot(111)
        ax_int.plot(aero.X, aero.Y, linewidth=3, color="#333333", zorder=10)

        if len(points) > 0:
            min_x, max_x = float(np.min(aero.X)), float(np.max(aero.X))
            min_y, max_y = float(np.min(aero.Y)), float(np.max(aero.Y))
            grid_x, grid_y = np.mgrid[min_x:max_x:complex(self.plot_cfg.internal_grid_res),
                                     min_y:max_y:complex(self.plot_cfg.internal_grid_res)]
            grid_z = griddata(points, values, (grid_x, grid_y), method="linear", fill_value=0.0)

            airfoil_poly = np.column_stack((aero.X, aero.Y))
            path = mpath.Path(airfoil_poly)
            grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            mask = path.contains_points(grid_points).reshape(grid_x.shape)
            grid_z[~mask] = np.nan

            contour = ax_int.contourf(grid_x, grid_y, grid_z, levels=100, cmap="plasma", zorder=1)
            cbar = plt.colorbar(contour, ax=ax_int, pad=0.02)
            cbar.set_label("Internal Pipe Velocity [m/s]", fontsize=12)

        nx.draw_networkx_edges(G, pos, ax=ax_int, edge_color="white", alpha=0.3, width=0.5)
        ax_int.set_title(f"Internal Flow Distribution\nMax Velocity: {max_v_found:.2f} m/s", fontsize=16)
        ax_int.set_xlabel("x/c")
        ax_int.set_ylabel("y/c")
        ax_int.axis("equal")
        ax_int.set_xlim(0.0, 1.05)

        fig6.savefig(os.path.join(case_dir, "06_Internal_Flow_Map_Contour.png"), dpi=250, bbox_inches="tight")
        plt.close(fig6)
    def plot_sweep_graphs(
        self,
        results: List["CaseResult"],
        cp_cache: Dict[int, Dict[str, np.ndarray]],
        xc: np.ndarray,
    ) -> None:
        """
        Creates sweep graphs:
          - Cp distribution at AoA -5, 0, 10 (solid vs porous)
          - CL vs CD (solid + porous across sweep)
          - Aerodynamic efficiency (CL/CD) vs AoA (solid + porous)
          - CL vs AoA (solid + porous)
        Saves all into base_out_dir.
        """

        # ---- 1) Cp distribution compare at selected AoAs ----
        for a in sorted(PLOT_AOAS):
            if a not in cp_cache:
                continue

            Cp_s = cp_cache[a]["Cp_solid"]
            Cp_p = cp_cache[a]["Cp_porous"]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.plot(xc, Cp_s, "k--", label="Solid")
            ax.plot(xc, Cp_p, "b-", label="Porous")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x/c")
            ax.set_ylabel("$C_p$")
            ax.set_title(f"$C_p$ Distribution Comparison (AoA = {a}°)")
            ax.legend()
            fig.savefig(os.path.join(self.base_out_dir, f"Cp_compare_AoA_{a:+d}.png".replace("+", "p").replace("-", "m")),
                        dpi=250, bbox_inches="tight")
            plt.close(fig)

        # ---- Prepare arrays from results ----
        alphas = np.array([r.alpha_deg for r in results], dtype=float)
        cl_s = np.array([r.CL_solid for r in results], dtype=float)
        cd_s = np.array([r.CD_solid for r in results], dtype=float)
        cl_p = np.array([r.CL_porous for r in results], dtype=float)
        cd_p = np.array([r.CD_porous for r in results], dtype=float)

        # Sort by AoA for clean lines
        order = np.argsort(alphas)
        alphas = alphas[order]
        cl_s, cd_s = cl_s[order], cd_s[order]
        cl_p, cd_p = cl_p[order], cd_p[order]

        # ---- 2) CL vs CD (polar) ----
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(cd_s, cl_s, "k--", label="Solid")
        ax.plot(cd_p, cl_p, "b-", label="Porous")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$C_D$")
        ax.set_ylabel("$C_L$")
        ax.set_title("$C_L$ vs $C_D$ (Polar)")
        ax.legend()
        fig.savefig(os.path.join(self.base_out_dir, "CL_vs_CD.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)

        # ---- 3) Aerodynamic efficiency (CL/CD) vs AoA ----
        # Protect against divide by zero
        eps = 1e-12
        eff_s = cl_s / (cd_s + eps)
        eff_p = cl_p / (cd_p + eps)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(alphas, eff_s, "k--", label="Solid")
        ax.plot(alphas, eff_p, "b-", label="Porous")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("AoA [deg]")
        ax.set_ylabel("$C_L/C_D$")
        ax.set_title("Aerodynamic Efficiency vs AoA")
        ax.legend()
        fig.savefig(os.path.join(self.base_out_dir, "Efficiency_CL_over_CD_vs_AoA.png"),
                    dpi=250, bbox_inches="tight")
        plt.close(fig)

        # ---- 4) CL vs AoA ----
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(alphas, cl_s, "k--", label="Solid")
        ax.plot(alphas, cl_p, "b-", label="Porous")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("AoA [deg]")
        ax.set_ylabel("$C_L$")
        ax.set_title("$C_L$ vs AoA")
        ax.legend()
        fig.savefig(os.path.join(self.base_out_dir, "CL_vs_AoA.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)


# ==============================================================================
# 8) SINGLE CASE RUN
# ==============================================================================

@dataclass
class CaseResult:
    alpha_deg: float
    CL_solid: float
    CD_solid: float
    CL_porous: float
    CD_porous: float
    iters: int
    converged: bool


def run_case(cfg: SimConfig, aero: PanelMethodAero, alpha_deg: float, reporter: Reporter) -> CaseResult:
    aero.set_alpha_deg(alpha_deg)

    V_INF = aero.V_INF
    q_inf = 0.5 * cfg.flow.rho * (V_INF**2)

    # Solid baseline
    V0 = np.zeros(aero.N)
    Cp_solid = aero.solve(V0)
    CL_solid, CD_solid, _, _ = compute_forces_like_iter1(Cp_solid, aero)

    # Build network (selection uses baseline pressure in Pa)
    P_solid = cfg.flow.p_inf + q_inf * Cp_solid
    builder = CrossFlowPlenumWithSpineBuilder(cfg.porous, cfg.design, cfg.flow.mu)
    build_res = builder.build(aero, P_solid)

    internal = InternalFlowSolverSparseLU(
        build_res.G,
        mu=cfg.flow.mu,
        surface_pore_radius=cfg.porous.surface_pore_radius,
    )
    coupling = RelaxedClippedCoupling(cfg.coupling.relax, cfg.coupling.clip)

    V_leak = np.zeros(aero.N)
    Cp = Cp_solid.copy()
    final_P_nodes = np.zeros(len(internal.nodes))

    converged = False
    it_used = 0

    for it in range(cfg.coupling.max_iter):
        it_used = it

        Cp = aero.solve(V_leak)
        P_ext = coupling.external_pressure(Cp, q_inf=q_inf, p_inf=cfg.flow.p_inf)
        P_map = coupling.boundary_map(P_ext, build_res.porous_panels)

        V_calc, P_nodes_iter = internal.solve(P_map)
        final_P_nodes = P_nodes_iter

        V_leak, max_diff = coupling.update_leakage(V_leak, V_calc, build_res.porous_panels)

        if (max_diff < cfg.coupling.tol) and (it >= cfg.coupling.min_iters):
            converged = True
            break

        if it % 10 == 0:
            print(f"   AoA {alpha_deg:+.0f}° | Iter {it:03d} | max_res={max_diff:.3e}")

    # Porous forces
    CL, CD, _, _ = compute_forces_like_iter1(Cp, aero)

    # Output per-case
    case_dir = os.path.join(cfg.output.out_dir, f"aoa_{alpha_deg:+03.0f}".replace("+", "p").replace("-", "m"))
    reporter.save_case_csv(
        case_dir=case_dir,
        aero=aero,
        Cp_solid=Cp_solid,
        Cp=Cp,
        V_leakage=V_leak,
        CL_solid=CL_solid,
        CL=CL,
        CD_solid=CD_solid,
        CD=CD,
    )
    if int(round(alpha_deg)) in PLOT_AOAS:
        reporter.plot_case(
            case_dir=case_dir,
            aero=aero,
            Cp_solid=Cp_solid,
            Cp=Cp,
            V_leakage=V_leak,
            CL_solid=CL_solid,
            CL=CL,
            G=build_res.G,
            P_nodes=final_P_nodes,
            node_order=internal.nodes,
        )


    print(f"-> AoA {alpha_deg:+.0f}° done | Solid CL={CL_solid:.4f} | Porous CL={CL:.4f} | iters={it_used} | converged={converged}")
    return CaseResult(
        alpha_deg=float(alpha_deg),
        CL_solid=float(CL_solid),
        CD_solid=float(CD_solid),
        CL_porous=float(CL),
        CD_porous=float(CD),
        iters=int(it_used),
        converged=bool(converged),
    )


# ==============================================================================
# 9) MAIN (AoA sweep)
# ==============================================================================

def run_simulation(cfg: SimConfig) -> None:
    print("=== iter2 porous-airfoil simulation ===")
    print(f"Output dir: {cfg.output.out_dir}")
    os.makedirs(cfg.output.out_dir, exist_ok=True)

    print("Config (key parts):")
    print(f"  Airfoil: NACA {cfg.airfoil.name}, panels={cfg.airfoil.n_panels}")
    print(f"  Flow: Re={cfg.flow.reynolds}, rho={cfg.flow.rho}, mu={cfg.flow.mu}, V_inf={cfg.flow.v_inf():.2f} m/s")
    print(f"  Design radii: {asdict(cfg.design)}")
    print(f"  Coupling: max_iter={cfg.coupling.max_iter}, relax={cfg.coupling.relax}, tol={cfg.coupling.tol}, clip={cfg.coupling.clip}")
    print(f"  Sweep: {cfg.sweep.alpha_start}..{cfg.sweep.alpha_end} step {cfg.sweep.alpha_step}")

    geom = make_geometry(cfg)
    aero = PanelMethodAero(geom, cfg.flow)  # cached influences + cached LU
    reporter = Reporter(cfg.output.out_dir, cfg.plot)
    # Store Cp arrays only for AoA in PLOT_AOAS so we can make Cp-compare plots later
    cp_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # AoA sweep
    results: List[CaseResult] = []
    if cfg.sweep.do_sweep:
        for alpha in range(cfg.sweep.alpha_start, cfg.sweep.alpha_end + 1, cfg.sweep.alpha_step):
            alpha_f = float(alpha)

            # --- Run the case normally ---
            res = run_case(cfg, aero, alpha_f, reporter)
            results.append(res)

            # --- For selected AoAs, recompute Cp_solid and Cp_porous and cache them ---
            if int(alpha) in PLOT_AOAS:
                aero.set_alpha_deg(alpha_f)
                Cp_solid = aero.solve(np.zeros(aero.N))

                # Rebuild the same network for this AoA baseline (same logic as run_case)
                V_INF = aero.V_INF
                q_inf = 0.5 * cfg.flow.rho * (V_INF**2)
                P_solid = cfg.flow.p_inf + q_inf * Cp_solid
                builder = CrossFlowPlenumWithSpineBuilder(cfg.porous, cfg.design, cfg.flow.mu)
                build_res = builder.build(aero, P_solid)

                internal = InternalFlowSolverSparseLU(
                    build_res.G,
                    mu=cfg.flow.mu,
                    surface_pore_radius=cfg.porous.surface_pore_radius,
                )
                coupling = RelaxedClippedCoupling(cfg.coupling.relax, cfg.coupling.clip)

                V_leak = np.zeros(aero.N)
                Cp_porous = Cp_solid.copy()

                # same coupling iteration to get final porous Cp (lightweight, but only 3 AoAs)
                for it in range(cfg.coupling.max_iter):
                    Cp_porous = aero.solve(V_leak)
                    P_ext = coupling.external_pressure(Cp_porous, q_inf=q_inf, p_inf=cfg.flow.p_inf)
                    P_map = coupling.boundary_map(P_ext, build_res.porous_panels)
                    V_calc, _ = internal.solve(P_map)
                    V_leak, max_diff = coupling.update_leakage(V_leak, V_calc, build_res.porous_panels)
                    if (max_diff < cfg.coupling.tol) and (it >= cfg.coupling.min_iters):
                        break

                cp_cache[int(alpha)] = {
                    "Cp_solid": Cp_solid.copy(),
                    "Cp_porous": Cp_porous.copy(),
                }

    else:
        results.append(run_case(cfg, aero, cfg.flow.alpha_deg, reporter))

    # Save sweep summary
    rows = []
    for r in results:
        rows.append({
            "alpha_deg": r.alpha_deg,
            "CL_solid": r.CL_solid,
            "CL_porous": r.CL_porous,
            "CD_solid": r.CD_solid,
            "CD_porous": r.CD_porous,
            "iters": r.iters,
            "converged": int(r.converged),
        })

    if rows:
        path = reporter.save_sweep_summary(rows)
        print(f"-> Sweep summary saved: {path}")
        # Make sweep-level graphs (Cp compare at -5/0/10, CL-CD, efficiency, CL vs AoA)
        reporter.plot_sweep_graphs(results, cp_cache, xc=aero.XC)
        print("-> Sweep graphs saved in output directory.")


    # Quick console table
    print("\nAoA sweep results:")
    print(" alpha |  CL_solid  CL_porous |  CD_solid  CD_porous | iters conv")
    for r in results:
        print(f"{r.alpha_deg:>+5.0f} | {r.CL_solid:>8.4f} {r.CL_porous:>9.4f} | {r.CD_solid:>8.4f} {r.CD_porous:>9.4f} | {r.iters:>5d}  {int(r.converged)}")


def main() -> None:
    cfg = SimConfig(
        # You asked to fix these radii; already embedded in DesignRadii default.
        design=DesignRadii(
            r_branch_in=0.0019590275837019517,
            r_branch_out=0.005243981588587115,
            r_spine=0.008,
        ),
        # Keep sweep on by default
        sweep=SweepConfig(do_sweep=True, alpha_start=-5, alpha_end=10, alpha_step=1),
    )
    run_simulation(cfg)


if __name__ == "__main__":
    main()
