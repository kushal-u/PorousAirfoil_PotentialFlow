#!/usr/bin/env python3
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

# Option A pipe topology: number of internal spine nodes
SPINE_NODES = 12


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
    alpha_deg: float = 6.0
    rho: float = 1.225
    mu: float = 1.78e-5
    p_inf: float = 0.0
    chord: float = 1.0

    def v_inf(self) -> float:
        return (self.reynolds * self.mu) / (self.rho * self.chord)


@dataclass(frozen=True)
class PorousConfig:
    # Effective surface pore radius used to convert flow rate Q to leakage velocity v = Q/A
    surface_pore_radius: float = 5000e-6  # 5mm

    # Porosity / number-of-pores knob:
    # total effective leakage area = area_mult * (pi * r_pore^2)
    area_mult: float = 50.0

    # Candidate regions
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
    tol: float = 1e-6          # RMS residual target
    clip: float = 80.0         # smooth clip tanh
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

    xu[-1], yu[-1] = 1.0, 0.0
    xl[-1], yl[-1] = 1.0, 0.0

    X = np.concatenate((xu[::-1], xl[1:]))
    Y = np.concatenate((yu[::-1], yl[1:]))
    return X, Y


def make_geometry(cfg: SimConfig) -> AirfoilGeometry:
    X, Y = naca4(cfg.airfoil.name, cfg.airfoil.n_panels)

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
        X=np.asarray(X), Y=np.asarray(Y),
        XC=np.asarray(XC), YC=np.asarray(YC),
        dx=np.asarray(dx), dy=np.asarray(dy),
        L=np.asarray(L),
        tx=np.asarray(tx), ty=np.asarray(ty),
        nx=np.asarray(nxv), ny=np.asarray(nyv),
    )


# ==============================================================================
# 3) AERO SOLVER (Panel Method, cached influences + cached LU)
# ==============================================================================

class PanelMethodAero:
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

        self.Is_n = np.zeros((self.N, self.N))
        self.Iv_n = np.zeros((self.N, self.N))
        self.Is_t = np.zeros((self.N, self.N))
        self.Iv_t = np.zeros((self.N, self.N))
        self._build_influence_matrices()

        self.A = None
        self.lu = None
        self.piv = None
        self._build_and_factorize_A()

        self.alpha = 0.0
        self.Vinf_x = 0.0
        self.Vinf_y = 0.0
        self.Vinf_n = np.zeros(self.N)
        self.Vinf_t = np.zeros(self.N)

        self.q = np.zeros(self.N)
        self.gamma = 0.0
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
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

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
# 4) PIPE TOPOLOGY BUILDER (Spine trunk + inlet/outlet branches)
# ==============================================================================

@dataclass
class NetworkBuildResult:
    G: nx.Graph
    porous_panels: List[int]


class PipeTopologyBuilder:
    """
    Pipe topology (Option A):
      - K spine nodes on a trunk line inside airfoil (default y=0)
      - Connect consecutive spine nodes with radius r_spine
      - Inlet boundary nodes connect to nearest spine node (r_branch_in)
      - Outlet boundary nodes connect to nearest spine node (r_branch_out)

    Selection:
      Inlets: lower surface y<0, x in [inlet_x_min, inlet_x_max], highest pressure
      Outlets: upper surface y>0, x >= outlet_x_min, *lowest pressure* with TE bias
    """

    def __init__(
        self,
        porous: PorousConfig,
        design: DesignRadii,
        mu: float,
        k_spine: int = SPINE_NODES,
        x_start: float = 0.20,
        x_end: float = 0.95,
        y_spine: float = 0.0,
        spine_id_base: int = 80000,
    ):
        self.porous = porous
        self.design = design
        self.mu = float(mu)
        self.k_spine = int(k_spine)
        self.x_start = float(x_start)
        self.x_end = float(x_end)
        self.y_spine = float(y_spine)
        self.spine_id_base = int(spine_id_base)

    @staticmethod
    def _conductance(radius: float, length: float, mu: float) -> float:
        return (np.pi * radius**4) / (8.0 * mu * (length + 1e-15))

    @staticmethod
    def _nearest_index(points: np.ndarray, pt: np.ndarray) -> int:
        d2 = np.sum((points - pt[None, :])**2, axis=1)
        return int(np.argmin(d2))

    def build(self, aero: PanelMethodAero, pressure_baseline: np.ndarray) -> NetworkBuildResult:
        xc, yc = aero.XC, aero.YC
        P = np.asarray(pressure_baseline, dtype=float)

        G = nx.Graph()

        # spine nodes positions
        K = max(self.k_spine, 2)
        xs = np.linspace(self.x_start, self.x_end, K)
        ys = np.full_like(xs, self.y_spine)
        spine_pos = np.column_stack([xs, ys])

        spine_ids = [self.spine_id_base + i for i in range(K)]
        for sid, (x, y) in zip(spine_ids, spine_pos):
            G.add_node(int(sid), pos=(float(x), float(y)), type="internal")

        # connect spine trunk
        for i in range(K - 1):
            u = spine_ids[i]
            v = spine_ids[i + 1]
            pu = spine_pos[i]
            pv = spine_pos[i + 1]
            length = float(np.linalg.norm(pv - pu)) + 1e-15
            cond = self._conductance(self.design.r_spine, length, self.mu)
            G.add_edge(int(u), int(v), length=length, cond=cond, radius=self.design.r_spine, type="spine")

        # inlet candidates
        inlet_candidates = [
            i for i in range(len(xc))
            if (yc[i] < 0.0 and self.porous.inlet_x_min <= xc[i] <= self.porous.inlet_x_max)
        ]
        inlet_scores = [(i, float(P[i])) for i in inlet_candidates]  # high pressure
        inlet_scores.sort(key=lambda t: t[1], reverse=True)
        selected_inlets = [i for (i, _) in inlet_scores[: self.porous.n_inlets]]

        # outlet candidates (upper + near TE)
        outlet_candidates = [
            i for i in range(len(xc))
            if (yc[i] > 0.0 and xc[i] >= self.porous.outlet_x_min)
        ]
        # score: low pressure preferred; also prefer larger x slightly (TE bias)
        outlet_scores = []
        for i in outlet_candidates:
            score = float(P[i]) - 0.25 * float(xc[i])  # minimize this
            outlet_scores.append((i, score))
        outlet_scores.sort(key=lambda t: t[1])
        selected_outlets = [i for (i, _) in outlet_scores[: self.porous.n_outlets]]

        # attach branches
        def attach(panel_id: int, r_branch: float, etype: str):
            if not G.has_node(panel_id):
                G.add_node(int(panel_id), pos=(float(xc[panel_id]), float(yc[panel_id])),
                           type="boundary", panel_idx=int(panel_id))
            pt = np.array([xc[panel_id], yc[panel_id]], dtype=float)
            k = self._nearest_index(spine_pos, pt)
            sid = int(spine_ids[k])
            length = float(np.linalg.norm(pt - spine_pos[k])) + 1e-15
            cond = self._conductance(r_branch, length, self.mu)
            G.add_edge(int(panel_id), sid, length=length, cond=cond, radius=r_branch, type=etype)

        for pid in selected_inlets:
            attach(pid, self.design.r_branch_in, "branch_in")

        for pid in selected_outlets:
            attach(pid, self.design.r_branch_out, "branch_out")

        porous_pids = selected_inlets + selected_outlets
        print(f"   -> PipeTopology: K={K}, inlets={len(selected_inlets)}, outlets={len(selected_outlets)}, "
              f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        return NetworkBuildResult(G=G, porous_panels=porous_pids)


# ==============================================================================
# 5) INTERNAL PIPE NETWORK SOLVER (Sparse LU)
# ==============================================================================

class InternalPipeNetworkSolver:
    """
    Resistive pipe network:
      Q_ij = cond_ij * (P_i - P_j)

    Boundary nodes: prescribed pressure from panel external pressure.
    Internal nodes: solved with sparse LU on internal-only system.
    Leakage velocity per porous panel:
      v_leak(pid) = - Q_net(boundary_node) / A_surface(pid)
    """

    def __init__(self, G: nx.Graph, surface_pore_radius: float, area_mult: float):
        self.G = G
        self.surface_pore_radius = float(surface_pore_radius)
        self.area_mult = float(area_mult)

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

    def area_surface(self) -> float:
        return float(self.area_mult * np.pi * self.surface_pore_radius**2)

    def solve(self, P_boundary_by_panel: Dict[int, float]) -> Tuple[Dict[int, float], np.ndarray]:
        # boundary pressures
        P_bnode: Dict[int, float] = {}
        for bn in self.boundary_nodes:
            pid = int(self.G.nodes[bn].get("panel_idx", -1))
            P_bnode[bn] = float(P_boundary_by_panel.get(pid, 0.0))

        # internal RHS
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

        # full node pressures in stable order
        P_all = np.zeros(len(self.nodes))
        for n in self.internal_nodes:
            P_all[self.node_index[n]] = P_int[self.int_index[n]]
        for n in self.boundary_nodes:
            P_all[self.node_index[n]] = P_bnode[n]

        # leakage velocity per boundary panel
        A = self.area_surface()
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

            velocities_by_panel[pid] = -Q_net / (A + 1e-30)

        return velocities_by_panel, P_all


# ==============================================================================
# 6) COUPLING (RMS residual + smooth clip + adaptive relaxation)
# ==============================================================================

class RelaxedClippedCoupling:
    def __init__(self, relax: float, clip: float):
        self.relax = float(relax)
        self.clip = float(clip)
        self.last_max_abs_diff: float = 0.0

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
        diffs: List[float] = []
        max_abs = 0.0

        for pid in porous_pids:
            pid = int(pid)
            v_calc = float(V_new_dict.get(pid, 0.0))
            v_old = float(V[pid])

            v_new = self.relax * v_calc + (1.0 - self.relax) * v_old
            v_new = self.clip * np.tanh(v_new / (self.clip + 1e-30))

            d = v_new - v_old
            ad = abs(d)
            max_abs = max(max_abs, ad)
            diffs.append(d)
            V[pid] = v_new

        self.last_max_abs_diff = float(max_abs)
        if len(diffs) == 0:
            return V, 0.0
        d = np.asarray(diffs, dtype=float)
        rms = float(np.sqrt(np.mean(d * d)))
        return V, rms


class AdaptiveRelaxedClippedCoupling(RelaxedClippedCoupling):
    def __init__(
        self,
        relax_init: float,
        clip: float,
        relax_min: float = 1e-5,
        relax_max: float = 0.2,
        grow: float = 1.05,
        shrink: float = 0.5,
    ):
        super().__init__(relax=relax_init, clip=clip)
        self.relax_min = float(relax_min)
        self.relax_max = float(relax_max)
        self.grow = float(grow)
        self.shrink = float(shrink)
        self.prev_res: Optional[float] = None

    def update_leakage(self, V_old: np.ndarray, V_new_dict: Dict[int, float], porous_pids: List[int]) -> Tuple[np.ndarray, float]:
        V, res = super().update_leakage(V_old, V_new_dict, porous_pids)

        if self.prev_res is not None:
            if res > self.prev_res * 1.02:
                self.relax = max(self.relax * self.shrink, self.relax_min)
            else:
                self.relax = min(self.relax * self.grow, self.relax_max)

        self.prev_res = res
        return V, res


# ==============================================================================
# 7) POST: forces + reporting
# ==============================================================================

def compute_forces_like_iter1(Cp: np.ndarray, aero: PanelMethodAero) -> Tuple[float, float, float, float]:
    fx_elem = -Cp * aero.nx * aero.L
    fy_elem = -Cp * aero.ny * aero.L
    Fx = float(np.sum(fx_elem))
    Fy = float(np.sum(fy_elem))
    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    return CL, CD, Fx, Fy


# ==============================================================================
# 8) REPORTER (same as your current; unchanged)
# ==============================================================================

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

    # ---- plotting methods kept exactly as your version ----
    # (omitted here for brevity in this message)
    # IMPORTANT: paste your existing Reporter.plot_case and Reporter.plot_sweep_graphs here unchanged.

    def plot_case(self, *args, **kwargs):
        # <<< PASTE YOUR EXISTING plot_case HERE (UNCHANGED) >>>
        pass

    def plot_sweep_graphs(self, *args, **kwargs):
        # <<< PASTE YOUR EXISTING plot_sweep_graphs HERE (UNCHANGED) >>>
        pass


# ==============================================================================
# 9) SINGLE CASE RUN + SWEEP
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

    # solid baseline
    Cp_solid = aero.solve(np.zeros(aero.N))
    CL_solid, CD_solid, _, _ = compute_forces_like_iter1(Cp_solid, aero)

    # build pipe topology network from baseline pressure (Pa)
    P_solid = cfg.flow.p_inf + q_inf * Cp_solid
    builder = PipeTopologyBuilder(cfg.porous, cfg.design, cfg.flow.mu, k_spine=SPINE_NODES)
    build_res = builder.build(aero, P_solid)

    internal = InternalPipeNetworkSolver(
        build_res.G,
        surface_pore_radius=cfg.porous.surface_pore_radius,
        area_mult=cfg.porous.area_mult,
    )

    coupling = AdaptiveRelaxedClippedCoupling(
        relax_init=cfg.coupling.relax,
        clip=cfg.coupling.clip,
        relax_min=1e-5,
        relax_max=0.2,
        grow=1.05,
        shrink=0.5,
    )

    V_leak = np.zeros(aero.N)
    Cp = Cp_solid.copy()
    final_P_nodes = np.zeros(len(internal.nodes))
    converged = False
    it_used = 0

    for it in range(int(cfg.coupling.max_iter)):
        it_used = it

        Cp = aero.solve(V_leak)
        P_ext = coupling.external_pressure(Cp, q_inf=q_inf, p_inf=cfg.flow.p_inf)
        P_map = coupling.boundary_map(P_ext, build_res.porous_panels)

        V_calc, P_nodes_iter = internal.solve(P_map)
        final_P_nodes = P_nodes_iter

        V_leak, res_rms = coupling.update_leakage(V_leak, V_calc, build_res.porous_panels)

        if (res_rms < cfg.coupling.tol) and (it >= cfg.coupling.min_iters):
            converged = True
            break

        if it % 10 == 0:
            print(
                f"   AoA {alpha_deg:+.0f}° | Iter {it:03d} "
                f"| res_rms={res_rms:.3e} | res_max={coupling.last_max_abs_diff:.3e} "
                f"| relax={coupling.relax:.3e}"
            )

    CL, CD, _, _ = compute_forces_like_iter1(Cp, aero)

    case_dir = os.path.join(cfg.output.out_dir, f"aoa_{alpha_deg:+03.0f}".replace("+", "p").replace("-", "m"))
    reporter.save_case_csv(case_dir, aero, Cp_solid, Cp, V_leak, CL_solid, CL, CD_solid, CD)

    if int(round(alpha_deg)) in PLOT_AOAS and cfg.plot.make_figures:
        # requires you to paste your Reporter.plot_case implementation back in
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


def run_simulation(cfg: SimConfig) -> None:
    print("=== iter2 porous-airfoil simulation (PIPE topology) ===")
    print(f"Output dir: {cfg.output.out_dir}")
    os.makedirs(cfg.output.out_dir, exist_ok=True)

    print("Config (key parts):")
    print(f"  Airfoil: NACA {cfg.airfoil.name}, panels={cfg.airfoil.n_panels}")
    print(f"  Flow: Re={cfg.flow.reynolds}, rho={cfg.flow.rho}, mu={cfg.flow.mu}, V_inf={cfg.flow.v_inf():.2f} m/s")
    print(f"  Design radii: {asdict(cfg.design)}")
    print(f"  Porous area_mult: {cfg.porous.area_mult}")
    print(f"  Coupling: max_iter={cfg.coupling.max_iter}, relax={cfg.coupling.relax}, tol={cfg.coupling.tol}, clip={cfg.coupling.clip}")
    print(f"  Sweep: {cfg.sweep.alpha_start}..{cfg.sweep.alpha_end} step {cfg.sweep.alpha_step}")

    geom = make_geometry(cfg)
    aero = PanelMethodAero(geom, cfg.flow)
    reporter = Reporter(cfg.output.out_dir, cfg.plot)

    cp_cache: Dict[int, Dict[str, np.ndarray]] = {}
    results: List[CaseResult] = []

    if cfg.sweep.do_sweep:
        for alpha in range(cfg.sweep.alpha_start, cfg.sweep.alpha_end + 1, cfg.sweep.alpha_step):
            res = run_case(cfg, aero, float(alpha), reporter)
            results.append(res)

            # optional: cache Cp for compare plots (only if plot_sweep_graphs is pasted back)
            if int(alpha) in PLOT_AOAS:
                aero.set_alpha_deg(float(alpha))
                Cp_solid = aero.solve(np.zeros(aero.N))
                # NOTE: we keep cp_cache minimal here; if you paste back plot_sweep_graphs,
                # you can also re-run coupling to store Cp_porous as in your previous file.
                cp_cache[int(alpha)] = {"Cp_solid": Cp_solid.copy(), "Cp_porous": Cp_solid.copy()}
    else:
        results.append(run_case(cfg, aero, cfg.flow.alpha_deg, reporter))

    rows = [{
        "alpha_deg": r.alpha_deg,
        "CL_solid": r.CL_solid,
        "CL_porous": r.CL_porous,
        "CD_solid": r.CD_solid,
        "CD_porous": r.CD_porous,
        "iters": r.iters,
        "converged": int(r.converged),
    } for r in results]

    if rows:
        path = reporter.save_sweep_summary(rows)
        print(f"-> Sweep summary saved: {path}")

        # If you paste back plot_sweep_graphs, you can re-enable:
        # reporter.plot_sweep_graphs(results, cp_cache, xc=aero.XC)

    print("\nAoA sweep results:")
    print(" alpha |  CL_solid  CL_porous |  CD_solid  CD_porous | iters conv")
    for r in results:
        print(f"{r.alpha_deg:>+5.0f} | {r.CL_solid:>8.4f} {r.CL_porous:>9.4f} | {r.CD_solid:>8.4f} {r.CD_porous:>9.4f} | {r.iters:>5d}  {int(r.converged)}")


def main() -> None:
    cfg = SimConfig(
        design=DesignRadii(
            r_branch_in=0.0019590275837019517,
            r_branch_out=0.005243981588587115,
            r_spine=0.008,
        ),
        porous=PorousConfig(
            area_mult=50.0,  # << this is now your porosity/number-of-pores knob
        ),
        sweep=SweepConfig(do_sweep=True, alpha_start=-5, alpha_end=10, alpha_step=1),
    )
    run_simulation(cfg)


if __name__ == "__main__":
    main()
