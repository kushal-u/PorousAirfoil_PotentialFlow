# =========================
# iter1.py
# =========================
"""
Core library: geometry, panel method, network builder, internal-flow solver, and coupled runner.

Key improvements vs original:
- PanelMethod caches geometry-dependent matrices + LU; AoA is updated via set_alpha() (no rebuild per AoA).
- InternalFlowSolver precomputes sparse matrix once; boundary RHS assembly uses precomputed neighbor lists.
- Coupling uses NumPy arrays (porous_idx) instead of Python dicts in the hot loop.
- Results are standardized via dataclasses, suitable for saving/plotting without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -------------------------
# Params / Results
# -------------------------

@dataclass
class FlowParams:
    rho: float = 1.225
    mu: float = 1.8e-5
    re: float = 1.0e6
    chord: float = 1.0
    v_inf: float = 1.0  # can be overwritten by set_from_re()

    def set_from_re(self) -> None:
        # Re = rho * V * c / mu  ->  V = Re * mu / (rho*c)
        self.v_inf = (self.re * self.mu) / (self.rho * self.chord)


@dataclass
class CouplingParams:
    max_iter: int = 200
    tol: float = 1e-6
    relaxation: float = 0.03
    v_clip: Tuple[float, float] = (0.0, 5.0)  # avoid negative / runaway leak speeds
    anderson_m: int = 0  # set >0 to enable Anderson acceleration (simple)


@dataclass
class CaseResult:
    aoa_deg: float
    converged: bool
    n_iter: int

    cl_solid: float
    cd_solid: float
    cl_porous: float
    cd_porous: float

    mean_leak: float
    total_leak: float

    # arrays (for plotting)
    x_mid: np.ndarray
    cp_solid: np.ndarray
    cp_porous: np.ndarray
    v_leak_panels: np.ndarray

    # port-level info (aligned with porous ports)
    p_ports: np.ndarray
    q_ports: np.ndarray

    # convergence traces
    residual_hist: np.ndarray
    mean_leak_hist: np.ndarray

    def to_summary_row(self) -> Dict:
        d = asdict(self)
        # drop big arrays for summary
        for k in ["x_mid", "cp_solid", "cp_porous", "v_leak_panels", "p_ports", "q_ports", "residual_hist", "mean_leak_hist"]:
            d.pop(k, None)
        return d


# -------------------------
# Geometry
# -------------------------

def naca4(m: float, p: float, t: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    NACA 4-digit airfoil coordinates (closed surface, TE->LE->TE).
    Returns x,y arrays sized (2*n-1,).
    """
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2  # cosine spacing [0..1]
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if p > 1e-12:
        i1 = x < p
        i2 = ~i1
        yc[i1] = (m / p**2) * (2 * p * x[i1] - x[i1] ** 2)
        dyc_dx[i1] = (2 * m / p**2) * (p - x[i1])

        yc[i2] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[i2] - x[i2] ** 2)
        dyc_dx[i2] = (2 * m / (1 - p) ** 2) * (p - x[i2])

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Assemble closed surface: TE upper -> LE -> TE lower
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y


def panel_midpoints(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return 0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:])


# -------------------------
# Panel Method (constant-strength source + vortex, Kutta)
# -------------------------

class PanelMethod:
    """
    2D inviscid panel method with cached geometry system matrix.
    AoA only changes RHS (freestream projection), so use set_alpha() instead of reconstructing.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, aoa_deg: float, flow: FlowParams):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        if self.X.ndim != 1 or self.Y.ndim != 1 or self.X.size != self.Y.size:
            raise ValueError("X,Y must be 1D arrays of equal length.")
        if self.X.size < 4:
            raise ValueError("Need at least 3 panels.")

        self.flow = flow
        self.N = self.X.size - 1  # number of panels

        # Precompute geometry-dependent things
        self.x_mid, self.y_mid = panel_midpoints(self.X, self.Y)
        dx = self.X[1:] - self.X[:-1]
        dy = self.Y[1:] - self.Y[:-1]
        self.s = np.sqrt(dx**2 + dy**2)
        self.t_hat = np.vstack([dx / self.s, dy / self.s]).T  # tangential unit vector
        self.n_hat = np.vstack([self.t_hat[:, 1], -self.t_hat[:, 0]]).T  # outward normal

        # Influence matrices + LU for system
        self._build_cached_system()

        # AoA-dependent freestream
        self.set_alpha(aoa_deg)

    def set_alpha(self, aoa_deg: float) -> None:
        self.aoa_deg = float(aoa_deg)
        a = np.deg2rad(self.aoa_deg)
        self.vx_inf = self.flow.v_inf * np.cos(a)
        self.vy_inf = self.flow.v_inf * np.sin(a)
        v_inf_vec = np.array([self.vx_inf, self.vy_inf])[None, :]
        self.vinf_n = np.sum(v_inf_vec * self.n_hat, axis=1)
        self.vinf_t = np.sum(v_inf_vec * self.t_hat, axis=1)

    def _build_cached_system(self) -> None:
        # Build influence matrix for source panels (normal velocity) and vortex term.
        # This is a standard approach; details kept compact.
        N = self.N
        xmid = self.x_mid
        ymid = self.y_mid
        X = self.X
        Y = self.Y
        s = self.s
        t_hat = self.t_hat
        n_hat = self.n_hat

        # Local transforms for each panel
        phi = np.arctan2(Y[1:] - Y[:-1], X[1:] - X[:-1])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        # Normal influence from unit-strength source on panel j to control point i
        An = np.zeros((N, N), dtype=float)
        At = np.zeros((N, N), dtype=float)

        for i in range(N):
            for j in range(N):
                if i == j:
                    # self influence of source: 0.5 for normal, 0 for tangential (classical)
                    An[i, j] = 0.5
                    At[i, j] = 0.0
                    continue

                # Transform control point i into panel-j local coords
                x_rel = xmid[i] - X[j]
                y_rel = ymid[i] - Y[j]
                x_local = x_rel * cosphi[j] + y_rel * sinphi[j]
                y_local = -x_rel * sinphi[j] + y_rel * cosphi[j]

                # Source panel influence integrals
                x2 = x_local - s[j]
                r1 = np.hypot(x_local, y_local)
                r2 = np.hypot(x2, y_local)
                theta1 = np.arctan2(y_local, x_local)
                theta2 = np.arctan2(y_local, x2)

                # Avoid numerical issues
                if r1 < 1e-14 or r2 < 1e-14:
                    r1 = max(r1, 1e-14)
                    r2 = max(r2, 1e-14)

                # Velocity induced by unit source in panel-local coordinates
                u_local = (1 / (2 * np.pi)) * np.log(r2 / r1)
                v_local = (1 / (2 * np.pi)) * (theta2 - theta1)

                # Transform back to global
                u = u_local * cosphi[j] - v_local * sinphi[j]
                v = u_local * sinphi[j] + v_local * cosphi[j]

                # Project onto i's normal/tangent
                An[i, j] = u * n_hat[i, 0] + v * n_hat[i, 1]
                At[i, j] = u * t_hat[i, 0] + v * t_hat[i, 1]

        # Vortex influence: sum of tangential components of sources approximations can be used,
        # but here we use a simple "constant vortex strength gamma" term:
        # normal velocity induced by unit vortex on all panels is -At[:, :] summed over panels.
        # A common formulation: An*sigma + B*gamma = -Vinf_n, with B = -sum(At over all panels)
        Bn = -np.sum(At, axis=1)  # shape (N,)

        # Kutta condition row: enforce tangential velocity at TE upper/lower sums to zero.
        # A compact Kutta: (At[0,:] + At[-1,:]) * sigma + (Bt) * gamma = -(Vinf_t[0] + Vinf_t[-1])
        Bt = (np.sum(An[0, :]) + np.sum(An[-1, :]))  # crude coupling for gamma in Kutta row
        kutta_row = np.zeros(N + 1, dtype=float)
        kutta_row[:N] = At[0, :] + At[-1, :]
        kutta_row[N] = Bt

        # Assemble system matrix A (geometry-only except Kutta RHS)
        A = np.zeros((N + 1, N + 1), dtype=float)
        A[:N, :N] = An
        A[:N, N] = Bn
        A[N, :] = kutta_row

        self.An = An
        self.At = At
        self.Bn = Bn
        self.A = A
        self._lu = scipy.linalg.lu_factor(A)

    def solve(self, v_leak: Optional[np.ndarray] = None) -> Dict[str, np.ndarray | float]:
        """
        Solve for panel source strengths + circulation and compute Cp, CL, CD.
        v_leak: normal velocity injected at panels (positive outward) [shape N], zeros if None.
        """
        N = self.N
        v_leak = np.zeros(N, dtype=float) if v_leak is None else np.asarray(v_leak, dtype=float)
        if v_leak.shape != (N,):
            raise ValueError(f"v_leak must have shape ({N},)")

        # RHS: - (Vinf_n + v_leak)
        b = np.zeros(N + 1, dtype=float)
        b[:N] = -(self.vinf_n + v_leak)
        # Kutta RHS depends on AoA (tangential freestream)
        b[N] = -(self.vinf_t[0] + self.vinf_t[-1])

        sol = scipy.linalg.lu_solve(self._lu, b)
        sigma = sol[:N]
        gamma = sol[N]

        # Tangential velocity at control points
        vt = self.vinf_t + self.At @ sigma + gamma * np.sum(self.An, axis=1)

        cp = 1.0 - (vt / self.flow.v_inf) ** 2

        # Force coefficients (very simplified integration in body axes)
        # Normal and axial force per panel from Cp * n * ds; convert to lift/drag in freestream axes.
        # Note: This is adequate for comparative/optimization use but not a high-fidelity force model.
        dFx = -cp * self.n_hat[:, 0] * self.s
        dFy = -cp * self.n_hat[:, 1] * self.s
        Fx = np.sum(dFx)
        Fy = np.sum(dFy)

        a = np.deg2rad(self.aoa_deg)
        # Lift positive perpendicular to freestream, drag along freestream
        L = -Fx * np.sin(a) + Fy * np.cos(a)
        D = Fx * np.cos(a) + Fy * np.sin(a)

        q = 0.5 * self.flow.rho * self.flow.v_inf**2
        cl = L / (q * self.flow.chord)
        cd = D / (q * self.flow.chord)

        return {
            "sigma": sigma,
            "gamma": gamma,
            "vt": vt,
            "cp": cp,
            "cl": float(cl),
            "cd": float(cd),
            "x_mid": self.x_mid.copy(),
        }


# -------------------------
# Network builder (spine manifold, fixed ports)
# -------------------------

@dataclass
class PipeDesign:
    r_in: float = 2.5e-3
    r_out: float = 2.5e-3
    r_spine: float = 3.5e-3


@dataclass
class ManifoldNetwork:
    """
    Array-based representation of a pipe network for fast solves.

    Nodes: 0..(nb-1) are boundary ports, nb..(nb+ni-1) are internal nodes.
    Edges: (u, v, conductance) with u/v in internal+boundary index space.

    boundary ports are aligned with porous panels order (porous_idx).
    """
    nb: int
    ni: int
    edges_u: np.ndarray
    edges_v: np.ndarray
    edges_g: np.ndarray  # conductance-like term for linear model
    boundary_pos: np.ndarray  # (nb,2) for viz (optional)
    internal_pos: np.ndarray  # (ni,2) for viz (optional)


def _poiseuille_conductance(r: float, L: float, mu: float) -> float:
    # Q = (pi r^4 / (8 mu L)) * dP  -> conductance = pi r^4 / (8 mu L)
    L = max(float(L), 1e-12)
    r = max(float(r), 1e-12)
    return (np.pi * r**4) / (8.0 * mu * L)


def generate_spine_manifold_network_fixed_ports(
    x_mid: np.ndarray,
    y_mid: np.ndarray,
    cp_ref: np.ndarray,
    porous_k: int,
    design: PipeDesign,
    flow: FlowParams,
) -> Tuple[ManifoldNetwork, np.ndarray]:
    """
    Build a simple "spine + branches" manifold network and select porous ports once using cp_ref.
    Returns (network, porous_idx).

    porous_k: number of porous ports to select (highest suction magnitude).
    """
    N = x_mid.size
    porous_k = int(np.clip(porous_k, 1, N))

    # Select porous panels (fixed for whole sweep)
    # pick most negative cp (highest suction) => smallest cp values
    porous_idx = np.argsort(cp_ref)[:porous_k]
    porous_idx.sort()

    # Create boundary port positions at those panel midpoints
    bpos = np.vstack([x_mid[porous_idx], y_mid[porous_idx]]).T
    nb = porous_k

    # Create internal spine nodes along chord line at y=0 (or mean y)
    spine_n = max(3, porous_k // 3)
    xs = np.linspace(np.min(x_mid), np.max(x_mid), spine_n)
    ys = np.full_like(xs, float(np.mean(y_mid)))
    ipos = np.vstack([xs, ys]).T
    ni = spine_n

    # Build edges: each boundary port connects to nearest spine node (in/out),
    # and spine nodes connected in a chain.
    edges_u: List[int] = []
    edges_v: List[int] = []
    edges_g: List[float] = []

    # indices in combined node space: boundary [0..nb-1], internal [nb..nb+ni-1]
    internal_offset = nb

    # Branch connections
    for bi in range(nb):
        p = bpos[bi]
        d = np.linalg.norm(ipos - p[None, :], axis=1)
        j = int(np.argmin(d))
        u = bi
        v = internal_offset + j
        L = float(d[j])
        g = _poiseuille_conductance(design.r_in, L, flow.mu)
        edges_u.append(u); edges_v.append(v); edges_g.append(g)

    # Spine chain
    for j in range(ni - 1):
        u = internal_offset + j
        v = internal_offset + (j + 1)
        L = float(np.linalg.norm(ipos[j + 1] - ipos[j]))
        g = _poiseuille_conductance(design.r_spine, L, flow.mu)
        edges_u.append(u); edges_v.append(v); edges_g.append(g)

    net = ManifoldNetwork(
        nb=nb,
        ni=ni,
        edges_u=np.asarray(edges_u, dtype=int),
        edges_v=np.asarray(edges_v, dtype=int),
        edges_g=np.asarray(edges_g, dtype=float),
        boundary_pos=bpos,
        internal_pos=ipos,
    )
    return net, porous_idx


# -------------------------
# Internal solver (linear resistive network with boundary pressures)
# -------------------------

class InternalFlowSolver:
    """
    Solves internal node pressures given boundary node pressures via linear pipe network:
    For each edge (i,j): Q = g * (P_i - P_j)

    Boundary nodes have prescribed pressures (Dirichlet).
    Internal nodes are unknown; apply mass balance at internal nodes => linear system.
    """

    def __init__(self, net: ManifoldNetwork):
        self.net = net
        self.nb = net.nb
        self.ni = net.ni
        self.nt = self.nb + self.ni

        # Split edges by types
        u = net.edges_u
        v = net.edges_v
        g = net.edges_g

        # Precompute adjacency lists for internal node equations:
        # For internal node k (global index = nb + k): sum g*(P_k - P_neighbor)=0
        self.int_neighbors: List[np.ndarray] = []
        self.int_g: List[np.ndarray] = []
        self.int_boundary_neighbors: List[np.ndarray] = []
        self.int_boundary_g: List[np.ndarray] = []

        # Build sparse matrix for internal unknowns
        A = sp.lil_matrix((self.ni, self.ni), dtype=float)

        # For RHS contributions from boundary pressures
        # We'll store boundary neighbor indices and conductances per internal node.

        for k in range(self.ni):
            self.int_neighbors.append(np.empty(0, dtype=int))
            self.int_g.append(np.empty(0, dtype=float))
            self.int_boundary_neighbors.append(np.empty(0, dtype=int))
            self.int_boundary_g.append(np.empty(0, dtype=float))

        # Build connections by scanning edges once
        for uu, vv, gg in zip(u, v, g):
            uu = int(uu); vv = int(vv); gg = float(gg)
            uu_is_internal = uu >= self.nb
            vv_is_internal = vv >= self.nb

            if uu_is_internal and vv_is_internal:
                ku = uu - self.nb
                kv = vv - self.nb
                # Off-diagonals
                A[ku, kv] -= gg
                A[kv, ku] -= gg
                # Diagonals
                A[ku, ku] += gg
                A[kv, kv] += gg

            elif uu_is_internal and (not vv_is_internal):
                ku = uu - self.nb
                # diagonal gets gg, RHS will add gg * P_boundary[vv]
                A[ku, ku] += gg
                self.int_boundary_neighbors[ku] = np.append(self.int_boundary_neighbors[ku], vv)
                self.int_boundary_g[ku] = np.append(self.int_boundary_g[ku], gg)

            elif (not uu_is_internal) and vv_is_internal:
                kv = vv - self.nb
                A[kv, kv] += gg
                self.int_boundary_neighbors[kv] = np.append(self.int_boundary_neighbors[kv], uu)
                self.int_boundary_g[kv] = np.append(self.int_boundary_g[kv], gg)
            else:
                # boundary-boundary edge (rare here) -> ignore for internal solve
                pass

        self.A = A.tocsc()
        self._lu = spla.splu(self.A)

    def solve_internal_pressures(self, p_boundary: np.ndarray) -> np.ndarray:
        p_boundary = np.asarray(p_boundary, dtype=float)
        if p_boundary.shape != (self.nb,):
            raise ValueError(f"p_boundary must have shape ({self.nb},)")

        rhs = np.zeros(self.ni, dtype=float)
        for k in range(self.ni):
            bnei = self.int_boundary_neighbors[k]
            bg = self.int_boundary_g[k]
            if bnei.size:
                rhs[k] = np.sum(bg * p_boundary[bnei])

        p_internal = self._lu.solve(rhs)
        return p_internal

    def compute_boundary_fluxes(self, p_boundary: np.ndarray, p_internal: np.ndarray) -> np.ndarray:
        """
        Return net outflow from each boundary node (positive if leaving boundary into network).
        """
        p_boundary = np.asarray(p_boundary, dtype=float)
        p_internal = np.asarray(p_internal, dtype=float)
        if p_boundary.shape != (self.nb,) or p_internal.shape != (self.ni,):
            raise ValueError("bad shapes for p_boundary/p_internal")

        Qb = np.zeros(self.nb, dtype=float)

        u = self.net.edges_u
        v = self.net.edges_v
        g = self.net.edges_g

        for uu, vv, gg in zip(u, v, g):
            uu = int(uu); vv = int(vv); gg = float(gg)
            if uu < self.nb and vv >= self.nb:
                # boundary -> internal
                Pi = p_boundary[uu]
                Pj = p_internal[vv - self.nb]
                Q = gg * (Pi - Pj)
                Qb[uu] += Q
            elif vv < self.nb and uu >= self.nb:
                Pi = p_internal[uu - self.nb]
                Pj = p_boundary[vv]
                Q = gg * (Pi - Pj)
                # flow from internal to boundary is negative out of boundary
                Qb[vv] -= Q

        return Qb


# -------------------------
# Coupled runner
# -------------------------

def run_coupled_case(
    aoa_deg: float,
    aero: PanelMethod,
    net: ManifoldNetwork,
    porous_idx: np.ndarray,
    flow: FlowParams,
    coupling: CouplingParams,
    warm_vleak: Optional[np.ndarray] = None,
) -> Tuple[CaseResult, np.ndarray]:
    """
    Runs: solid aero -> coupled porous iterations (aero <-> internal network).
    Returns (CaseResult, warm_vleak_for_next_case) where warm_vleak is porous-aligned.
    """
    aero.set_alpha(aoa_deg)

    # Solid baseline (no leak)
    solid = aero.solve(v_leak=np.zeros(aero.N))
    cp_solid = solid["cp"]
    cl_solid = float(solid["cl"])
    cd_solid = float(solid["cd"])
    x_mid = solid["x_mid"]

    # Setup leak arrays
    porous_idx = np.asarray(porous_idx, dtype=int)
    nb = net.nb
    if porous_idx.shape != (nb,):
        raise ValueError("porous_idx must be aligned with number of boundary ports net.nb")

    # leak on all panels but only porous panels updated
    v_leak_panels = np.zeros(aero.N, dtype=float)
    v_port = np.zeros(nb, dtype=float) if warm_vleak is None else np.asarray(warm_vleak, dtype=float).copy()
    if v_port.shape != (nb,):
        v_port = np.zeros(nb, dtype=float)

    # internal solver
    internal = InternalFlowSolver(net)

    residual_hist = []
    mean_leak_hist = []

    # (optional) Anderson buffers for porous v
    m = int(coupling.anderson_m)
    if m > 0:
        F_hist = []
        X_hist = []

    converged = False
    for it in range(1, coupling.max_iter + 1):
        # put porous leaks into full panel array
        v_leak_panels[:] = 0.0
        v_leak_panels[porous_idx] = v_port

        # aero solve with leak
        porous = aero.solve(v_leak=v_leak_panels)
        cp_porous = porous["cp"]
        cl_porous = float(porous["cl"])
        cd_porous = float(porous["cd"])

        # boundary pressures from Cp: P = P_inf + q * Cp
        q = 0.5 * flow.rho * flow.v_inf**2
        p_inf = 0.0  # gauge
        p_panels = p_inf + q * cp_porous
        p_ports = p_panels[porous_idx]  # aligned with boundary ports

        # internal solve
        p_int = internal.solve_internal_pressures(p_ports)
        q_ports = internal.compute_boundary_fluxes(p_ports, p_int)

        # Convert port flow to an equivalent leakage speed.
        # Here we use a simple proportional mapping:
        # v_new ~ Q / A_port, where A_port is an effective area per port.
        # Choose A_port based on local panel length * unit depth.
        # (This keeps behavior consistent and vectorized; adjust if you have a more physical mapping.)
        panel_len = aero.s[porous_idx]
        A_port = np.maximum(panel_len * 1.0, 1e-12)  # unit depth
        v_new = q_ports / A_port

        # Relaxation + clip
        v_next = coupling.relaxation * v_new + (1.0 - coupling.relaxation) * v_port
        v_next = np.clip(v_next, coupling.v_clip[0], coupling.v_clip[1])

        # Residual for convergence
        res = float(np.max(np.abs(v_next - v_port)))
        residual_hist.append(res)
        mean_leak_hist.append(float(np.mean(v_next)))

        # Optional Anderson acceleration on porous v
        if m > 0:
            F_hist.append((v_next - v_port).copy())
            X_hist.append(v_port.copy())
            if len(F_hist) > m:
                F_hist.pop(0); X_hist.pop(0)

            if len(F_hist) >= 2:
                # Solve least squares for mixing of past residuals
                Fm = np.column_stack(F_hist)  # (nb, k)
                # minimize ||Fm * c|| s.t. sum c = 1 -> simple unconstrained approx:
                # c = argmin ||Fm c||, use pseudoinverse then normalize
                try:
                    c = np.linalg.lstsq(Fm, np.zeros(nb), rcond=None)[0]
                except np.linalg.LinAlgError:
                    c = None
                if c is not None and np.all(np.isfinite(c)):
                    if np.linalg.norm(c) > 0:
                        c = c / np.sum(c) if np.sum(c) != 0 else c
                        x_mix = np.zeros_like(v_port)
                        for ci, xi in zip(c, X_hist):
                            x_mix += ci * xi
                        v_next = np.clip(x_mix + (v_next - v_port), coupling.v_clip[0], coupling.v_clip[1])

        v_port = v_next

        if res < coupling.tol:
            converged = True
            break

    # final recompute arrays for result consistency
    v_leak_panels[:] = 0.0
    v_leak_panels[porous_idx] = v_port
    porous_final = aero.solve(v_leak=v_leak_panels)
    cp_porous_final = porous_final["cp"]
    cl_porous_final = float(porous_final["cl"])
    cd_porous_final = float(porous_final["cd"])

    q = 0.5 * flow.rho * flow.v_inf**2
    p_panels = 0.0 + q * cp_porous_final
    p_ports_final = p_panels[porous_idx]
    p_int_final = internal.solve_internal_pressures(p_ports_final)
    q_ports_final = internal.compute_boundary_fluxes(p_ports_final, p_int_final)

    result = CaseResult(
        aoa_deg=float(aoa_deg),
        converged=converged,
        n_iter=it,
        cl_solid=cl_solid,
        cd_solid=cd_solid,
        cl_porous=cl_porous_final,
        cd_porous=cd_porous_final,
        mean_leak=float(np.mean(v_port)),
        total_leak=float(np.sum(q_ports_final)),
        x_mid=np.asarray(x_mid),
        cp_solid=np.asarray(cp_solid),
        cp_porous=np.asarray(cp_porous_final),
        v_leak_panels=np.asarray(v_leak_panels),
        p_ports=np.asarray(p_ports_final),
        q_ports=np.asarray(q_ports_final),
        residual_hist=np.asarray(residual_hist, dtype=float),
        mean_leak_hist=np.asarray(mean_leak_hist, dtype=float),
    )
    return result, v_port


# -------------------------
# Minimal demo (optional)
# -------------------------

def _demo() -> None:
    flow = FlowParams(re=1e6, chord=1.0)
    flow.set_from_re()

    X, Y = naca4(m=0.02, p=0.4, t=0.12, n=240)
    aero = PanelMethod(X, Y, aoa_deg=2.0, flow=flow)

    # Reference CP to select ports
    solid = aero.solve()
    cp_ref = solid["cp"]
    net, porous_idx = generate_spine_manifold_network_fixed_ports(
        x_mid=solid["x_mid"], y_mid=aero.y_mid, cp_ref=cp_ref, porous_k=24,
        design=PipeDesign(), flow=flow,
    )

    coupling = CouplingParams(max_iter=120, tol=1e-6, relaxation=0.03, anderson_m=0)
    res, _ = run_coupled_case(aoa_deg=4.0, aero=aero, net=net, porous_idx=porous_idx, flow=flow, coupling=coupling)

    print("Demo result:", res.to_summary_row())


if __name__ == "__main__":
    _demo()
