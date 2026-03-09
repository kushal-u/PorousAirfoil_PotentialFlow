"""
Optimized Porous-Airfoil Panel + Internal Network Solver
=======================================================

Goal
----
Rewrite your full script into an "optimized architecture" while keeping results
as similar as possible to your original code.

Key optimizations (compared to your original)
---------------------------------------------
1) Panel method:
   - Build the (N+1)x(N+1) linear system matrix A ONCE.
   - LU-factorize A ONCE.
   - Each iteration only updates RHS b and does a fast LU backsolve.

2) Internal flow solver:
   - Build sparse system structure ONCE (including node ordering).
   - Factorize ONCE (splu).
   - Each iteration only updates boundary pressures (RHS) and solves quickly.

3) Consistency & correctness (also improves stability):
   - Store graph node ordering once and reuse everywhere (plotting included).
   - Store edge radius and compute internal pipe velocities using Q/A (not a
     global PORE_RADIUS for every edge).
   - Avoid repeated allocations and repeated computations inside iteration loop.

Outputs
-------
- Saves CSV and figures to output folder (same as your original intention):
  01_Geometry_and_Cp.png
  03_Pressure_Vectors.png
  05_Flow_Field.png
  06_Internal_Flow_Map_Contour.png
  simulation_data.csv

Notes
-----
- Results should remain very similar; any differences should primarily come
  from fixing inconsistent assumptions (e.g., internal velocity calculation)
  and from numerical conditioning improvements.
"""

import os
import warnings
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg
import matplotlib

# Headless-safe backend (must be set BEFORE importing pyplot)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
from scipy.interpolate import griddata


warnings.filterwarnings("ignore")


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Geometry & Mesh
AIRFOIL_NAME = "0018"
N_PANELS = 320        # Number of panels (closed loop => N points = N+1)
N_PORES = 50          # Not directly used in this plenum model (kept for config)

# Porous Region (kept for future extensions)
X_START = 0.01
X_END = 0.99

# IMPORTANT: This is used as the "surface pore area" radius in leakage velocity
# conversion unless you choose to change that mapping.
# Your old comment "2mm" did NOT match 5000e-6. 5000e-6 is 5mm radius.
PORE_RADIUS = 5000e-6

# Physics
REYNOLDS_NUM = 250000
ANGLE_OF_ATTACK = 6.0
RHO = 1.225
MU = 1.78e-5
P_INF = 0.0
CHORD = 1.0

# Velocity (based on Re)
V_INF = (REYNOLDS_NUM * MU) / (RHO * CHORD)

# Coupling iteration
MAX_ITER = 100
RELAXATION = 0.01
CONVERGENCE_TOL = 1e-8

# Plotting / field
FLOWFIELD_RES = 100   # external flow mesh resolution
INTERNAL_GRID_RES = 200  # internal contour grid density


# ==============================================================================
# 2. GEOMETRY GENERATION (NACA 4-digit)
# ==============================================================================
def naca4(number: str, n_panels: int = 160):
    """
    Returns closed airfoil coordinates for a NACA 4-digit airfoil.

    Your original approach uses cosine clustering and the "closed TE" coefficient
    0.1036 in thickness distribution.
    """
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0

    # Cosine clustering for half the surface
    beta = np.linspace(0.0, np.pi, n_panels // 2 + 1)
    x = (1.0 - np.cos(beta)) / 2.0

    # Thickness distribution (closed trailing edge)
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    # Camber line: for symmetric 00xx => yc = 0
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

    # Concatenate: TE upper -> LE -> TE lower (closed curve)
    X = np.concatenate((xu[::-1], xl[1:]))
    Y = np.concatenate((yu[::-1], yl[1:]))

    return X, Y


# ==============================================================================
# 3. AERODYNAMIC SOLVER (Panel Method) - OPTIMIZED
# ==============================================================================
class PanelMethod:
    """
    Source+Vortex panel method with:
    - Influence matrices built once
    - System matrix A built once
    - LU factorization cached
    - Freestream normal/tangential components cached

    solve(V_leakage) becomes:
        b = constant + V_leakage term
        x = LU_solve(A, b)
        compute Cp
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, alpha_deg: float):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        self.alpha = np.radians(alpha_deg)
        self.N = len(self.X) - 1

        # Panel geometry
        self.XC = (self.X[:-1] + self.X[1:]) / 2.0
        self.YC = (self.Y[:-1] + self.Y[1:]) / 2.0
        self.dx = self.X[1:] - self.X[:-1]
        self.dy = self.Y[1:] - self.Y[:-1]
        self.L = np.sqrt(self.dx**2 + self.dy**2)

        # Tangents and normals
        self.tx = self.dx / self.L
        self.ty = self.dy / self.L
        self.nx = self.dy / self.L
        self.ny = -self.dx / self.L

        # Precompute freestream vector and its components on panels (constant)
        self.Vinf_x = V_INF * np.cos(self.alpha)
        self.Vinf_y = V_INF * np.sin(self.alpha)
        self.Vinf_n = self.Vinf_x * self.nx + self.Vinf_y * self.ny
        self.Vinf_t = self.Vinf_x * self.tx + self.Vinf_y * self.ty

        # Build influence matrices once
        self._build_influence_matrices()

        # Build and factorize system matrix A once
        self._build_and_factorize_system_matrix()

        # Unknowns
        self.q = np.zeros(self.N)
        self.gamma = 0.0

    def _build_influence_matrices(self):
        N = self.N
        self.Is_n = np.zeros((N, N))
        self.Iv_n = np.zeros((N, N))
        self.Is_t = np.zeros((N, N))
        self.Iv_t = np.zeros((N, N))

        # Nested loops: OK for N=320; major speed win comes from LU caching in solve().
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

                # Local transform relative to panel j
                x_local = dx * self.tx[j] + dy * self.ty[j]
                y_local = -dx * self.ty[j] + dy * self.tx[j]

                r1_sq = x_local**2 + y_local**2
                r2_sq = (x_local - self.L[j])**2 + y_local**2

                theta1 = np.arctan2(y_local, x_local)
                theta2 = np.arctan2(y_local, x_local - self.L[j])
                dtheta = theta2 - theta1

                # unwrap
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi

                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc = 1.0 / np.pi * dtheta

                # Vortex influence in local coords
                uv_loc, vv_loc = -vs_loc, us_loc

                # Rotate to global (panel-j basis -> global)
                us_glob = us_loc * self.tx[j] - vs_loc * self.ty[j]
                vs_glob = us_loc * self.ty[j] + vs_loc * self.tx[j]
                uv_glob = uv_loc * self.tx[j] - vv_loc * self.ty[j]
                vv_glob = uv_loc * self.ty[j] + vv_loc * self.tx[j]

                # Project onto panel-i normal/tangent
                self.Is_n[i, j] = us_glob * self.nx[i] + vs_glob * self.ny[i]
                self.Is_t[i, j] = us_glob * self.tx[i] + vs_glob * self.ty[i]
                self.Iv_n[i, j] = uv_glob * self.nx[i] + vv_glob * self.ny[i]
                self.Iv_t[i, j] = uv_glob * self.tx[i] + vv_glob * self.ty[i]

    def _build_and_factorize_system_matrix(self):
        """
        A is constant across coupling iterations.
        Only b changes due to leakage velocities.
        """
        N = self.N
        A = np.zeros((N + 1, N + 1))

        # Flow tangency (normal)
        A[:N, :N] = self.Is_n
        A[:N, N] = np.sum(self.Iv_n, axis=1)

        # Kutta condition row
        A[N, :N] = self.Is_t[0, :] + self.Is_t[N - 1, :]
        A[N, N] = np.sum(self.Iv_t[0, :] + self.Iv_t[N - 1, :])

        self.A = A
        self.lu, self.piv = scipy.linalg.lu_factor(A)

    def solve(self, V_leakage: np.ndarray):
        """
        Solve panel method for given leakage normal velocity on panels.
        """
        N = self.N
        V_leakage = np.asarray(V_leakage)
        if V_leakage.shape[0] != N:
            raise ValueError(f"V_leakage must have length {N}")

        # RHS only (A is constant)
        b = np.zeros(N + 1)
        b[:N] = V_leakage - self.Vinf_n
        b[N] = -(self.Vinf_t[0] + self.Vinf_t[N - 1])

        x = scipy.linalg.lu_solve((self.lu, self.piv), b)

        self.q = x[:N]
        self.gamma = x[N]

        # Tangential velocity and Cp
        Vt = self.Vinf_t + self.Is_t @ self.q + self.gamma * np.sum(self.Iv_t, axis=1)
        Cp = 1.0 - (Vt / V_INF) ** 2
        return Cp

    def compute_velocity_field(self, X_grid, Y_grid):
        """
        External velocity field on a grid (same logic as your original).
        """
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
# 4. STRUCTURED POROUS "CROSS-FLOW PLENUM" NETWORK (same concept, cleaner)
# ==============================================================================
def generate_tangential_mesh(
    xc, yc, tx, ty, cp_solid,
    n_target, pore_radius, mu
):
    """
    Generates a cross-flow plenum network:
      - A central "plenum" internal node.
      - Multiple inlet boundary nodes on bottom forward section.
      - Multiple outlet boundary nodes on top aft section.

    Optimizations / fixes vs your original:
      - Uses G.has_node() checks.
      - Stores edge 'radius' so internal velocities can be computed consistently.
      - Keeps selection logic same, so "results similar".

    NOTE: n_target and pore_radius are not directly used here (plenum model sets
    separate inlet/outlet radii), but we keep signature consistent with your call.
    """
    G = nx.Graph()

    # Plenum node id (stable)
    plenum_id = 99999
    plenum_pos = np.array([0.5, 0.0])
    G.add_node(plenum_id, pos=plenum_pos, type="internal")

    # --- Configuration ---
    N_INLETS = 40
    R_INLET = 3000e-6

    N_OUTLETS = 15
    R_OUTLET = 4000e-6

    # Outlet candidates: top surface and aft
    outlet_candidates = [i for i in range(len(xc)) if (yc[i] > 0.0 and xc[i] >= 0.85)]
    # Inlet candidates: bottom surface and forward
    inlet_candidates = [i for i in range(len(xc)) if (yc[i] < 0.0 and 0.02 <= xc[i] <= 0.20)]

    # Select inlets by highest Cp (highest pressure)
    inlet_scores = [{"id": i, "cp": cp_solid[i]} for i in inlet_candidates]
    inlet_scores.sort(key=lambda d: d["cp"], reverse=True)
    selected_inlets = [d["id"] for d in inlet_scores[:N_INLETS]]

    # Select outlets by max x (closest to trailing edge)
    outlet_scores = [{"id": i, "x": xc[i]} for i in outlet_candidates]
    outlet_scores.sort(key=lambda d: d["x"], reverse=True)
    selected_outlets = [d["id"] for d in outlet_scores[:N_OUTLETS]]

    # Connect inlets to plenum
    for u in selected_inlets:
        node_pos = np.array([xc[u], yc[u]])
        length = np.linalg.norm(node_pos - plenum_pos) + 1e-15
        cond = (np.pi * R_INLET**4) / (8.0 * mu * length)

        if not G.has_node(u):
            G.add_node(u, pos=(xc[u], yc[u]), type="boundary", panel_idx=u)

        G.add_edge(u, plenum_id, length=length, cond=cond, radius=R_INLET, type="plenum_in")

    # Connect plenum to outlets
    for v in selected_outlets:
        node_pos = np.array([xc[v], yc[v]])
        length = np.linalg.norm(node_pos - plenum_pos) + 1e-15
        cond = (np.pi * R_OUTLET**4) / (8.0 * mu * length)

        if not G.has_node(v):
            G.add_node(v, pos=(xc[v], yc[v]), type="boundary", panel_idx=v)

        G.add_edge(plenum_id, v, length=length, cond=cond, radius=R_OUTLET, type="plenum_out")

    print(f"   -> Cross-Flow Plenum: {len(selected_inlets)} Inlets -> {len(selected_outlets)} Outlets (TE).")
    porous_pids = selected_inlets + selected_outlets
    return G, porous_pids


# ==============================================================================
# 5. INTERNAL FLOW SOLVER - OPTIMIZED (build once, solve many)
# ==============================================================================
class InternalFlowSolver:
    """
    Solves node pressures in a resistive network with mixed boundary/internal nodes.

    Model:
      For each edge (i,j) with conductance c:
          Q_ij = c * (P_i - P_j)

      Internal node mass conservation:
          sum_j c_ij (P_i - P_j) = 0

      Boundary nodes:
          P_boundary(node) is prescribed each iteration from external pressure at
          its panel index.

    Optimization:
      - Fix node ordering ONCE.
      - Build sparse system for internal unknown pressures once.
      - Factorize once with splu().
      - Each iteration:
          - fill boundary pressures
          - build RHS
          - solve internal pressures quickly

    Returns:
      - leakage velocities at boundary panels (dict pid -> v_leak)
      - P_all array for all nodes in fixed ordering
    """

    def __init__(self, G: nx.Graph, mu: float, surface_pore_radius: float):
        self.G = G
        self.mu = mu
        self.surface_pore_radius = surface_pore_radius

        # Fixed node ordering (critical for correctness and for plotting)
        self.nodes = list(G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.nodes)}

        # Identify boundary/internal nodes
        self.boundary_nodes = [n for n in self.nodes if G.nodes[n].get("type") == "boundary"]
        self.internal_nodes = [n for n in self.nodes if G.nodes[n].get("type") != "boundary"]

        # Mapping internal node -> unknown index
        self.int_index = {n: k for k, n in enumerate(self.internal_nodes)}
        self.n_int = len(self.internal_nodes)

        # Prebuild sparse matrix A_int and factorize
        self._build_factorized_internal_matrix()

    def _build_factorized_internal_matrix(self):
        """
        Build A_int for internal unknown pressures:
            A_int * P_int = rhs(boundary pressures)
        """
        rows = []
        cols = []
        data = []

        # For RHS contribution from boundary neighbors
        # rhs[k] accumulates sum_j c_ij * P_boundary(j) for boundary neighbors j
        # We'll compute rhs each iteration (because boundary pressures change),
        # but A_int stays constant.

        for n in self.internal_nodes:
            k = self.int_index[n]
            diag = 0.0

            for nbr in self.G.neighbors(n):
                c = self.G[n][nbr]["cond"]
                diag += c

                if nbr in self.int_index:
                    # internal neighbor => off-diagonal -c
                    kk = self.int_index[nbr]
                    rows.append(k)
                    cols.append(kk)
                    data.append(-c)
                else:
                    # boundary neighbor => handled in RHS each solve
                    pass

            # diagonal
            rows.append(k)
            cols.append(k)
            data.append(diag)

        A_int = sp.csc_matrix((data, (rows, cols)), shape=(self.n_int, self.n_int))
        self.A_int = A_int
        self.lu_int = spla.splu(A_int)

    def solve(self, P_boundary_by_panel: dict):
        """
        P_boundary_by_panel: dict {panel_idx: pressure_value} for porous panels

        Boundary node pressures:
            P(node) = P_boundary_by_panel[panel_idx] if provided else 0.0

        Returns:
            velocities_by_panel: dict {panel_idx: leakage_velocity}
            P_all: pressures for all nodes in fixed ordering self.nodes
        """
        # Build boundary pressure lookup for boundary nodes
        P_bnode = {}
        for bn in self.boundary_nodes:
            pid = self.G.nodes[bn].get("panel_idx", None)
            P_bnode[bn] = float(P_boundary_by_panel.get(pid, 0.0))

        # RHS for internal system
        rhs = np.zeros(self.n_int)
        for n in self.internal_nodes:
            k = self.int_index[n]
            s = 0.0
            for nbr in self.G.neighbors(n):
                if nbr in self.int_index:
                    continue
                # boundary neighbor
                c = self.G[n][nbr]["cond"]
                s += c * P_bnode.get(nbr, 0.0)
            rhs[k] = s

        # Solve internal pressures
        if self.n_int > 0:
            P_int = self.lu_int.solve(rhs)
        else:
            P_int = np.array([])

        # Assemble full pressure vector in fixed node ordering
        P_all = np.zeros(len(self.nodes))
        for n in self.internal_nodes:
            P_all[self.node_index[n]] = P_int[self.int_index[n]]
        for n in self.boundary_nodes:
            P_all[self.node_index[n]] = P_bnode[n]

        # Compute leakage velocity for each boundary node (net flow / surface pore area)
        velocities_by_panel = {}
        A_surface = np.pi * self.surface_pore_radius**2

        for bn in self.boundary_nodes:
            pid = self.G.nodes[bn]["panel_idx"]
            i = self.node_index[bn]
            P_i = P_all[i]

            Q_net = 0.0
            for nbr in self.G.neighbors(bn):
                c = self.G[bn][nbr]["cond"]
                j = self.node_index[nbr]
                Q_net += c * (P_i - P_all[j])

            # Same sign convention as your original: velocities[pid] = -Q_net / area
            velocities_by_panel[pid] = -Q_net / (A_surface + 1e-30)

        return velocities_by_panel, P_all


# ==============================================================================
# 6. RESULTS / PLOTTING
# ==============================================================================
def plot_results(
    aero: PanelMethod,
    Cp: np.ndarray,
    Cp_solid: np.ndarray,
    V_leakage: np.ndarray,
    CL: float,
    CL_solid: float,
    CD: float,
    CD_solid: float,
    G: nx.Graph,
    P_nodes: np.ndarray,
    node_order: list,
    output_dir: str = "porous_airfoil_results",
):
    """
    Generates plots and saves CSV similar to your original code, but with
    corrected internal velocity computation using edge Q/A and stable node ordering.
    """
    # Determine output path
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    full_output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    print(f"-> Saving results to: {full_output_dir}")

    # --- SAVE CSV ---
    csv_path = os.path.join(full_output_dir, "simulation_data.csv")
    try:
        with open(csv_path, "w") as f:
            f.write("--- GLOBAL RESULTS ---\n")
            f.write("Metric,Solid_Baseline,Porous_Result,Change_Percent\n")

            cl_change = ((CL - CL_solid) / (abs(CL_solid) + 1e-9)) * 100.0
            cd_change = ((CD - CD_solid) / (abs(CD_solid) + 1e-9)) * 100.0
            f.write(f"CL,{CL_solid:.6f},{CL:.6f},{cl_change:.2f}%\n")
            f.write(f"CD,{CD_solid:.6f},{CD:.6f},{cd_change:.2f}%\n\n")

            f.write("--- PANEL DISTRIBUTION DATA ---\n")
            f.write("Panel_ID,XC,YC,Cp_Solid,Cp_Porous,V_leakage\n")
            for i in range(aero.N):
                f.write(
                    f"{i},{aero.XC[i]:.6f},{aero.YC[i]:.6f},"
                    f"{Cp_solid[i]:.6f},{Cp[i]:.6f},{V_leakage[i]:.6f}\n"
                )
        print("-> CSV Data saved successfully.")
    except Exception as e:
        print(f"ERROR saving CSV: {e}")

    pos = nx.get_node_attributes(G, "pos")

    # --- FIGURE 1: Geometry & Cp ---
    print("-> Generating Figure 1...")
    fig1 = plt.figure(figsize=(12, 12))
    gs1 = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])

    ax1 = fig1.add_subplot(gs1[0])
    ax1.plot(aero.X, aero.Y, linewidth=2, color="black", label="Airfoil")
    ax1.fill(aero.X, aero.Y, "whitesmoke")

    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="cyan", alpha=0.6, width=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), ax=ax1, node_size=30, node_color="black")
    ax1.set_title("1. Geometry & Porous Network", fontsize=14)
    ax1.axis("equal")

    ax2 = fig1.add_subplot(gs1[1])
    ax2.plot(aero.XC, Cp_solid, "k--", label=f"Solid ($C_L = {CL_solid:.3f}$)")
    ax2.plot(aero.XC, Cp, "b-", label=f"Porous ($C_L = {CL:.3f}$)")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("2. Pressure Coefficient ($C_p$)", fontsize=14)

    fig1.savefig(os.path.join(full_output_dir, "01_Geometry_and_Cp.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # --- FIGURE 3: Surface Vectors ---
    print("-> Generating Figure 3...")
    fig3 = plt.figure(figsize=(10, 6))
    ax5 = fig3.add_subplot(111)
    ax5.plot(aero.X, aero.Y, linewidth=2, color="black")
    ax5.fill(aero.X, aero.Y, "whitesmoke")

    # Visualize pressure as vectors along normals (same idea as your code)
    U_p = -Cp * aero.nx * 0.15
    V_p = -Cp * aero.ny * 0.15
    ax5.quiver(
        aero.XC, aero.YC, U_p, V_p, Cp,
        cmap="coolwarm_r", scale=1, scale_units="xy", width=0.004
    )
    ax5.set_title("3. Surface Pressure & Leakage", fontsize=14)
    ax5.axis("equal")

    fig3.savefig(os.path.join(full_output_dir, "03_Pressure_Vectors.png"), dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # --- FIGURE 5: External Flow Field ---
    print("-> Generating Figure 5...")
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.6, 0.6
    Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, FLOWFIELD_RES),
                         np.linspace(y_min, y_max, FLOWFIELD_RES))
    Ug, Vg = aero.compute_velocity_field(Xg, Yg)
    vel_mag = np.sqrt(Ug**2 + Vg**2)

    fig5 = plt.figure(figsize=(12, 7))
    ax9 = fig5.add_subplot(111)
    ax9.contourf(Xg, Yg, vel_mag, levels=50, cmap="viridis", extend="both", alpha=0.9)

    if np.max(vel_mag) > 1e-6:
        seed_points = np.column_stack((np.ones(40) * x_min, np.linspace(y_min, y_max, 40)))
        ax9.streamplot(Xg, Yg, Ug, Vg, color="white", linewidth=0.8, density=2, start_points=seed_points)

    ax9.fill(aero.X, aero.Y, color="black", zorder=3)
    ax9.set_title("5. External Flow Field", fontsize=14)
    ax9.axis("equal")

    fig5.savefig(os.path.join(full_output_dir, "05_Flow_Field.png"), dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # --- FIGURE 6: INTERNAL FLOW CONTOUR ---
    print("-> Generating Figure 6 (Internal Flow Contour)...")

    # Stable node map using the *same ordering* as internal solver returned
    node_map = {node: i for i, node in enumerate(node_order)}

    points = []
    values = []
    max_v_found = 0.0

    # Sample along each pipe and compute average pipe velocity = |Q|/A_pipe
    for u, v, data in G.edges(data=True):
        idx_u = node_map[u]
        idx_v = node_map[v]

        Pu = P_nodes[idx_u]
        Pv = P_nodes[idx_v]
        deltaP = Pu - Pv

        cond = data["cond"]
        Q = cond * deltaP

        R_edge = data.get("radius", PORE_RADIUS)  # fallback
        A_pipe = np.pi * (R_edge**2)
        v_mag = abs(Q) / (A_pipe + 1e-30)

        if v_mag > max_v_found:
            max_v_found = v_mag

        pos_u = np.array(pos[u])
        pos_v = np.array(pos[v])

        # Interpolate points along the edge
        for t in np.linspace(0, 1, 5):
            pt = pos_u + t * (pos_v - pos_u)
            points.append(pt)
            values.append(v_mag)

    points = np.array(points) if len(points) else np.zeros((0, 2))
    values = np.array(values) if len(values) else np.zeros((0,))

    fig6 = plt.figure(figsize=(14, 8))
    ax_int = fig6.add_subplot(111)

    # Airfoil outline
    ax_int.plot(aero.X, aero.Y, linewidth=3, color="#333333", zorder=10)

    if len(points) > 0:
        min_x, max_x = float(np.min(aero.X)), float(np.max(aero.X))
        min_y, max_y = float(np.min(aero.Y)), float(np.max(aero.Y))
        grid_x, grid_y = np.mgrid[min_x:max_x:complex(INTERNAL_GRID_RES),
                                 min_y:max_y:complex(INTERNAL_GRID_RES)]

        grid_z = griddata(points, values, (grid_x, grid_y), method="linear", fill_value=0.0)

        # Mask outside airfoil
        airfoil_poly = np.column_stack((aero.X, aero.Y))
        path = mpath.Path(airfoil_poly)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        mask = path.contains_points(grid_points).reshape(grid_x.shape)
        grid_z[~mask] = np.nan

        contour = ax_int.contourf(grid_x, grid_y, grid_z, levels=100, cmap="plasma", zorder=1)
        cbar = plt.colorbar(contour, ax=ax_int, pad=0.02)
        cbar.set_label("Internal Pipe Velocity [m/s]", fontsize=12)

    # Overlay network edges faintly
    nx.draw_networkx_edges(G, pos, ax=ax_int, edge_color="white", alpha=0.3, width=0.5)

    ax_int.set_title(f"6. Internal Flow Distribution\nMax Velocity: {max_v_found:.2f} m/s", fontsize=16)
    ax_int.set_xlabel("x/c")
    ax_int.set_ylabel("y/c")
    ax_int.axis("equal")
    ax_int.set_xlim(0.0, 1.05)

    fig6.savefig(os.path.join(full_output_dir, "06_Internal_Flow_Map_Contour.png"), dpi=300, bbox_inches="tight")
    plt.close(fig6)

    print(f"-> All figures saved to: {full_output_dir}")


# ==============================================================================
# 7. MAIN SIMULATION LOOP (Optimized Coupling)
# ==============================================================================
def run_simulation():
    print(f"--- SIMULATION START: NACA {AIRFOIL_NAME} @ {ANGLE_OF_ATTACK} deg ---")
    print(f"--- Re={REYNOLDS_NUM}, V_inf={V_INF:.2f} m/s ---")

    # Build geometry and panel method
    X, Y = naca4(AIRFOIL_NAME, n_panels=N_PANELS)
    aero = PanelMethod(X, Y, ANGLE_OF_ATTACK)

    # Baseline solid
    print("-> Solving Baseline (Solid)...")
    V0 = np.zeros(aero.N)
    Cp_solid = aero.solve(V0)

    # Solid forces
    fx_elem = -Cp_solid * aero.nx * aero.L
    fy_elem = -Cp_solid * aero.ny * aero.L
    Fx_solid = float(np.sum(fx_elem))
    Fy_solid = float(np.sum(fy_elem))
    CL_solid = Fy_solid * np.cos(aero.alpha) - Fx_solid * np.sin(aero.alpha)
    CD_solid = Fx_solid * np.cos(aero.alpha) + Fy_solid * np.sin(aero.alpha)
    print(f"   Baseline CL: {CL_solid:.4f}")

    # Build porous network
    print("-> Generating Asymmetric Plenum Mesh...")
    G, porous_pids = generate_tangential_mesh(
        aero.XC, aero.YC, aero.tx, aero.ty, Cp_solid,
        50, PORE_RADIUS, MU
    )
    print(f"   Generated network with {len(G.nodes())} nodes.")

    # Internal solver setup ONCE (optimized)
    internal_solver = InternalFlowSolver(G, mu=MU, surface_pore_radius=PORE_RADIUS)

    # Coupling iteration variables
    V_leakage = np.zeros(aero.N)
    q_inf = 0.5 * RHO * (V_INF**2)

    final_P_nodes = np.zeros(len(internal_solver.nodes))
    Cp = Cp_solid.copy()

    # Iteration loop
    for it in range(MAX_ITER):
        # 1) Aero solve with current leakage (fast LU backsolve)
        Cp = aero.solve(V_leakage)

        # 2) External pressure at porous panels
        P_ext = P_INF + q_inf * Cp
        P_map = {pid: float(P_ext[pid]) for pid in porous_pids}

        # 3) Internal solve (fast sparse LU)
        V_calculated, P_nodes_iter = internal_solver.solve(P_map)
        final_P_nodes = P_nodes_iter

        # 4) Relax update only at porous panels
        max_diff = 0.0
        for pid, v_calc in V_calculated.items():
            v_old = V_leakage[pid]
            v_new = RELAXATION * v_calc + (1.0 - RELAXATION) * v_old

            # Keep your clipping
            v_new = max(min(v_new, 80.0), -80.0)

            diff = abs(v_new - v_old)
            if diff > max_diff:
                max_diff = diff

            V_leakage[pid] = v_new

        # convergence
        if max_diff < CONVERGENCE_TOL and it > 5:
            print(f"-> Converged at Iter {it}")
            break

        if it % 10 == 0:
            print(f"   Iter {it}: Max Resid = {max_diff:.6e}")

    # Porous forces
    fx_elem = -Cp * aero.nx * aero.L
    fy_elem = -Cp * aero.ny * aero.L
    Fx_porous = float(np.sum(fx_elem))
    Fy_porous = float(np.sum(fy_elem))

    CL = Fy_porous * np.cos(aero.alpha) - Fx_porous * np.sin(aero.alpha)
    CD = Fx_porous * np.cos(aero.alpha) + Fy_porous * np.sin(aero.alpha)

    print(f"-> Final Results: Solid CL={CL_solid:.4f}, Porous CL={CL:.4f}")

    plot_results(
        aero=aero,
        Cp=Cp,
        Cp_solid=Cp_solid,
        V_leakage=V_leakage,
        CL=CL,
        CL_solid=CL_solid,
        CD=CD,
        CD_solid=CD_solid,
        G=G,
        P_nodes=final_P_nodes,
        node_order=internal_solver.nodes,  # IMPORTANT: consistent ordering
    )


if __name__ == "__main__":
    run_simulation()
