import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg as la
import numba as nb

from input import Config


@nb.jit(nopython=True, fastmath=True, parallel=True)
def _fast_velocity_field(X_grid, Y_grid, X_panel, Y_panel, tx, ty, L, q, gamma, Vinf_x, Vinf_y):
    """Highly optimized JIT-compiled loop for computing external velocity fields."""
    rows, cols = X_grid.shape
    N = len(q)

    u = np.full((rows, cols), Vinf_x)
    v = np.full((rows, cols), Vinf_y)

    for i in nb.prange(rows):
        for j in range(cols):
            px = X_grid[i, j]
            py = Y_grid[i, j]

            u_pt = 0.0
            v_pt = 0.0

            for k in range(N):
                dx = px - X_panel[k]
                dy = py - Y_panel[k]

                x_loc = dx * tx[k] + dy * ty[k]
                y_loc = -dx * ty[k] + dy * tx[k]

                r1_sq = x_loc**2 + y_loc**2
                r2_sq = (x_loc - L[k])**2 + y_loc**2

                theta1 = np.arctan2(y_loc, x_loc)
                theta2 = np.arctan2(y_loc, x_loc - L[k])

                dtheta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi

                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc = 1.0 / np.pi * dtheta

                u_ind = (us_loc * q[k] - vs_loc * gamma) * tx[k] - \
                        (vs_loc * q[k] + us_loc * gamma) * ty[k]
                v_ind = (us_loc * q[k] - vs_loc * gamma) * ty[k] + \
                        (vs_loc * q[k] + us_loc * gamma) * tx[k]

                u_pt += u_ind
                v_pt += v_ind

            u[i, j] += u_pt
            v[i, j] += v_pt

    return u, v


# ==============================================================================
# 3. AERODYNAMIC SOLVER
# ==============================================================================
class PanelMethod:
    def __init__(self, X, Y, config: Config):
        self.X, self.Y = X, Y
        self.cfg = config
        self.alpha = np.radians(config.ANGLE_OF_ATTACK)
        self.N = len(X) - 1

        # Geometry
        self.XC = (X[:-1] + X[1:]) / 2
        self.YC = (Y[:-1] + Y[1:]) / 2
        self.dx = X[1:] - X[:-1]
        self.dy = Y[1:] - Y[:-1]
        self.L = np.sqrt(self.dx**2 + self.dy**2)
        self.nx, self.ny = self.dy / self.L, -self.dx / self.L
        self.tx, self.ty = self.dx / self.L, self.dy / self.L

        self._build_influence_matrices()
        self._prepare_linear_system()

        self.q = np.zeros(self.N)
        self.gamma = 0.0

    def _prepare_linear_system(self):
        """Builds the aerodynamic influence matrix A once and precomputes its LU factorization."""
        A = np.zeros((self.N + 1, self.N + 1))

        A[:self.N, :self.N] = self.Is_n
        A[:self.N, self.N] = np.sum(self.Iv_n, axis=1)
        A[self.N, :self.N] = self.Is_t[0, :] + self.Is_t[self.N - 1, :]
        A[self.N, self.N] = np.sum(self.Iv_t[0, :] + self.Iv_t[self.N - 1, :])

        self.lu_A, self.piv_A = la.lu_factor(A)

    def solve(self, V_leakage=None):
        if V_leakage is None:
            V_leakage = np.zeros(self.N)

        Vinf_x = self.cfg.V_INF * np.cos(self.alpha)
        Vinf_y = self.cfg.V_INF * np.sin(self.alpha)
        Vinf_n = Vinf_x * self.nx + Vinf_y * self.ny
        Vinf_t = Vinf_x * self.tx + Vinf_y * self.ty

        b = np.zeros(self.N + 1)
        b[:self.N] = V_leakage - Vinf_n
        b[self.N] = -(Vinf_t[0] + Vinf_t[self.N - 1])

        try:
            x = la.lu_solve((self.lu_A, self.piv_A), b)
        except Exception:
            return np.zeros(self.N)

        self.q = x[:self.N]
        self.gamma = x[self.N]

        Vt = Vinf_t + np.dot(self.Is_t, self.q) + self.gamma * np.sum(self.Iv_t, axis=1)
        Cp = 1.0 - (Vt / self.cfg.V_INF) ** 2
        return Cp

    def _build_influence_matrices(self):
        """Vectorized computation of aerodynamic influence matrices without Python for-loops."""
        XC_i = self.XC[:, np.newaxis]
        YC_i = self.YC[:, np.newaxis]
        nx_i = self.nx[:, np.newaxis]
        ny_i = self.ny[:, np.newaxis]
        tx_i = self.tx[:, np.newaxis]
        ty_i = self.ty[:, np.newaxis]

        X_j = self.X[:-1][np.newaxis, :]
        Y_j = self.Y[:-1][np.newaxis, :]
        tx_j = self.tx[np.newaxis, :]
        ty_j = self.ty[np.newaxis, :]
        L_j = self.L[np.newaxis, :]

        dx = XC_i - X_j
        dy = YC_i - Y_j

        x_loc = dx * tx_j + dy * ty_j
        y_loc = -dx * ty_j + dy * tx_j

        r1_sq = x_loc**2 + y_loc**2
        r2_sq = (x_loc - L_j) ** 2 + y_loc**2

        theta1 = np.arctan2(y_loc, x_loc)
        theta2 = np.arctan2(y_loc, x_loc - L_j)

        dtheta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi

        us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
        vs_loc = 1.0 / np.pi * dtheta

        us_glob = us_loc * tx_j - vs_loc * ty_j
        vs_glob = us_loc * ty_j + vs_loc * tx_j
        uv_glob = -vs_loc * tx_j - us_loc * ty_j
        vv_glob = -vs_loc * ty_j + us_loc * tx_j

        self.Is_n = us_glob * nx_i + vs_glob * ny_i
        self.Is_t = us_glob * tx_i + vs_glob * ty_i
        self.Iv_n = uv_glob * nx_i + vv_glob * ny_i
        self.Iv_t = uv_glob * tx_i + vv_glob * ty_i

        np.fill_diagonal(self.Is_n, 0.5 * np.pi)
        np.fill_diagonal(self.Iv_t, 0.5 * np.pi)
        np.fill_diagonal(self.Is_t, 0.0)
        np.fill_diagonal(self.Iv_n, 0.0)

    def compute_velocity_field(self, X_grid, Y_grid):
        Vinf_x = self.cfg.V_INF * np.cos(self.alpha)
        Vinf_y = self.cfg.V_INF * np.sin(self.alpha)

        return _fast_velocity_field(
            X_grid, Y_grid,
            self.X, self.Y,
            self.tx, self.ty,
            self.L, self.q, self.gamma,
            Vinf_x, Vinf_y
        )


# ==============================================================================
# 4. POROUS NETWORK SYSTEM
# ==============================================================================
class PorousNetwork:
    def __init__(self, aero: PanelMethod, cp_solid, config: Config):
        self.aero = aero
        self.cfg = config
        self.G = nx.Graph()
        self.active_pores = []

        self._build_network(cp_solid)

    def _build_network(self, cp_solid):
        xc, yc = self.aero.XC, self.aero.YC
        self.G = nx.Graph()

        # 1. CENTRAL SPAR (The Plenum)
        spar_id = 99999
        spar_pos = np.array([0.25, 0.0])
        self.G.add_node(spar_id, pos=spar_pos, type='internal')

        # 2. SELECT ZONES (Where to drill the holes)
        inlet_candidates = [i for i in range(len(xc)) if yc[i] > 0 and xc[i] >= 0.85]
        outlet_candidates = [i for i in range(len(xc)) if yc[i] < 0 and 0.05 <= xc[i] <= 0.20]

        inlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in inlet_candidates]
        inlet_scores.sort(key=lambda x: x['cp'], reverse=True)
        selected_inlets = [x['id'] for x in inlet_scores[:self.cfg.N_INLETS]]

        outlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in outlet_candidates]
        outlet_scores.sort(key=lambda x: x['cp'])
        num_outlets_to_use = min(self.cfg.N_OUTLETS, len(outlet_scores))
        selected_outlets = [x['id'] for x in outlet_scores[:num_outlets_to_use]]

        self.active_pores = selected_inlets + selected_outlets

        # 3. CONNECT WITH PURE GEOMETRY
        print(f"   -> Generating Passive Network: {len(selected_inlets)} Pores Top-TE -> {len(selected_outlets)} Pores Bottom-LE.")

        for pid in self.active_pores:
            p_pos = np.array([xc[pid], yc[pid]])
            dist = np.linalg.norm(p_pos - spar_pos)

            if pid not in self.G:
                self.G.add_node(pid, pos=(xc[pid], yc[pid]), type='boundary', panel_idx=pid)

            if pid in selected_inlets:
                r = self.cfg.PORE_RADIUS_INLET
                etype = 'plenum_in'
            else:
                r = self.cfg.PORE_RADIUS_OUTLET
                etype = 'plenum_out'

            cond = (np.pi * r**4) / (8 * self.cfg.MU * dist)
            self.G.add_edge(pid, spar_id, length=dist, cond=cond, type=etype)

        self._prepare_network_solver()

    def _prepare_network_solver(self):
        """Assembles the sparse conductance matrix once and precomputes the factorization."""
        self.nodes = list(self.G.nodes())
        self.n_nodes = len(self.nodes)
        self.node_map = {node: i for i, node in enumerate(self.nodes)}
        self.boundary_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['type'] == 'boundary']

        A = sp.lil_matrix((self.n_nodes, self.n_nodes))

        for node in self.nodes:
            idx = self.node_map[node]
            if node in self.boundary_nodes:
                A[idx, idx] = 1.0
            else:
                sigma_cond = 0.0
                for nbr in self.G.neighbors(node):
                    c = self.G[node][nbr]['cond']
                    nbr_idx = self.node_map[nbr]
                    A[idx, nbr_idx] = -c
                    sigma_cond += c
                A[idx, idx] = sigma_cond if sigma_cond > 0 else 1.0

        self.net_solver = scipy.sparse.linalg.factorized(A.tocsc())

    def solve_flow(self, P_boundary):
        """Solves the network using the pre-factorized matrix."""
        b = np.zeros(self.n_nodes)

        for node in self.boundary_nodes:
            idx = self.node_map[node]
            pid = self.G.nodes[node]['panel_idx']
            b[idx] = P_boundary.get(pid, 0.0)

        try:
            P_nodes = self.net_solver(b)
        except Exception as e:
            print(f"   [!] Matrix solve failed: {e}")
            return {}, np.zeros(self.n_nodes)

        velocities = {}
        for node in self.boundary_nodes:
            pid = self.G.nodes[node]['panel_idx']
            idx = self.node_map[node]

            is_inlet = any(self.G[node][nbr]['type'] == 'plenum_in' for nbr in self.G.neighbors(node))
            radius = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * radius**2

            Q_net = 0.0
            for nbr in self.G.neighbors(node):
                c = self.G[node][nbr]['cond']
                nbr_idx = self.node_map[nbr]
                Q_net += c * (P_nodes[idx] - P_nodes[nbr_idx])

            velocities[pid] = (-Q_net / area) if area > 0 else 0.0

        return velocities, P_nodes