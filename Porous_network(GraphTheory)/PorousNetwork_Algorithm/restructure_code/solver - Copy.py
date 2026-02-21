# solver.py
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg
from input import Config

# ------------------------------------------------------------
# Optional Numba acceleration (hard fallback if not installed)
# ------------------------------------------------------------
_NUMBA_OK = False
try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False


# ============================================================
# NUMBA KERNELS (CPU ONLY)
# ============================================================
if _NUMBA_OK:

    @njit(cache=True, fastmath=True)
    def _wrap_dtheta(dtheta: float) -> float:
        if dtheta > np.pi:
            dtheta -= 2.0 * np.pi
        elif dtheta < -np.pi:
            dtheta += 2.0 * np.pi
        return dtheta

    @njit(cache=True, fastmath=True)
    def build_influence_matrices_numba(X, Y, XC, YC, L, nx_arr, ny_arr, tx, ty):
        """
        Build influence matrices for constant-strength source panels and a single
        constant circulation gamma (implemented via summed vortex influences).
        Returns:
            Is_n, Iv_n, Is_t, Iv_t  each (N,N)
        """
        N = XC.shape[0]
        Is_n = np.zeros((N, N), dtype=np.float64)
        Iv_n = np.zeros((N, N), dtype=np.float64)
        Is_t = np.zeros((N, N), dtype=np.float64)
        Iv_t = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                if i == j:
                    # Self influence for source normal and vortex tangential
                    Is_n[i, j] = 0.5 * np.pi
                    Iv_t[i, j] = 0.5 * np.pi
                    continue

                dx = XC[i] - X[j]
                dy = YC[i] - Y[j]

                # transform to panel-local coordinates (panel j)
                x_loc = dx * tx[j] + dy * ty[j]
                y_loc = -dx * ty[j] + dy * tx[j]

                r1_sq = x_loc * x_loc + y_loc * y_loc
                xmL = x_loc - L[j]
                r2_sq = xmL * xmL + y_loc * y_loc

                theta1 = np.arctan2(y_loc, x_loc)
                theta2 = np.arctan2(y_loc, xmL)
                dtheta = _wrap_dtheta(theta2 - theta1)

                # Local induced velocity for unit source panel:
                # (us_loc, vs_loc) in panel-local frame
                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc = 1.0 / np.pi * dtheta

                # source -> global
                us_glob = us_loc * tx[j] - vs_loc * ty[j]
                vs_glob = us_loc * ty[j] + vs_loc * tx[j]

                # vortex -> global (90-degree rotation of source influence)
                uv_glob = -vs_loc * tx[j] - us_loc * ty[j]
                vv_glob = -vs_loc * ty[j] + us_loc * tx[j]

                # project onto control-point normal/tangent (panel i)
                Is_n[i, j] = us_glob * nx_arr[i] + vs_glob * ny_arr[i]
                Is_t[i, j] = us_glob * tx[i] + vs_glob * ty[i]
                Iv_n[i, j] = uv_glob * nx_arr[i] + vv_glob * ny_arr[i]
                Iv_t[i, j] = uv_glob * tx[i] + vv_glob * ty[i]

        return Is_n, Iv_n, Is_t, Iv_t

    @njit(cache=True, fastmath=True)
    def compute_velocity_field_numba(
        x_flat, y_flat,
        X, Y, L, tx, ty,
        q, gamma,
        Vinf_x, Vinf_y
    ):
        """
        Evaluate velocity field at points (x_flat, y_flat) given solved q and gamma.
        CPU-only Numba kernel.
        """
        npts = x_flat.shape[0]
        N = q.shape[0]

        u = np.empty(npts, dtype=np.float64)
        v = np.empty(npts, dtype=np.float64)

        for p in range(npts):
            up = Vinf_x
            vp = Vinf_y
            Xp = x_flat[p]
            Yp = y_flat[p]

            for j in range(N):
                dx = Xp - X[j]
                dy = Yp - Y[j]

                x_loc = dx * tx[j] + dy * ty[j]
                y_loc = -dx * ty[j] + dy * tx[j]

                r1_sq = x_loc * x_loc + y_loc * y_loc
                xmL = x_loc - L[j]
                r2_sq = xmL * xmL + y_loc * y_loc

                theta1 = np.arctan2(y_loc, x_loc)
                theta2 = np.arctan2(y_loc, xmL)
                dtheta = (theta2 - theta1 + np.pi) % (2.0 * np.pi) - np.pi

                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc = 1.0 / np.pi * dtheta

                # induced velocity from source sheet q[j] and global circulation gamma
                # (same algebra as your original)
                u_ind = (us_loc * q[j] - vs_loc * gamma) * tx[j] - (vs_loc * q[j] + us_loc * gamma) * ty[j]
                v_ind = (us_loc * q[j] - vs_loc * gamma) * ty[j] + (vs_loc * q[j] + us_loc * gamma) * tx[j]

                up += u_ind
                vp += v_ind

            u[p] = up
            v[p] = vp

        return u, v

    import os
import numpy as np
import matplotlib.pyplot as plt


def plot_force_distribution(aero, cfg, Cp_solid, Cp_porous=None, out_dir=None, fname="07_Force_Distribution.png"):
    """
    Plot per-panel pressure force distribution along the airfoil.

    Parameters
    ----------
    aero : PanelMethod
        Must have XC, YC, L, nx, ny, alpha (radians)
    cfg : Config
        Must have RHO, V_INF, P_INF (P_INF not used here; Cp is nondim)
    Cp_solid : (N,) array
        Cp distribution for solid baseline
    Cp_porous : (N,) array or None
        Cp distribution for porous case (optional)
    out_dir : str or None
        Where to save figure. If None, uses cfg.OUTPUT_DIR relative to cwd.
    fname : str
        Output filename

    Notes
    -----
    Pressure force per panel (nondimensional Cp):
        pressure difference ~ q_inf * Cp
        force on body due to pressure: dF = - (q_inf * Cp) * n * dS
    where n is outward normal, dS ~ panel length L.
    """
    Cp_solid = np.asarray(Cp_solid, dtype=float)
    if Cp_porous is not None:
        Cp_porous = np.asarray(Cp_porous, dtype=float)

    N = aero.N
    if Cp_solid.shape[0] != N:
        raise ValueError(f"Cp_solid length {Cp_solid.shape[0]} != aero.N {N}")
    if Cp_porous is not None and Cp_porous.shape[0] != N:
        raise ValueError(f"Cp_porous length {Cp_porous.shape[0]} != aero.N {N}")

    # dynamic pressure
    q_inf = 0.5 * cfg.RHO * (cfg.V_INF ** 2)

    # panel arc-length coordinate (midpoint along surface)
    s = np.cumsum(aero.L) - 0.5 * aero.L

    # pressure force components per panel (global axes)
    # dF = -q_inf * Cp * n * dS
    dFx_s = -q_inf * Cp_solid * aero.nx * aero.L
    dFy_s = -q_inf * Cp_solid * aero.ny * aero.L

    # lift/drag contributions per panel (rotate by AoA)
    ca, sa = np.cos(aero.alpha), np.sin(aero.alpha)
    dL_s = dFy_s * ca - dFx_s * sa
    dD_s = dFx_s * ca + dFy_s * sa

    if Cp_porous is not None:
        dFx_p = -q_inf * Cp_porous * aero.nx * aero.L
        dFy_p = -q_inf * Cp_porous * aero.ny * aero.L
        dL_p = dFy_p * ca - dFx_p * sa
        dD_p = dFx_p * ca + dFy_p * sa

    # output dir
    if out_dir is None:
        out_dir = getattr(cfg, "OUTPUT_DIR", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    # -------------------------
    # Plotting
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # (1) dFx(s)
    ax = axes[0, 0]
    ax.plot(s, dFx_s, "k--", lw=1.6, label="Solid")
    if Cp_porous is not None:
        ax.plot(s, dFx_p, "b-", lw=1.6, label="Porous")
    ax.set_title("Panel force distribution: dFx")
    ax.set_xlabel("Surface coordinate s")
    ax.set_ylabel("dFx [N per panel]")
    ax.grid(alpha=0.3)
    ax.legend()

    # (2) dFy(s)
    ax = axes[0, 1]
    ax.plot(s, dFy_s, "k--", lw=1.6, label="Solid")
    if Cp_porous is not None:
        ax.plot(s, dFy_p, "b-", lw=1.6, label="Porous")
    ax.set_title("Panel force distribution: dFy")
    ax.set_xlabel("Surface coordinate s")
    ax.set_ylabel("dFy [N per panel]")
    ax.grid(alpha=0.3)
    ax.legend()

    # (3) dL(s)
    ax = axes[1, 0]
    ax.plot(s, dL_s, "k--", lw=1.6, label="Solid")
    if Cp_porous is not None:
        ax.plot(s, dL_p, "b-", lw=1.6, label="Porous")
    ax.set_title("Panel contribution: dL")
    ax.set_xlabel("Surface coordinate s")
    ax.set_ylabel("dL [N per panel]")
    ax.grid(alpha=0.3)
    ax.legend()

    # (4) dD(s) + cumulative (secondary)
    ax = axes[1, 1]
    ax.plot(s, dD_s, "k--", lw=1.6, label="Solid dD")
    if Cp_porous is not None:
        ax.plot(s, dD_p, "b-", lw=1.6, label="Porous dD")
    ax.set_title("Panel contribution: dD (+ cumulative)")
    ax.set_xlabel("Surface coordinate s")
    ax.set_ylabel("dD [N per panel]")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(s, np.cumsum(dD_s), "k:", lw=1.2, label="Solid cum D")
    if Cp_porous is not None:
        ax2.plot(s, np.cumsum(dD_p), "b:", lw=1.2, label="Porous cum D")
    ax2.set_ylabel("Cumulative D [N]")

    # combined legend for last panel
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle(f"Force distribution (AoA={np.degrees(aero.alpha):.2f} deg, q_inf={q_inf:.1f} Pa)", fontsize=14)
    fig.savefig(out_path, dpi=getattr(cfg, "FIG_DPI", 200), bbox_inches="tight")
    plt.close(fig)

    return out_path


# ============================================================
# Panel Method 
# ============================================================
class PanelMethod:
    def __init__(self, X, Y, config: Config):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.cfg = config
        self.alpha = np.radians(config.ANGLE_OF_ATTACK)

        # Number of panels
        self.N = len(self.X) - 1

        # Geometry
        self.XC = 0.5 * (self.X[:-1] + self.X[1:])
        self.YC = 0.5 * (self.Y[:-1] + self.Y[1:])
        self.dx = self.X[1:] - self.X[:-1]
        self.dy = self.Y[1:] - self.Y[:-1]
        self.L = np.sqrt(self.dx * self.dx + self.dy * self.dy)

        # Normals / tangents
        self.nx = self.dy / self.L
        self.ny = -self.dx / self.L
        self.tx = self.dx / self.L
        self.ty = self.dy / self.L

        # Influence matrices (constant)
        self._build_influence_matrices()

        # Precompute sums for gamma coupling
        self.sum_Iv_n = np.sum(self.Iv_n, axis=1)
        self.sum_Iv_t = np.sum(self.Iv_t, axis=1)

        # Build constant system matrix and factorize once
        self.A_panel = self._build_system_matrix()
        self._lu, self._piv = scipy.linalg.lu_factor(
            self.A_panel, overwrite_a=False, check_finite=False
        )

        # Solution state
        self.q = np.zeros(self.N, dtype=float)
        self.gamma = 0.0

        # Freestream projections (constant)
        self.Vinf_x = self.cfg.V_INF * np.cos(self.alpha)
        self.Vinf_y = self.cfg.V_INF * np.sin(self.alpha)
        self.Vinf_n = self.Vinf_x * self.nx + self.Vinf_y * self.ny
        self.Vinf_t = self.Vinf_x * self.tx + self.Vinf_y * self.ty

    def _build_influence_matrices(self):
        N = self.N
        use_numba = bool(getattr(self.cfg, "USE_NUMBA", False) and _NUMBA_OK)

        if use_numba:
            self.Is_n, self.Iv_n, self.Is_t, self.Iv_t = build_influence_matrices_numba(
                self.X, self.Y, self.XC, self.YC, self.L, self.nx, self.ny, self.tx, self.ty
            )
            return

        # Pure numpy fallback
        self.Is_n = np.zeros((N, N), dtype=float)
        self.Iv_n = np.zeros((N, N), dtype=float)
        self.Is_t = np.zeros((N, N), dtype=float)
        self.Iv_t = np.zeros((N, N), dtype=float)

        for i in range(N):
            for j in range(N):
                if i == j:
                    self.Is_n[i, j] = 0.5 * np.pi
                    self.Iv_t[i, j] = 0.5 * np.pi
                    continue

                dx = self.XC[i] - self.X[j]
                dy = self.YC[i] - self.Y[j]

                x_loc = dx * self.tx[j] + dy * self.ty[j]
                y_loc = -dx * self.ty[j] + dy * self.tx[j]

                r1_sq = x_loc * x_loc + y_loc * y_loc
                xmL = x_loc - self.L[j]
                r2_sq = xmL * xmL + y_loc * y_loc

                theta1 = np.arctan2(y_loc, x_loc)
                theta2 = np.arctan2(y_loc, xmL)
                dtheta = theta2 - theta1
                if dtheta > np.pi:
                    dtheta -= 2.0 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2.0 * np.pi

                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc = 1.0 / np.pi * dtheta

                # source -> global
                us_glob = us_loc * self.tx[j] - vs_loc * self.ty[j]
                vs_glob = us_loc * self.ty[j] + vs_loc * self.tx[j]

                # vortex -> global
                uv_glob = -vs_loc * self.tx[j] - us_loc * self.ty[j]
                vv_glob = -vs_loc * self.ty[j] + us_loc * self.tx[j]

                self.Is_n[i, j] = us_glob * self.nx[i] + vs_glob * self.ny[i]
                self.Is_t[i, j] = us_glob * self.tx[i] + vs_glob * self.ty[i]
                self.Iv_n[i, j] = uv_glob * self.nx[i] + vv_glob * self.ny[i]
                self.Iv_t[i, j] = uv_glob * self.tx[i] + vv_glob * self.ty[i]

    def _build_system_matrix(self) -> np.ndarray:
       
        N = self.N
        A = np.zeros((N + 1, N + 1), dtype=float)

        # no-penetration: Is_n q + (sum Iv_n)*gamma = (V_leak - Vinf_n)
        A[:N, :N] = self.Is_n
        A[:N, N] = self.sum_Iv_n

        # Kutta row: enforce finite TE by tangential condition
        A[N, :N] = self.Is_t[0, :] + self.Is_t[N - 1, :]
        A[N, N] = np.sum(self.Iv_t[0, :] + self.Iv_t[N - 1, :])
        return A

    def solve(self, V_leakage=None) -> np.ndarray:
        """
        Solve for q and gamma given transpiration normal velocity V_leakage (len N).
        Returns Cp at control points.
        """
        if V_leakage is None:
            V_leakage = np.zeros(self.N, dtype=float)
        else:
            V_leakage = np.asarray(V_leakage, dtype=float)
            if V_leakage.shape[0] != self.N:
                raise ValueError(f"V_leakage must have length {self.N}, got {V_leakage.shape[0]}")

        b = np.zeros(self.N + 1, dtype=float)
        b[: self.N] = V_leakage - self.Vinf_n
        b[self.N] = -(self.Vinf_t[0] + self.Vinf_t[self.N - 1])

        x = scipy.linalg.lu_solve((self._lu, self._piv), b, check_finite=False)
        self.q = x[: self.N]
        self.gamma = float(x[self.N])

        Vt = self.Vinf_t + (self.Is_t @ self.q) + self.gamma * self.sum_Iv_t
        Cp = 1.0 - (Vt / (self.cfg.V_INF + 1e-12)) ** 2
        return Cp

    def compute_velocity_field(self, X_grid, Y_grid):
        """
        Velocity field for plotting on a grid.
        Uses Numba CPU kernel if enabled+available; otherwise pure numpy loops.
        """
        Xg = np.asarray(X_grid, dtype=np.float64)
        Yg = np.asarray(Y_grid, dtype=np.float64)

        if getattr(self.cfg, "USE_NUMBA", False) and _NUMBA_OK:
            x_flat = Xg.ravel()
            y_flat = Yg.ravel()
            u_flat, v_flat = compute_velocity_field_numba(
                x_flat, y_flat,
                self.X, self.Y, self.L, self.tx, self.ty,
                self.q, self.gamma,
                float(self.Vinf_x), float(self.Vinf_y),
            )
            return u_flat.reshape(Xg.shape), v_flat.reshape(Yg.shape)

        # Fallback (slow)
        u = np.zeros_like(Xg) + self.Vinf_x
        v = np.zeros_like(Yg) + self.Vinf_y

        for j in range(self.N):
            dx = Xg - self.X[j]
            dy = Yg - self.Y[j]

            x_loc = dx * self.tx[j] + dy * self.ty[j]
            y_loc = -dx * self.ty[j] + dy * self.tx[j]

            r1_sq = x_loc * x_loc + y_loc * y_loc
            xmL = x_loc - self.L[j]
            r2_sq = xmL * xmL + y_loc * y_loc

            theta1 = np.arctan2(y_loc, x_loc)
            theta2 = np.arctan2(y_loc, xmL)
            dtheta = (theta2 - theta1 + np.pi) % (2.0 * np.pi) - np.pi

            us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
            vs_loc = 1.0 / np.pi * dtheta

            u_ind = (us_loc * self.q[j] - vs_loc * self.gamma) * self.tx[j] - (
                vs_loc * self.q[j] + us_loc * self.gamma
            ) * self.ty[j]
            v_ind = (us_loc * self.q[j] - vs_loc * self.gamma) * self.ty[j] + (
                vs_loc * self.q[j] + us_loc * self.gamma
            ) * self.tx[j]

            u += u_ind
            v += v_ind

        return u, v

    def compute_pressure_field(self, X_grid, Y_grid):
        """
        Convenience: compute pressure and Cp field from the velocity field.
        """
        u, v = self.compute_velocity_field(X_grid, Y_grid)
        mag = np.sqrt(u * u + v * v)
        Cp_field = 1.0 - (mag / (self.cfg.V_INF + 1e-12)) ** 2
        P_field = self.cfg.P_INF + 0.5 * self.cfg.RHO * (self.cfg.V_INF ** 2) * Cp_field
        return P_field, Cp_field


# ============================================================
# Porous Network (unchanged logic; CPU only)
# ============================================================
class PorousNetwork:
    def __init__(self, aero: PanelMethod, cp_solid: np.ndarray, config: Config, topology: str = "spine"):
        self.aero = aero
        self.cfg = config
        self.topology = str(topology).lower()
        self.G = nx.Graph()
        self.active_pores = []
        self._build_network(cp_solid)

    def _build_network(self, cp_solid: np.ndarray):
        if self.topology in ("spine", "original", "baseline"):
            self._build_network_spine(cp_solid)
        elif self.topology in ("suction_web", "web", "suction"):
            self._build_network_suction_web(cp_solid)
        elif self.topology in ("pressure_web", "pressure", "mirrored"):
            self._build_network_pressure_web(cp_solid)
        else:
            raise ValueError(f"Unknown network topology: {self.topology}")

    def _build_network_spine(self, cp_solid: np.ndarray):
        xc, yc = self.aero.XC, self.aero.YC
        self.G = nx.Graph()

        # Internal reference node (spar/plenum)
        self.spar_id = -1
        spar_pos = np.array([0.25, 0.0], dtype=float)
        self.G.add_node(self.spar_id, pos=spar_pos, type="internal")

        x_in_min = float(getattr(self.cfg, "SPINE_INLET_X_MIN", 0.80))
        x_in_max = float(getattr(self.cfg, "SPINE_INLET_X_MAX", 1.00))

        x_out_min = float(getattr(self.cfg, "SPINE_OUTLET_X_MIN", 0.05))
        x_out_max = float(getattr(self.cfg, "SPINE_OUTLET_X_MAX", 0.20))

        inlet_candidates = [
            i for i in range(len(xc))
            if (yc[i] > 0.0) and (x_in_min <= xc[i] <= x_in_max)
        ]
        outlet_candidates = [
            i for i in range(len(xc))
            if (yc[i] < 0.0) and (x_out_min <= xc[i] <= x_out_max)
        ]

        inlet_scores = [{"id": i, "cp": float(cp_solid[i])} for i in inlet_candidates]
        inlet_scores.sort(key=lambda x: x["cp"], reverse=True)
        selected_inlets = [x["id"] for x in inlet_scores[: self.cfg.N_INLETS]]

        outlet_scores = [{"id": i, "cp": float(cp_solid[i])} for i in outlet_candidates]
        outlet_scores.sort(key=lambda x: x["cp"])
        num_outlets_to_use = min(self.cfg.N_OUTLETS, len(outlet_scores))
        selected_outlets = [x["id"] for x in outlet_scores[:num_outlets_to_use]]

        self.active_pores = selected_inlets + selected_outlets

        avg_cp_in = float(np.mean([cp_solid[i] for i in selected_inlets])) if selected_inlets else 0.0
        avg_cp_out = float(np.mean([cp_solid[i] for i in selected_outlets])) if selected_outlets else 0.0
        estimated_spar_cp = 0.5 * (avg_cp_in + avg_cp_out)

        print(f"   -> Generating Smart Recirculation: {len(selected_inlets)} Inlets -> {len(selected_outlets)} Outlets.")

        for pid in self.active_pores:
            p_pos = np.array([xc[pid], yc[pid]], dtype=float)
            dist = float(np.linalg.norm(p_pos - spar_pos))

            if pid not in self.G:
                self.G.add_node(pid, pos=(float(xc[pid]), float(yc[pid])), type="boundary", panel_idx=int(pid))

            local_cp = float(cp_solid[pid])
            delta_cp = abs(local_cp - estimated_spar_cp) + 0.01

            if pid in selected_inlets:
                r_eff = float(self.cfg.PORE_RADIUS_INLET)
                etype = "plenum_in"
                cond = (np.pi * r_eff ** 4) / (8.0 * self.cfg.MU * dist)
            else:
                r_base = float(self.cfg.PORE_RADIUS_OUTLET)
                area_ratio = len(selected_outlets) / max(len(selected_inlets), 1)
                damping = 1.0 / max(area_ratio, 1.0)
                throttle = np.sqrt(0.1 / delta_cp)
                throttle = min(float(throttle), 1.0)
                cond = ((np.pi * r_base ** 4) / (8.0 * self.cfg.MU * dist)) * damping * throttle
                etype = "plenum_out"

            self.G.add_edge(pid, self.spar_id, length=dist, cond=float(cond), type=etype)
    
    
    def _build_network_suction_web(self, cp_solid: np.ndarray):
        """
        Suction-side openings connected ONLY to an internal spine.
        - suction side only (yc > 0)
        - select 4 lowest Cp + 3 low-Cp pores from 3 x-bins (equi-spaced)
        - enforce minimum spacing so pores don't overlap
        - build internal spine (chain of internal nodes)
        - connect each pore to nearest spine node
        - connect spine to spar/plenum
        """
        xc, yc = self.aero.XC, self.aero.YC
        cp = np.asarray(cp_solid, dtype=float)

        self.G = nx.Graph()
        self.active_pores = []

        # -----------------------
        # Internal spar/plenum node
        # -----------------------
        self.spar_id = -1
        spar_pos = np.array([0.25, 0.0], dtype=float)
        self.G.add_node(self.spar_id, pos=(float(spar_pos[0]), float(spar_pos[1])), type="internal")

        # -----------------------
        # Config knobs
        # -----------------------
        n_low = int(getattr(self.cfg, "N_SUCTION_LOWEST", 4))
        n_bins = int(getattr(self.cfg, "N_SUCTION_BINS", 3))

        # overlap control
        min_pore_spacing = float(getattr(self.cfg, "MIN_PORE_SPACING", 0.03))  # chord units
        min_panel_gap = int(getattr(self.cfg, "MIN_PORE_PANEL_GAP", 2))

        # spine geometry
        n_spine_nodes = int(getattr(self.cfg, "N_SPINE_NODES", 6))
        spine_y = float(getattr(self.cfg, "SPINE_Y", 0.0))  # midline
        x_pad_lo = float(getattr(self.cfg, "SPINE_X_PAD_LO", 0.15))
        x_pad_hi = float(getattr(self.cfg, "SPINE_X_PAD_HI", 0.95))

        # radii
        r_pore_to_spine = float(getattr(self.cfg, "PORE_RADIUS_WEB", getattr(self.cfg, "PORE_RADIUS_INLET", 0.001)))
        r_spine = float(getattr(self.cfg, "PORE_RADIUS_SPINE", r_pore_to_spine))
        r_spine_to_spar = float(getattr(self.cfg, "PORE_RADIUS_WEB_TO_SPAR", r_spine))

        # -----------------------
        # Helper: Poiseuille edge
        # -----------------------
        def add_pipe(u, v, pos_u, pos_v, radius, etype):
            dist = float(np.linalg.norm(pos_u - pos_v)) + 1e-12
            cond = (np.pi * radius**4) / (8.0 * self.cfg.MU * dist)
            self.G.add_edge(u, v, length=dist, cond=float(cond), type=etype)

        # -----------------------
        # suction candidates (top surface)
        # -----------------------
        x_min = float(getattr(self.cfg, "SUCTION_PORE_X_MIN", -1e9))
        x_max = float(getattr(self.cfg, "SUCTION_PORE_X_MAX", 0.75))

        suction = np.where((yc > 0.0) & (xc >= x_min) & (xc <= x_max))[0]
        if suction.size == 0:
            print("   -> Suction openings: no suction-side panels (yc>0).")
            return

        def far_enough(pid, chosen):
            if not chosen:
                return True
            p = np.array([xc[pid], yc[pid]], dtype=float)
            for qid in chosen:
                if abs(int(pid) - int(qid)) < min_panel_gap:
                    return False
                q = np.array([xc[qid], yc[qid]], dtype=float)
                if float(np.linalg.norm(p - q)) < min_pore_spacing:
                    return False
            return True

        # -----------------------
        # 1) pick N lowest Cp (most negative) with spacing
        # -----------------------
        suction_sorted = suction[np.argsort(cp[suction])]
        pores = []
        for pid in suction_sorted:
            if len(pores) >= n_low:
                break
            pid = int(pid)
            if far_enough(pid, pores):
                pores.append(pid)

        # -----------------------
        # 2) +3 equi-spaced via x-bins, pick lowest Cp per bin with spacing
        # -----------------------
        xs = xc[suction]
        x_min, x_max = float(xs.min()), float(xs.max())
        edges = np.linspace(x_min, x_max, n_bins + 1)

        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if b == n_bins - 1:
                in_bin = suction[(xc[suction] >= lo) & (xc[suction] <= hi)]
            else:
                in_bin = suction[(xc[suction] >= lo) & (xc[suction] < hi)]
            if len(in_bin) == 0:
                continue

            in_bin_sorted = sorted([int(i) for i in in_bin], key=lambda i: float(cp[i]))
            for pid in in_bin_sorted:
                if pid in pores:
                    continue
                if far_enough(pid, pores):
                    pores.append(pid)
                    break

        # Force a pore near x_max (0.75c) so the most-aft pore is at that location
        if bool(getattr(self.cfg, "FORCE_PORE_AT_XMAX", True)):
            x_max = float(getattr(self.cfg, "SUCTION_PORE_X_MAX", 0.75))
            tol = float(getattr(self.cfg, "XMAX_TARGET_TOL", 0.02))  # ± window

            # suction candidates near x_max (but still <= x_max)
            near = [int(i) for i in suction if (xc[int(i)] >= (x_max - tol)) and (xc[int(i)] <= x_max)]
            # pick the one closest to x_max (largest x)
            near.sort(key=lambda i: float(xc[i]), reverse=True)

            if near:
                target_pid = near[0]
                if target_pid not in pores and far_enough(target_pid, pores):
                    pores.append(target_pid)
            else:
                print(f"   -> Warning: no suction panel found in [{x_max-tol:.2f}, {x_max:.2f}] to place the last pore.")

        # -----------------------
        # Add TE pores on suction side (top surface near x ~ 1)
        # -----------------------
        if bool(getattr(self.cfg, "ADD_TE_PORES", True)):
            n_te = int(getattr(self.cfg, "N_TE_PORES", 2))
            te_x_thr = float(getattr(self.cfg, "TE_X_THRESHOLD", 0.95))

            # suction side candidates near TE
            te_candidates = [int(i) for i in suction if (xc[int(i)] >= te_x_thr)]
            # sort by x descending (closest to TE first)
            te_candidates.sort(key=lambda i: float(xc[i]), reverse=True)

            added = 0
            for pid in te_candidates:
                if added >= n_te:
                    break
                if pid in pores:
                    continue
                if far_enough(pid, pores):   # uses your overlap/spacing rule
                    pores.append(pid)
                    added += 1

            if added < n_te:
                print(f"   -> Warning: only added {added}/{n_te} TE suction pores (spacing/availability limited).")

        self.active_pores = pores
        print(f"   -> Suction openings (n={len(pores)}): {pores}")

        if len(pores) == 0:
            return

        # -----------------------
        # Add boundary pore nodes
        # -----------------------
        pore_pos = {}
        for pid in pores:
            ppos = np.array([xc[pid], yc[pid]], dtype=float)
            pore_pos[pid] = ppos
            self.G.add_node(pid, pos=(float(ppos[0]), float(ppos[1])), type="boundary", panel_idx=int(pid))

        

        # -----------------------
        # Build internal spine (chain)
        # -----------------------
        px = np.array([xc[i] for i in pores], dtype=float)
        x0 = max(float(px.min()) - 0.05, x_pad_lo)
        x1 = min(float(px.max()) + 0.05, x_pad_hi)
        if x1 <= x0:
            x0, x1 = x_pad_lo, x_pad_hi

        spine_ids = []
        spine_pos = {}

        for s in range(n_spine_nodes):
            sid = -(10 + s)
            while sid in self.G:
                sid -= 1
            xs = x0 + (x1 - x0) * (s / max(n_spine_nodes - 1, 1))
            sp = np.array([float(xs), float(spine_y)], dtype=float)
            self.G.add_node(sid, pos=(float(sp[0]), float(sp[1])), type="spine")
            spine_ids.append(sid)
            spine_pos[sid] = sp

        # connect spine chain
        for a, b in zip(spine_ids[:-1], spine_ids[1:]):
            add_pipe(a, b, spine_pos[a], spine_pos[b], r_spine, "spine_link")

        # connect spine to spar (tie into system)
        add_pipe(spine_ids[0], self.spar_id, spine_pos[spine_ids[0]], spar_pos, r_spine_to_spar, "spine_to_spar")

        # -----------------------
        # Connect each pore to nearest spine node (ONLY connection for pores)
        # -----------------------
        for pid in pores:
            ppos = pore_pos[pid]
            nearest = min(spine_ids, key=lambda sid: float(np.linalg.norm(ppos - spine_pos[sid])))
            add_pipe(pid, nearest, ppos, spine_pos[nearest], r_pore_to_spine, "pore_to_spine")
            
    def _build_network_pressure_web(self, cp_solid: np.ndarray):
        """
        Pressure-side (mirrored) openings connected ONLY to an internal spine.

        Strategy:
        1) Build the same suction-side pore *selection* (for x-locations)
        2) Mirror each selected suction pore to pressure side by choosing
            yc<0 panel with closest xc (same x)
        3) Build network using those pressure-side pores:
            pore -> nearest spine node
            spine chain -> spar
        """
        xc, yc = self.aero.XC, self.aero.YC
        cp = np.asarray(cp_solid, dtype=float)

        self.G = nx.Graph()
        self.active_pores = []

        # -----------------------
        # Internal spar/plenum node
        # -----------------------
        self.spar_id = -1
        spar_pos = np.array([0.25, 0.0], dtype=float)
        self.G.add_node(self.spar_id, pos=(float(spar_pos[0]), float(spar_pos[1])), type="internal")

        # -----------------------
        # Config knobs (same as suction)
        # -----------------------
        n_low = int(getattr(self.cfg, "N_SUCTION_LOWEST", 4))
        n_bins = int(getattr(self.cfg, "N_SUCTION_BINS", 3))

        min_pore_spacing = float(getattr(self.cfg, "MIN_PORE_SPACING", 0.03))
        min_panel_gap = int(getattr(self.cfg, "MIN_PORE_PANEL_GAP", 2))

        n_spine_nodes = int(getattr(self.cfg, "N_SPINE_NODES", 6))
        spine_y = float(getattr(self.cfg, "SPINE_Y", 0.0))
        x_pad_lo = float(getattr(self.cfg, "SPINE_X_PAD_LO", 0.15))
        x_pad_hi = float(getattr(self.cfg, "SPINE_X_PAD_HI", 0.95))

        r_pore_to_spine = float(getattr(self.cfg, "PORE_RADIUS_WEB", getattr(self.cfg, "PORE_RADIUS_INLET", 0.001)))
        r_spine = float(getattr(self.cfg, "PORE_RADIUS_SPINE", r_pore_to_spine))
        r_spine_to_spar = float(getattr(self.cfg, "PORE_RADIUS_WEB_TO_SPAR", r_spine))

        def add_pipe(u, v, pos_u, pos_v, radius, etype):
            dist = float(np.linalg.norm(pos_u - pos_v)) + 1e-12
            cond = (np.pi * radius**4) / (8.0 * self.cfg.MU * dist)
            self.G.add_edge(u, v, length=dist, cond=float(cond), type=etype)

        # -----------------------
        # 1) Select suction-side pores FIRST (as reference x-locations)
        # -----------------------
        x_min = float(getattr(self.cfg, "SUCTION_PORE_X_MIN", -1e9))
        x_max = float(getattr(self.cfg, "SUCTION_PORE_X_MAX", 0.75))

        suction = np.where((yc > 0.0) & (xc >= x_min) & (xc <= x_max))[0]
        pressure = np.where((yc < 0.0) & (xc >= x_min) & (xc <= x_max))[0]

        if suction.size == 0:
            print("   -> Pressure web: no suction panels available to mirror from.")
            return
        if pressure.size == 0:
            print("   -> Pressure web: no pressure-side panels found (yc<0) in x-range.")
            return

        def far_enough_by_pos(pid, chosen, xy):
            """Spacing check using actual positions for candidate pid."""
            if not chosen:
                return True
            p = xy[pid]
            for qid in chosen:
                if abs(int(pid) - int(qid)) < min_panel_gap:
                    return False
                if float(np.linalg.norm(p - xy[qid])) < min_pore_spacing:
                    return False
            return True

        # suction positions for spacing during suction selection
        xy = {int(i): np.array([xc[int(i)], yc[int(i)]], dtype=float) for i in suction}

        suction_sorted = suction[np.argsort(cp[suction])]
        suction_pores = []
        for pid in suction_sorted:
            if len(suction_pores) >= n_low:
                break
            pid = int(pid)
            if far_enough_by_pos(pid, suction_pores, xy):
                suction_pores.append(pid)

        xs = xc[suction]
        edges = np.linspace(float(xs.min()), float(xs.max()), n_bins + 1)
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if b == n_bins - 1:
                in_bin = suction[(xc[suction] >= lo) & (xc[suction] <= hi)]
            else:
                in_bin = suction[(xc[suction] >= lo) & (xc[suction] < hi)]
            if len(in_bin) == 0:
                continue
            in_bin_sorted = sorted([int(i) for i in in_bin], key=lambda i: float(cp[i]))
            for pid in in_bin_sorted:
                if pid in suction_pores:
                    continue
                if far_enough_by_pos(pid, suction_pores, xy):
                    suction_pores.append(pid)
                    break

        # Force pore near x_max (0.75c) if enabled
        if bool(getattr(self.cfg, "FORCE_PORE_AT_XMAX", True)):
            x_max_target = float(getattr(self.cfg, "SUCTION_PORE_X_MAX", 0.75))
            tol = float(getattr(self.cfg, "XMAX_TARGET_TOL", 0.02))
            near = [int(i) for i in suction if (xc[int(i)] >= (x_max_target - tol)) and (xc[int(i)] <= x_max_target)]
            near.sort(key=lambda i: float(xc[i]), reverse=True)
            if near:
                pid = near[0]
                if pid not in suction_pores and far_enough_by_pos(pid, suction_pores, xy):
                    suction_pores.append(pid)

        suction_pores = list(dict.fromkeys(suction_pores))

        # -----------------------
        # 2) Mirror suction pores to pressure side by nearest xc
        # -----------------------
        pressure_candidates = [int(i) for i in pressure]
        pressure_pores = []
        for sp in suction_pores:
            x_target = float(xc[int(sp)])
            best = min(pressure_candidates, key=lambda j: abs(float(xc[j]) - x_target))
            pressure_pores.append(int(best))

        # De-duplicate and re-apply spacing on pressure side
        pressure_pores = list(dict.fromkeys(pressure_pores))
        pxy = {int(i): np.array([xc[int(i)], yc[int(i)]], dtype=float) for i in pressure_candidates}

        filtered = []
        for pid in pressure_pores:
            if far_enough_by_pos(pid, filtered, pxy):
                filtered.append(pid)
        pressure_pores = filtered

        self.active_pores = pressure_pores
        print(f"   -> Pressure openings mirrored from suction (n={len(pressure_pores)}): {pressure_pores}")

        if len(pressure_pores) == 0:
            return

        # -----------------------
        # Add boundary pore nodes (pressure side)
        # -----------------------
        pore_pos = {}
        for pid in pressure_pores:
            ppos = np.array([xc[pid], yc[pid]], dtype=float)
            pore_pos[pid] = ppos
            self.G.add_node(pid, pos=(float(ppos[0]), float(ppos[1])), type="boundary", panel_idx=int(pid))

        # -----------------------
        # Build internal spine (chain)
        # -----------------------
        px = np.array([xc[i] for i in pressure_pores], dtype=float)
        x0 = max(float(px.min()) - 0.05, x_pad_lo)
        x1 = min(float(px.max()) + 0.05, x_pad_hi)
        if x1 <= x0:
            x0, x1 = x_pad_lo, x_pad_hi

        spine_ids = []
        spine_pos = {}
        for s in range(n_spine_nodes):
            sid = -(10 + s)
            while sid in self.G:
                sid -= 1
            xs = x0 + (x1 - x0) * (s / max(n_spine_nodes - 1, 1))
            sp = np.array([float(xs), float(spine_y)], dtype=float)
            self.G.add_node(sid, pos=(float(sp[0]), float(sp[1])), type="spine")
            spine_ids.append(sid)
            spine_pos[sid] = sp

        for a, b in zip(spine_ids[:-1], spine_ids[1:]):
            add_pipe(a, b, spine_pos[a], spine_pos[b], r_spine, "spine_link")

        add_pipe(spine_ids[0], self.spar_id, spine_pos[spine_ids[0]], spar_pos, r_spine_to_spar, "spine_to_spar")

        # -----------------------
        # Connect each pressure pore to nearest spine node
        # -----------------------
        for pid in pressure_pores:
            ppos = pore_pos[pid]
            nearest = min(spine_ids, key=lambda sid: float(np.linalg.norm(ppos - spine_pos[sid])))
            add_pipe(pid, nearest, ppos, spine_pos[nearest], r_pore_to_spine, "pore_to_spine")

    def solve_flow(self, P_boundary: dict):
        """
        Passive porous network solve (NO check valves).
        Given boundary pressures at pore nodes, solve internal pressures, then compute
        pore velocities from resulting flows. Flow direction is determined naturally
        by pressure differences.

        Returns:
            velocities: dict(panel_idx -> normal velocity)
            P_nodes: ndarray aligned with nodes list ordering
        """
        nodes = list(self.G.nodes())
        n = len(nodes)
        node_map = {node: i for i, node in enumerate(nodes)}
        boundary_nodes = [nd for nd in self.G.nodes() if self.G.nodes[nd].get("type") == "boundary"]

        # Build linear system A P = b
        A = sp.lil_matrix((n, n), dtype=float)
        b = np.zeros(n, dtype=float)

        for node in nodes:
            idx = node_map[node]
            if node in boundary_nodes:
                pid = self.G.nodes[node]["panel_idx"]
                A[idx, idx] = 1.0
                b[idx] = float(P_boundary.get(pid, 0.0))
            else:
                sigma_cond = 0.0
                for nbr in self.G.neighbors(node):
                    c = float(self.G[node][nbr].get("cond", 0.0))
                    nbr_idx = node_map[nbr]
                    A[idx, nbr_idx] = -c
                    sigma_cond += c
                A[idx, idx] = sigma_cond if sigma_cond > 0.0 else 1.0

        try:
            P_nodes = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        except Exception:
            return {}, np.zeros(n, dtype=float)

        P_nodes = np.asarray(P_nodes, dtype=float)

        # Compute velocities at boundary pores from net volumetric flow
        velocities = {}
        for node in boundary_nodes:
            pid = self.G.nodes[node]["panel_idx"]
            idx = node_map[node]

            # Use radius based on the edge "type" only to compute area (NOT for valves)
            # If you don't care about different inlet/outlet sizes, you can just pick one radius.
            is_inlet = any(self.G[node][nbr].get("type") == "plenum_in" for nbr in self.G.neighbors(node))
            radius = float(self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET)
            area = np.pi * radius * radius

            Q_net = 0.0
            for nbr in self.G.neighbors(node):
                c = float(self.G[node][nbr].get("cond", 0.0))
                nbr_idx = node_map[nbr]
                Q_net += c * (P_nodes[idx] - P_nodes[nbr_idx])

            # Same sign convention as before
            velocities[pid] = -Q_net / (area + 1e-30)

        return velocities, P_nodes


def compute_forces(aero: PanelMethod, Cp: np.ndarray):
    """
    Integrate pressure coefficient over panels to get (CL, CD) in freestream axes.
    """
    Cp = np.asarray(Cp, dtype=float)
    fx = -Cp * aero.nx * aero.L
    fy = -Cp * aero.ny * aero.L

    Fx = float(np.sum(fx))
    Fy = float(np.sum(fy))

    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    return CL, CD


# ============================================================
# Anderson(m) monolithic coupled solver 
# ============================================================
class MonolithicCoupledSolverAnderson:
    """
    Fixed-point solve: v = F(v)
    Solve for active pore leakage velocities v_active using Anderson acceleration.

    This is robust when the mapping has kinks (your check valves).
    """
    def __init__(self, aero: PanelMethod, net: PorousNetwork, cfg: Config, v_clip: float = 80.0):
        self.aero = aero
        self.net = net
        self.cfg = cfg
        self.active = np.array(net.active_pores, dtype=int)
        self.v_clip = float(v_clip)
        self.last_P_nodes = None

    def _F(self, v_active: np.ndarray):
        V = np.zeros(self.aero.N, dtype=float)
        V[self.active] = np.clip(v_active, -self.v_clip, self.v_clip)

        Cp = self.aero.solve(V)
        P_ext = self.cfg.P_INF + 0.5 * self.cfg.RHO * (self.cfg.V_INF ** 2) * Cp
        P_map = {pid: float(P_ext[pid]) for pid in self.net.active_pores}

        vel_calc, P_nodes = self.net.solve_flow(P_map)
        self.last_P_nodes = P_nodes

        F_active = np.array([vel_calc.get(pid, 0.0) for pid in self.active], dtype=float)
        F_active = np.clip(F_active, -self.v_clip, self.v_clip)
        return F_active, Cp

    def solve(
        self,
        v0=None,
        tol=None,
        maxiter=None,
        m=None,
        beta=None,
        verbose=True,
        callback=None,
    ):
        if v0 is None:
            v = np.zeros(len(self.active), dtype=float)
        else:
            v = np.asarray(v0, dtype=float).copy()

        tol = float(self.cfg.CONVERGENCE_TOL if tol is None else tol)
        maxiter = int(self.cfg.ANDERSON_MAXITER if maxiter is None else maxiter)
        m = int(self.cfg.ANDERSON_M if m is None else m)
        beta = float(self.cfg.ANDERSON_BETA if beta is None else beta)

        G_hist = []
        dV_hist = []

        v_prev = None
        g_prev = None

        for k in range(maxiter):
            Fv, Cp = self._F(v)
            g = Fv - v
            g_inf = float(np.max(np.abs(g)))

            # callback logging every iteration
            if callback is not None:
                callback(
                    k,
                    v.copy(),
                    g_inf,
                    self.aero.q.copy(),
                    float(self.aero.gamma),
                    Cp.copy(),
                    self.last_P_nodes.copy() if self.last_P_nodes is not None else None,
                )

            if verbose and (k % 5 == 0 or g_inf < tol):
                print(f"   AND iter {k:3d}: ||g||_inf = {g_inf:.3e}")

            if g_inf < tol and k > 2:
                break

            # base relaxed update
            v_new = v + beta * g

            # Anderson mixing using secant differences
            if k > 0 and v_prev is not None and g_prev is not None:
                dv = v - v_prev
                dg = g - g_prev

                dV_hist.append(dv)
                G_hist.append(dg)

                if len(G_hist) > m:
                    G_hist.pop(0)
                    dV_hist.pop(0)

                if len(G_hist) >= 2:
                    DG = np.column_stack(G_hist)
                    lam = float(getattr(self.cfg, "ANDERSON_DAMPING", 0.0))
                    A = DG.T @ DG + lam * np.eye(DG.shape[1])
                    b = DG.T @ g
                    try:
                        c = np.linalg.solve(A, b)
                        DV = np.column_stack(dV_hist)
                        v_new = v + beta * g - DV @ c
                    except np.linalg.LinAlgError:
                        pass

            v_prev = v
            g_prev = g
            v = np.clip(v_new, -self.v_clip, self.v_clip)

        Fv, Cp = self._F(v)
        return v, Cp, self.last_P_nodes


# ============================================================
# Core runner (Anderson)
# ============================================================
def run_core_anderson(X, Y, cfg: Config):
    aero = PanelMethod(X, Y, cfg)

    # baseline solid
    Cp_solid = aero.solve(np.zeros(aero.N, dtype=float))
    CL_solid, CD_solid = compute_forces(aero, Cp_solid)

    # snapshot for plotting (solid)
    aero_solid = PanelMethod(X, Y, cfg)
    _ = aero_solid.solve(np.zeros(aero_solid.N, dtype=float))


    # network from solid Cp
    topology = getattr(cfg, "NETWORK_TOPOLOGY", "spine")
    net = PorousNetwork(aero, Cp_solid, cfg, topology=topology)

    print("-> Solving coupled system with Anderson acceleration...")
    coupled = MonolithicCoupledSolverAnderson(aero, net, cfg, v_clip=80.0)

    v0 = np.zeros(len(net.active_pores), dtype=float)
    v_active, Cp, P_nodes = coupled.solve(v0=v0, verbose=True)

    V_leakage = np.zeros(aero.N, dtype=float)
    V_leakage[np.array(net.active_pores, dtype=int)] = np.clip(v_active, -80.0, 80.0)

    CL, CD = compute_forces(aero, Cp)

    return {
        "aero": aero,
        "aero_solid": aero_solid,
        "net": net,
        "Cp": Cp,
        "Cp_solid": Cp_solid,
        "V_leakage": V_leakage,
        "P_nodes": P_nodes,
        "CL": CL,
        "CD": CD,
        "CL_solid": CL_solid,
        "CD_solid": CD_solid,
    }