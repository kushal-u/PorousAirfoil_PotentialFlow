# solver.py
import numpy as np
import scipy.linalg as la
import numba as nb
from input import Config

# ==============================================================================
# JIT-COMPILED VELOCITY FIELD
# ==============================================================================
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

                x_loc =  dx * tx[k] + dy * ty[k]
                y_loc = -dx * ty[k] + dy * tx[k]

                r1_sq = x_loc**2 + y_loc**2
                r2_sq = (x_loc - L[k])**2 + y_loc**2

                theta1 = np.arctan2(y_loc, x_loc)
                theta2 = np.arctan2(y_loc, x_loc - L[k])

                dtheta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi

                eps = 1e-12
                us_loc = -0.5 / np.pi * np.log((r2_sq + eps) / (r1_sq + eps))
                vs_loc =  1.0 / np.pi * dtheta

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
# AERODYNAMIC SOLVER
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

    def _prepare_linear_system(self):
        """Build the aerodynamic influence matrix A once and precompute LU factorization."""
        A = np.zeros((self.N + 1, self.N + 1))

        A[:self.N, :self.N] = self.Is_n
        A[:self.N, self.N] = np.sum(self.Iv_n, axis=1)
        A[self.N, :self.N] = self.Is_t[0, :] + self.Is_t[self.N - 1, :]
        A[self.N, self.N] = np.sum(self.Iv_t[0, :] + self.Iv_t[self.N - 1, :])

        self.lu_A, self.piv_A = la.lu_factor(A)

    def solve(self, V_leakage=None):
        """
        Solve aerodynamic system for a given leakage distribution.

        Returns
        -------
        q : ndarray
            Source strengths per panel.
        gamma : float
            Constant vortex sheet strength.
        Cp : ndarray
            Pressure coefficient distribution.
        """
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
        except Exception as e:
            raise RuntimeError(f"Aerodynamic solve failed: {e}")

        q = x[:self.N]
        gamma = x[self.N]

        Vt = Vinf_t + np.dot(self.Is_t, q) + gamma * np.sum(self.Iv_t, axis=1)
        Cp = 1.0 - (Vt / self.cfg.V_INF)**2

        if not np.all(np.isfinite(Cp)):
            raise RuntimeError("Aerodynamic solve produced non-finite Cp values.")

        return q, gamma, Cp

    def _build_influence_matrices(self):
        """Vectorized computation of aerodynamic influence matrices without Python loops."""
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

        x_loc =  dx * tx_j + dy * ty_j
        y_loc = -dx * ty_j + dy * tx_j

        r1_sq = x_loc**2 + y_loc**2
        r2_sq = (x_loc - L_j)**2 + y_loc**2

        theta1 = np.arctan2(y_loc, x_loc)
        theta2 = np.arctan2(y_loc, x_loc - L_j)

        dtheta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi

        eps = 1e-12
        us_loc = -0.5 / np.pi * np.log((r2_sq + eps) / (r1_sq + eps))
        vs_loc =  1.0 / np.pi * dtheta

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

    def compute_velocity_field(self, X_grid, Y_grid, q, gamma):
        """
        Compute velocity field for a previously solved aerodynamic state.
        """
        Vinf_x = self.cfg.V_INF * np.cos(self.alpha)
        Vinf_y = self.cfg.V_INF * np.sin(self.alpha)

        return _fast_velocity_field(
            X_grid, Y_grid,
            self.X, self.Y,
            self.tx, self.ty,
            self.L, q, gamma,
            Vinf_x, Vinf_y
        )