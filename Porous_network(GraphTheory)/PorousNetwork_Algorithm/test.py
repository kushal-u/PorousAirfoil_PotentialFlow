import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
from scipy.interpolate import griddata
import warnings
import os
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    """Central configuration for physics and geometry."""
    # File / Output
    AIRFOIL_NAME: str = "0018"
    OUTPUT_DIR: str = "porous_airfoil_results"
    
    # Geometry
    N_PANELS: int = 320
    CHORD: float = 1.0
    
    # Porous Network Settings (Explicitly defined now)
    PORE_RADIUS_INLET: float = 3000e-6
    PORE_RADIUS_OUTLET: float = 4000e-6
    N_INLETS: int = 40   # Bottom/LE
    N_OUTLETS: int = 15  # Top/TE
    
    # Physics
    REYNOLDS_NUM: float = 250000
    ANGLE_OF_ATTACK: float = 6.0
    RHO: float = 1.225
    MU: float = 1.78e-5
    P_INF: float = 0.0
    
    # Solver
    MAX_ITER: int = 100
    RELAXATION: float = 0.01
    CONVERGENCE_TOL: float = 1e-8

    @property
    def V_INF(self):
        return (self.REYNOLDS_NUM * self.MU) / (self.RHO * self.CHORD)

# ==============================================================================
# 2. GEOMETRY GENERATOR
# ==============================================================================
class AirfoilGenerator:
    @staticmethod
    def generate_naca4(number, n_panels=160):
        """Generates NACA 4-digit airfoil coordinates."""
        m = int(number[0]) / 100.0
        p = int(number[1]) / 10.0
        t = int(number[2:]) / 100.0

        beta = np.linspace(0, np.pi, n_panels // 2 + 1)
        x = (1 - np.cos(beta)) / 2
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
        
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        if m != 0:
            yc[x <= p] = m / p**2 * (2 * p * x[x <= p] - x[x <= p]**2)
            yc[x > p] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[x > p] - x[x > p]**2)
            dyc_dx[x <= p] = 2 * m / p**2 * (p - x[x <= p])
            dyc_dx[x > p] = 2 * m / (1 - p)**2 * (p - x[x > p])

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
        self.q = np.zeros(self.N)
        self.gamma = 0.0

    def _build_influence_matrices(self):
        self.Is_n = np.zeros((self.N, self.N))
        self.Iv_n = np.zeros((self.N, self.N))
        self.Is_t = np.zeros((self.N, self.N))
        self.Iv_t = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.Is_n[i, j] = 0.5 * np.pi
                    self.Iv_t[i, j] = 0.5 * np.pi
                    continue

                dx = self.XC[i] - self.X[j]
                dy = self.YC[i] - self.Y[j]
                x_loc =  dx * self.tx[j] + dy * self.ty[j]
                y_loc = -dx * self.ty[j] + dy * self.tx[j]

                r1_sq = x_loc**2 + y_loc**2
                r2_sq = (x_loc - self.L[j])**2 + y_loc**2
                
                theta1 = np.arctan2(y_loc, x_loc)
                theta2 = np.arctan2(y_loc, x_loc - self.L[j])
                dtheta = theta2 - theta1
                if dtheta > np.pi: dtheta -= 2*np.pi
                elif dtheta < -np.pi: dtheta += 2*np.pi

                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc =  1.0 / np.pi * dtheta
                
                # Global transformation
                us_glob = us_loc * self.tx[j] - vs_loc * self.ty[j]
                vs_glob = us_loc * self.ty[j] + vs_loc * self.tx[j]
                uv_glob = -vs_loc * self.tx[j] - us_loc * self.ty[j] # uv_loc = -vs_loc
                vv_glob = -vs_loc * self.ty[j] + us_loc * self.tx[j] # vv_loc = us_loc

                self.Is_n[i, j] = us_glob * self.nx[i] + vs_glob * self.ny[i]
                self.Is_t[i, j] = us_glob * self.tx[i] + vs_glob * self.ty[i]
                self.Iv_n[i, j] = uv_glob * self.nx[i] + vv_glob * self.ny[i]
                self.Iv_t[i, j] = uv_glob * self.tx[i] + vv_glob * self.ty[i]

    def solve(self, V_leakage=None):
        if V_leakage is None: V_leakage = np.zeros(self.N)
        
        Vinf_x = self.cfg.V_INF * np.cos(self.alpha)
        Vinf_y = self.cfg.V_INF * np.sin(self.alpha)
        Vinf_n = Vinf_x * self.nx + Vinf_y * self.ny
        Vinf_t = Vinf_x * self.tx + Vinf_y * self.ty

        A = np.zeros((self.N + 1, self.N + 1))
        b = np.zeros(self.N + 1)

        A[:self.N, :self.N] = self.Is_n
        A[:self.N, self.N] = np.sum(self.Iv_n, axis=1) 
        b[:self.N] = V_leakage - Vinf_n

        A[self.N, :self.N] = self.Is_t[0, :] + self.Is_t[self.N-1, :]
        A[self.N, self.N] = np.sum(self.Iv_t[0, :] + self.Iv_t[self.N-1, :])
        b[self.N] = -(Vinf_t[0] + Vinf_t[self.N-1])

        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(self.N)

        self.q = x[:self.N]
        self.gamma = x[self.N]
        
        Vt = Vinf_t + np.dot(self.Is_t, self.q) + self.gamma * np.sum(self.Iv_t, axis=1)
        Cp = 1.0 - (Vt / self.cfg.V_INF)**2
        return Cp

    def compute_velocity_field(self, X_grid, Y_grid):
        u = np.zeros_like(X_grid) + self.cfg.V_INF * np.cos(self.alpha)
        v = np.zeros_like(Y_grid) + self.cfg.V_INF * np.sin(self.alpha)

        for j in range(self.N):
            dx = X_grid - self.X[j]
            dy = Y_grid - self.Y[j]
            x_loc =  dx * self.tx[j] + dy * self.ty[j]
            y_loc = -dx * self.ty[j] + dy * self.tx[j]
            
            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - self.L[j])**2 + y_loc**2
            
            theta1 = np.arctan2(y_loc, x_loc)
            theta2 = np.arctan2(y_loc, x_loc - self.L[j])
            dtheta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
            
            us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
            vs_loc =  1.0 / np.pi * dtheta
            
            u_ind = (us_loc * self.q[j] - vs_loc * self.gamma) * self.tx[j] - \
                    (vs_loc * self.q[j] + us_loc * self.gamma) * self.ty[j]
            v_ind = (us_loc * self.q[j] - vs_loc * self.gamma) * self.ty[j] + \
                    (vs_loc * self.q[j] + us_loc * self.gamma) * self.tx[j]
            u += u_ind
            v += v_ind
        return u, v

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
        """Constructs the internal plenum graph."""
        xc, yc = self.aero.XC, self.aero.YC
        
        # Plenum Centroid
        plenum_id = 99999
        plenum_pos = np.array([np.mean(xc), np.mean(yc)])
        self.G.add_node(plenum_id, pos=plenum_pos, type='internal')

        # Filter Candidates
        outlet_candidates = [i for i in range(len(xc)) if yc[i] > 0 and xc[i] >= 0.85]
        inlet_candidates = [i for i in range(len(xc)) if yc[i] < 0 and 0.02 <= xc[i] <= 0.20]

        # Select Inlets (Highest Cp)
        inlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in inlet_candidates]
        inlet_scores.sort(key=lambda x: x['cp'], reverse=True)
        selected_inlets = [x['id'] for x in inlet_scores[:self.cfg.N_INLETS]]

        # Select Outlets (Rearmost)
        outlet_scores = [{'id': i, 'x': xc[i]} for i in outlet_candidates]
        outlet_scores.sort(key=lambda x: x['x'], reverse=True)
        selected_outlets = [x['id'] for x in outlet_scores[:self.cfg.N_OUTLETS]]

        self.active_pores = selected_inlets + selected_outlets

        # Build Edges
        # 1. Inlets
        for u in selected_inlets:
            length = np.linalg.norm(np.array([xc[u], yc[u]]) - plenum_pos)
            cond = (np.pi * self.cfg.PORE_RADIUS_INLET**4) / (8 * self.cfg.MU * length)
            if u not in self.G: 
                self.G.add_node(u, pos=(xc[u], yc[u]), type='boundary', panel_idx=u)
            self.G.add_edge(u, plenum_id, length=length, cond=cond, type='plenum_in')

        # 2. Outlets
        for v in selected_outlets:
            length = np.linalg.norm(np.array([xc[v], yc[v]]) - plenum_pos)
            cond = (np.pi * self.cfg.PORE_RADIUS_OUTLET**4) / (8 * self.cfg.MU * length)
            if v not in self.G: 
                self.G.add_node(v, pos=(xc[v], yc[v]), type='boundary', panel_idx=v)
            self.G.add_edge(plenum_id, v, length=length, cond=cond, type='plenum_out')

    def solve_flow(self, P_boundary):
        """Solves Darcy flow through the network."""
        nodes = list(self.G.nodes())
        n = len(nodes)
        node_map = {node: i for i, node in enumerate(nodes)}

        A = sp.lil_matrix((n, n))
        b = np.zeros(n)
        boundary_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['type'] == 'boundary']

        for node in nodes:
            idx = node_map[node]
            if node in boundary_nodes:
                pid = self.G.nodes[node]['panel_idx']
                if pid in P_boundary:
                    A[idx, idx] = 1.0
                    b[idx] = P_boundary[pid]
                else:
                    A[idx, idx] = 1.0
            else:
                sigma_cond = 0.0
                for nbr in self.G.neighbors(node):
                    c = self.G[node][nbr]['cond']
                    nbr_idx = node_map[nbr]
                    A[idx, nbr_idx] = -c
                    sigma_cond += c
                A[idx, idx] = sigma_cond

        try:
            P_nodes = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        except:
            return {}, np.zeros(n)

        velocities = {}
        # Note: Using INLET radius for reference area calculation can be ambiguous
        # but maintaining logic from original script for consistency
        # Ideally, we should use the specific radius of the pore in question
        
        for node in boundary_nodes:
            pid = self.G.nodes[node]['panel_idx']
            idx = node_map[node]
            
            # Determine which radius to use based on connectivity
            # (Simplified check based on original logic)
            is_inlet = any(self.G[node][nbr]['type'] == 'plenum_in' for nbr in self.G.neighbors(node))
            radius = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * radius**2

            Q_net = 0
            for nbr in self.G.neighbors(node):
                c = self.G[node][nbr]['cond']
                nbr_idx = node_map[nbr]
                Q_net += c * (P_nodes[idx] - P_nodes[nbr_idx])
            velocities[pid] = -Q_net / area

        return velocities, P_nodes

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
class Visualizer:
    def __init__(self, config: Config):
        self.cfg = config
        
        # Setup output directory
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        self.output_dir = os.path.join(base_dir, self.cfg.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_csv(self, aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid):
        path = os.path.join(self.output_dir, 'simulation_data.csv')
        with open(path, 'w') as f:
            f.write("Metric,Solid_Baseline,Porous_Result,Change_Percent\n")
            cl_chg = ((CL - CL_solid) / (abs(CL_solid) + 1e-9)) * 100
            cd_chg = ((CD - CD_solid) / (abs(CD_solid) + 1e-9)) * 100
            f.write(f"CL,{CL_solid:.6f},{CL:.6f},{cl_chg:.2f}%\n")
            f.write(f"CD,{CD_solid:.6f},{CD:.6f},{cd_chg:.2f}%\n\n")
            f.write("Panel_ID,XC,YC,Cp_Solid,Cp_Porous,V_leakage\n")
            for i in range(aero.N):
                f.write(f"{i},{aero.XC[i]:.6f},{aero.YC[i]:.6f},{Cp_solid[i]:.6f},{Cp[i]:.6f},{V_leakage[i]:.6f}\n")

    def plot_all(self, aero, porous_net, Cp, Cp_solid, P_nodes):
        print(f"-> Generating plots in {self.output_dir}...")
        self._plot_geometry_cp(aero, porous_net, Cp, Cp_solid)
        self._plot_flow_field(aero)
        self._plot_internal_flow(aero, porous_net, P_nodes)

    def _plot_geometry_cp(self, aero, net, Cp, Cp_solid):
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1)
        
        # Mesh
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(aero.X, aero.Y, 'k-')
        ax1.fill(aero.X, aero.Y, 'whitesmoke')
        pos = nx.get_node_attributes(net.G, 'pos')
        b_nodes = [n for n in net.G.nodes if net.G.nodes[n]['type']=='boundary']
        nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax1, node_size=20, node_color='r')
        nx.draw_networkx_edges(net.G, pos, ax=ax1, edge_color='b', alpha=0.3)
        ax1.axis('equal')
        ax1.set_title("Network Topology")

        # Cp
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(aero.XC, Cp_solid, 'k--', label='Solid')
        ax2.plot(aero.XC, Cp, 'b-', label='Porous')
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.legend()
        ax2.set_title("Pressure Coefficient")
        
        fig.savefig(os.path.join(self.output_dir, '01_Geometry_Cp.png'), dpi=150)
        plt.close(fig)

    def _plot_flow_field(self, aero):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        xg, yg = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.6, 0.6, 100))
        u, v = aero.compute_velocity_field(xg, yg)
        mag = np.sqrt(u**2 + v**2)
        
        ax.contourf(xg, yg, mag, 30, cmap='viridis')
        ax.fill(aero.X, aero.Y, 'k')
        ax.axis('equal')
        ax.set_title("External Velocity Magnitude")
        fig.savefig(os.path.join(self.output_dir, '05_Flow_Field.png'), dpi=150)
        plt.close(fig)

    def _plot_internal_flow(self, aero, net, P_nodes):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(aero.X, aero.Y, 'k-', lw=2)
        
        # Prepare data for interpolation
        points, values = [], []
        pos = nx.get_node_attributes(net.G, 'pos')
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}
        
        max_v = 0
        for u, v, d in net.G.edges(data=True):
            idx_u, idx_v = node_map[u], node_map[v]
            dp = P_nodes[idx_u] - P_nodes[idx_v]
            l = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            
            # Estimate velocity in pipe
            # Note: Approximating radius for viz
            rad = self.cfg.PORE_RADIUS_INLET 
            vel = (rad**2 * abs(dp)) / (8 * self.cfg.MU * l)
            max_v = max(max_v, vel)

            # Interpolate points along edge
            for t in np.linspace(0, 1, 15):
                pt = np.array(pos[u]) + t * (np.array(pos[v]) - np.array(pos[u]))
                points.append(pt)
                values.append(vel)

        if points:
            gx, gy = np.mgrid[0:1:200j, -0.2:0.2:200j]
            gz = griddata(np.array(points), np.array(values), (gx, gy), method='linear', fill_value=0)
            
            # Mask
            path = mpath.Path(np.column_stack((aero.X, aero.Y)))
            mask = path.contains_points(np.column_stack((gx.ravel(), gy.ravel()))).reshape(gx.shape)
            gz[~mask] = np.nan
            
            c = ax.contourf(gx, gy, gz, levels=50, cmap='plasma')
            plt.colorbar(c, ax=ax, label="Internal Vel [m/s]")

        ax.axis('equal')
        ax.set_title("Internal Flow Velocity")
        fig.savefig(os.path.join(self.output_dir, '06_Internal_Flow.png'), dpi=150)
        plt.close(fig)

# ==============================================================================
# 6. MAIN SIMULATION LOOP
# ==============================================================================
def run_simulation():
    cfg = Config()
    
    print(f"--- SIMULATION START: NACA {cfg.AIRFOIL_NAME} ---")
    
    # 1. Setup
    X, Y = AirfoilGenerator.generate_naca4(cfg.AIRFOIL_NAME, cfg.N_PANELS)
    aero = PanelMethod(X, Y, cfg)
    viz = Visualizer(cfg)

    # 2. Baseline Solution
    print("-> Solving Solid Baseline...")
    Cp_solid = aero.solve(np.zeros(aero.N))
    
    # Forces (Solid)
    fx = -Cp_solid * aero.nx * aero.L
    fy = -Cp_solid * aero.ny * aero.L
    Fx, Fy = np.sum(fx), np.sum(fy)
    CL_solid = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD_solid = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    print(f"   Baseline CL: {CL_solid:.4f}")

    # 3. Build Network
    print("-> Building Porous Network...")
    net = PorousNetwork(aero, Cp_solid, cfg)
    
    # 4. Iteration Loop
    V_leakage = np.zeros(aero.N)
    P_nodes = None
    
    print(f"-> Iterating (Max {cfg.MAX_ITER})...")
    for i in range(cfg.MAX_ITER):
        # External
        Cp = aero.solve(V_leakage)
        P_ext = cfg.P_INF + (0.5 * cfg.RHO * cfg.V_INF**2) * Cp
        P_map = {pid: P_ext[pid] for pid in net.active_pores}
        
        # Internal
        vel_calc, P_nodes = net.solve_flow(P_map)
        
        # Update & Relax
        max_diff = 0.0
        for pid, v_new in vel_calc.items():
            v_relaxed = cfg.RELAXATION * v_new + (1 - cfg.RELAXATION) * V_leakage[pid]
            v_relaxed = max(min(v_relaxed, 80.0), -80.0) # Limiter
            diff = abs(v_relaxed - V_leakage[pid])
            max_diff = max(max_diff, diff)
            V_leakage[pid] = v_relaxed
            
        if max_diff < cfg.CONVERGENCE_TOL and i > 5:
            print(f"   Converged at Iter {i}")
            break
        if i % 10 == 0: print(f"   Iter {i}: Resid={max_diff:.6f}")

    # 5. Final Forces
    fx = -Cp * aero.nx * aero.L
    fy = -Cp * aero.ny * aero.L
    Fx, Fy = np.sum(fx), np.sum(fy)
    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    
    print(f"-> Result: CL Solid={CL_solid:.4f} -> Porous={CL:.4f}")

    # 6. Output
    viz.save_csv(aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid)
    viz.plot_all(aero, net, Cp, Cp_solid, P_nodes)

if __name__ == "__main__":
    run_simulation()