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
    N_PANELS: int = 3000
    CHORD: float = 1.0
    
    # Porous Network Settings 
    PORE_RADIUS_INLET: float = 10000e-6
    PORE_RADIUS_OUTLET: float = 10000e-6
    N_INLETS: int = 1   # Top/TE 
    N_OUTLETS: int = 1  # Bottom/LE 
    
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
        
        xc, yc = self.aero.XC, self.aero.YC
        self.G = nx.Graph()
        
        # ======================================================================
        # 1. CENTRAL SPAR
        # ======================================================================
        spar_id = 99999
        spar_pos = np.array([0.25, 0.0]) 
        self.G.add_node(spar_id, pos=spar_pos, type='internal')
        
        # ======================================================================
        # 2. SELECT ZONES
        # ======================================================================
        # INLETS: Top TE 
        inlet_candidates = [i for i in range(len(xc)) if yc[i] > 0 and xc[i] >= 0.85]
        
        # OUTLETS: Bottom LE ( 5% to 20% chord)
        outlet_candidates = [i for i in range(len(xc)) if yc[i] < 0 and 0.05 <= xc[i] <= 0.20]
        
        # Selection: Inlets -> Pick Best (Highest Pressure Recovery)
        inlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in inlet_candidates]
        inlet_scores.sort(key=lambda x: x['cp'], reverse=True)
        selected_inlets = [x['id'] for x in inlet_scores[:self.cfg.N_INLETS]]
        
        # Selection: Outlets -> Pick Best (Lowest Pressure for Diffusion) 
        outlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in outlet_candidates]
        outlet_scores.sort(key=lambda x: x['cp']) # Ascending order to target maximum suction areas
        
        # ENFORCEMENT:  user's N_OUTLETS setting
        num_outlets_to_use = min(self.cfg.N_OUTLETS, len(outlet_scores))
        selected_outlets = [x['id'] for x in outlet_scores[:num_outlets_to_use]]
        
        self.active_pores = selected_inlets + selected_outlets

        # Calculate Reference Pressures
        avg_cp_in = np.mean([cp_solid[i] for i in selected_inlets]) if selected_inlets else 0.0
        avg_cp_out = np.mean([cp_solid[i] for i in selected_outlets]) if selected_outlets else 0.0
        estimated_spar_cp = (avg_cp_in + avg_cp_out) / 2.0

        # ======================================================================
        # 3. CONNECT WITH ADAPTIVE LOGIC
        # ======================================================================
        print(f"   -> Generating Smart Recirculation: {len(selected_inlets)} Inlets -> {len(selected_outlets)} Outlets.")
        
        for pid in self.active_pores:
            p_pos = np.array([xc[pid], yc[pid]])
            dist = np.linalg.norm(p_pos - spar_pos)
            
            if pid not in self.G:
                self.G.add_node(pid, pos=(xc[pid], yc[pid]), type='boundary', panel_idx=pid)
            
            local_cp = cp_solid[pid]
            delta_cp = abs(local_cp - estimated_spar_cp) + 0.01
            
            if pid in selected_inlets:
                # INLET: Low Resistance
                r_eff = self.cfg.PORE_RADIUS_INLET
                etype = 'plenum_in'
                cond = (np.pi * r_eff**4) / (8 * self.cfg.MU * dist)
                
            else:
                # OUTLET: ADAPTIVE THROTTLING
                r_base = self.cfg.PORE_RADIUS_OUTLET 
                
                # Area Damping (Balance Mass Flow)
                area_ratio = len(selected_outlets) / max(len(selected_inlets), 1)
                damping = 1.0 / max(area_ratio, 1.0)
                
                # Iso-Velocity Throttling
                throttle = np.sqrt(0.1 / delta_cp)
                throttle = min(throttle, 1.0)
                
                cond = ((np.pi * r_base**4) / (8 * self.cfg.MU * dist)) * damping * throttle
                etype = 'plenum_out'
                
            self.G.add_edge(pid, spar_id, length=dist, cond=cond, type=etype)

    def solve_flow(self, P_boundary):
        
        nodes = list(self.G.nodes())
        n = len(nodes)
        node_map = {node: i for i, node in enumerate(nodes)}
        spar_node = 99999
        boundary_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['type'] == 'boundary']

        closed_valves = set() # Store nodes whose check valves have been triggered shut

        # Sub-iteration loop to resolve any backflow attempts mathematically
        for _ in range(10): 
            A = sp.lil_matrix((n, n))
            b = np.zeros(n)

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
                        # If a check valve is triggered, this branch acts as if it has zero conductance
                        if nbr in closed_valves:
                            continue 
                        
                        c = self.G[node][nbr]['cond']
                        nbr_idx = node_map[nbr]
                        A[idx, nbr_idx] = -c
                        sigma_cond += c
                    
                    # Prevent a singular matrix crash if all check-valves to the spar close entirely
                    A[idx, idx] = sigma_cond if sigma_cond > 0 else 1.0 

            try:
                P_nodes = scipy.sparse.linalg.spsolve(A.tocsr(), b)
            except:
                return {}, np.zeros(n)

            # Check Valve Logic: Detect if the pressure gradient is trying to reverse the flow
            new_valves_closed = False
            spar_idx = node_map[spar_node]
            p_spar = P_nodes[spar_idx]

            for node in boundary_nodes:
                if node in closed_valves:
                    continue
                
                is_inlet = any(self.G[node][nbr]['type'] == 'plenum_in' for nbr in self.G.neighbors(node))
                pid = self.G.nodes[node]['panel_idx']
                p_bound = P_boundary.get(pid, p_spar)

                # INLET rule: Air must flow IN (P_bound > P_spar). If P_bound < P_spar, reverse flow occurs.
                if is_inlet and p_bound < p_spar:
                    closed_valves.add(node)
                    new_valves_closed = True
                
                # OUTLET rule: Air must flow OUT (P_spar > P_bound). If P_spar < P_bound, reverse flow occurs.
                elif not is_inlet and p_spar < p_bound:
                    closed_valves.add(node)
                    new_valves_closed = True

            # If no new check-valves had to close this iteration, the mass flow network is stable and valid
            if not new_valves_closed:
                break

        # Calculate final constrained velocities ensuring strict mass conservation
        velocities = {}
        for node in boundary_nodes:
            pid = self.G.nodes[node]['panel_idx']
            
            # If the valve shut to prevent backflow, the velocity is strictly zero
            if node in closed_valves:
                velocities[pid] = 0.0 
                continue
            
            idx = node_map[node]
            is_inlet = any(self.G[node][nbr]['type'] == 'plenum_in' for nbr in self.G.neighbors(node))
            radius = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * radius**2

            Q_net = 0
            for nbr in self.G.neighbors(node):
                if nbr in closed_valves:
                    continue
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

    def plot_all(self, aero_solid, aero_porous, porous_net, Cp, Cp_solid, P_nodes):
        print(f"-> Generating plots in {self.output_dir}...")
        self._plot_geometry_cp(aero_porous, porous_net, Cp, Cp_solid)

        # NEW: side-by-side external comparisons
        self._plot_flow_field_comparison(aero_solid, aero_porous, porous_net, P_nodes)
        self._plot_pressure_field_comparison(aero_solid, aero_porous, porous_net, P_nodes)

        # Keep your internal network plot (already good)
        self._plot_internal_flow(aero_porous, porous_net, P_nodes)


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

    def _plot_flow_field(self, aero, porous_net, P_nodes):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        xg, yg = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.6, 0.6, 100))
        u, v = aero.compute_velocity_field(xg, yg)
        mag = np.sqrt(u**2 + v**2)
        
        # Draw the external flow velocity contour
        ax.contourf(xg, yg, mag, 30, cmap='viridis')
        
        # Draw the solid airfoil body
        ax.fill(aero.X, aero.Y, 'k', zorder=2)
        
        # Overlay the porous network nodes
        pos = nx.get_node_attributes(porous_net.G, 'pos')
        b_nodes = [n for n in porous_net.G.nodes if porous_net.G.nodes[n]['type']=='boundary']
        if pos:
            # Removed the problematic 'zorder=3' argument here
            nx.draw_networkx_nodes(porous_net.G, pos, nodelist=b_nodes, ax=ax, 
                                   node_size=15, node_color='red', edgecolors='white')

        ax.axis('equal')
        ax.set_title("External Velocity Magnitude & Porous Boundaries")
        fig.savefig(os.path.join(self.output_dir, '05_Flow_Field.png'), dpi=150)
        plt.close(fig)

    def _plot_internal_flow(self, aero, net, P_nodes):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # 1. Plot Airfoil Boundary
        ax.plot(aero.X, aero.Y, 'k-', lw=1.5, zorder=1)
        ax.fill(aero.X, aero.Y, 'whitesmoke', zorder=0)
        
        pos = nx.get_node_attributes(net.G, 'pos')
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}
        
        # 2. Extract Data and Calculate Flow Directions
        edge_data = []
        max_vel = 1e-9  # Prevent division by zero
        
        for u, v, d in net.G.edges(data=True):
            idx_u, idx_v = node_map[u], node_map[v]
            P_u, P_v = P_nodes[idx_u], P_nodes[idx_v]
            
            # Volumetric flow rate Q = Conductance * dP
            c = d['cond']
            Q = c * (P_u - P_v)
            
            # Determine true direction of flow
            if Q >= 0:
                source, target = u, v
            else:
                source, target = v, u
                Q = abs(Q)
                
            # Estimate velocity for the edge
            is_inlet = d.get('type') == 'plenum_in'
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad**2
            vel = Q / area
            max_vel = max(max_vel, vel)
            
            edge_data.append({
                'source': source, 'target': target, 
                'Q': Q, 'vel': vel
            })

        # 3. Draw Pipes and Directional Arrows
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Setup colormap for velocity on the edges
        cmap_vel = cm.get_cmap('plasma')
        norm_vel = mcolors.Normalize(vmin=0, vmax=max_vel)

        for edge in edge_data:
            x_s, y_s = pos[edge['source']]
            x_t, y_t = pos[edge['target']]
            vel = edge['vel']
            color = cmap_vel(norm_vel(vel))
            
            # Draw the pipe
            ax.plot([x_s, x_t], [y_s, y_t], color=color, lw=2, alpha=0.7, zorder=2)
            
            # Draw flow direction arrow slightly offset from the midpoint
            mid_x, mid_y = x_s + 0.6 * (x_t - x_s), y_s + 0.6 * (y_t - y_s)
            dir_x, dir_y = x_t - x_s, y_t - y_s
            length = np.hypot(dir_x, dir_y)
            
            if length > 0 and vel > 1e-4:  # Only draw arrow if flow is non-negligible
                dx, dy = (dir_x / length) * 0.03, (dir_y / length) * 0.03
                ax.annotate('', xy=(mid_x + dx, mid_y + dy), xytext=(mid_x - dx, mid_y - dy),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, 
                                            mutation_scale=15), zorder=4)

        # 4. Draw Nodes Colored by Pressure
        p_values = [P_nodes[node_map[n]] for n in node_list]
        sc = ax.scatter([pos[n][0] for n in node_list],
                        [pos[n][1] for n in node_list],
                        c=p_values, cmap='viridis', s=60, zorder=5, edgecolors='black')
        
        # 5. Add Colorbars and Formatting
        cbar1 = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("Node Pressure [Pa]")
        
        sm = plt.cm.ScalarMappable(cmap=cmap_vel, norm=norm_vel)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label("Pipe Velocity [m/s]")

        ax.axis('equal')
        ax.set_title("Internal Porous Network: Pressure & Flow Direction")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, '06_Internal_Flow.png'), dpi=150)
        plt.close(fig)
    def _compute_directed_network_edges(self, net, P_nodes):
        """
        Returns a list of dicts:
        {x0,y0,x1,y1, vel, Q}
        direction is from high->low pressure (positive flow) based on conductance.
        """
        if P_nodes is None or len(net.G.nodes()) == 0:
            return []

        pos = nx.get_node_attributes(net.G, 'pos')
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        directed = []
        for u, v, d in net.G.edges(data=True):
            iu, iv = node_map[u], node_map[v]
            Pu, Pv = P_nodes[iu], P_nodes[iv]
            c = d.get('cond', 0.0)
            Q = c * (Pu - Pv)  # positive means u->v

            # Decide direction
            if Q >= 0:
                src, dst = u, v
                Qabs = Q
            else:
                src, dst = v, u
                Qabs = -Q

            # Estimate velocity using pore radius rule you already use
            is_inlet = d.get('type') == 'plenum_in'
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad**2
            vel = (Qabs / area) if area > 0 else 0.0

            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            directed.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, vel=vel, Q=Qabs))

        return directed
    def _plot_flow_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5),
            sharex=False, sharey=False,
            constrained_layout=True
        )

        xg, yg = np.meshgrid(np.linspace(-0.5, 1.5, 1000),
                            np.linspace(-0.6, 0.6, 1000))

        cases = [
            ("SOLID: |V| + Streamlines", aero_solid, False),
            ("POROUS: |V| + Streamlines + Network Dir", aero_porous, True),
        ]

        directed_edges = self._compute_directed_network_edges(net, P_nodes)

        cf = None
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            u, v = aero_case.compute_velocity_field(xg, yg)
            mag = np.sqrt(u**2 + v**2)

            cf = ax.contourf(xg, yg, mag, 40, cmap="viridis")
            ax.streamplot(xg, yg, u, v, density=1.6, linewidth=0.7, arrowsize=0.9)

            ax.fill(aero_case.X, aero_case.Y, "k", zorder=3)

            # boundary nodes
            pos = nx.get_node_attributes(net.G, "pos")
            b_nodes = [n for n in net.G.nodes if net.G.nodes[n].get("type") == "boundary"]
            if pos and len(b_nodes) > 0:
                nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax,
                                    node_size=14, node_color="red", edgecolors="white")

            # internal network overlay (porous only)
            if overlay_net and directed_edges:
                for e in directed_edges:
                    ax.plot([e["x0"], e["x1"]], [e["y0"], e["y1"]],
                            lw=2.0, alpha=0.9, zorder=4)

                for e in directed_edges:
                    dx = e["x1"] - e["x0"]
                    dy = e["y1"] - e["y0"]
                    L = np.hypot(dx, dy)
                    if L < 1e-12:
                        continue
                    mx = e["x0"] + 0.6 * dx
                    my = e["y0"] + 0.6 * dy
                    ax.annotate(
                        "",
                        xy=(mx + 0.03 * dx / L, my + 0.03 * dy / L),
                        xytext=(mx - 0.03 * dx / L, my - 0.03 * dy / L),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, mutation_scale=14),
                        zorder=5
                    )

            ax.set_title(title)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.6, 0.6)
            ax.set_aspect('equal', adjustable='box')

        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("Velocity Magnitude |V| [m/s]")

        fig.savefig(os.path.join(self.output_dir, "05_Compare_FlowField_Velocity.png"), dpi=150)
        plt.close(fig)

    def _plot_pressure_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5),
            sharex=False, sharey=False,
            constrained_layout=True
        )

        xg, yg = np.meshgrid(np.linspace(-0.5, 1.5, 1000),
                            np.linspace(-0.6, 0.6, 1000))

        cases = [
            ("SOLID: Pressure + Streamlines", aero_solid, False),
            ("POROUS: Pressure + Streamlines + Network Dir", aero_porous, True),
        ]

        directed_edges = self._compute_directed_network_edges(net, P_nodes)

        cf = None
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            u, v = aero_case.compute_velocity_field(xg, yg)
            mag = np.sqrt(u**2 + v**2)

            Cp_field = 1.0 - (mag / (self.cfg.V_INF + 1e-12))**2
            P_field = self.cfg.P_INF + 0.5 * self.cfg.RHO * (self.cfg.V_INF**2) * Cp_field

            cf = ax.contourf(xg, yg, P_field, 40, cmap="viridis")
            ax.streamplot(xg, yg, u, v, density=1.6, linewidth=0.7, arrowsize=0.9)

            ax.fill(aero_case.X, aero_case.Y, "k", zorder=3)

            pos = nx.get_node_attributes(net.G, "pos")
            b_nodes = [n for n in net.G.nodes if net.G.nodes[n].get("type") == "boundary"]
            if pos and len(b_nodes) > 0:
                nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax,
                                    node_size=14, node_color="red", edgecolors="white")

            if overlay_net and directed_edges:
                for e in directed_edges:
                    ax.plot([e["x0"], e["x1"]], [e["y0"], e["y1"]],
                            lw=2.0, alpha=0.9, zorder=4)

                for e in directed_edges:
                    dx = e["x1"] - e["x0"]
                    dy = e["y1"] - e["y0"]
                    L = np.hypot(dx, dy)
                    if L < 1e-12:
                        continue
                    mx = e["x0"] + 0.6 * dx
                    my = e["y0"] + 0.6 * dy
                    ax.annotate(
                        "",
                        xy=(mx + 0.03 * dx / L, my + 0.03 * dy / L),
                        xytext=(mx - 0.03 * dx / L, my - 0.03 * dy / L),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, mutation_scale=14),
                        zorder=5
                    )

            ax.set_title(title)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.6, 0.6)
            ax.set_aspect('equal', adjustable='box')

        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("Pressure [Pa]")

        fig.savefig(os.path.join(self.output_dir, "05b_Compare_FlowField_Pressure.png"), dpi=150)
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

    # Keep a snapshot object for solid external field plotting later
    aero_solid = PanelMethod(X, Y, cfg)
    _ = aero_solid.solve(np.zeros(aero_solid.N))  # sets q, gamma, etc for solid state


    
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
    #viz.plot_all(aero, net, Cp, Cp_solid, P_nodes)
    viz.plot_all(aero_solid, aero, net, Cp, Cp_solid, P_nodes)



if __name__ == "__main__":
    run_simulation()