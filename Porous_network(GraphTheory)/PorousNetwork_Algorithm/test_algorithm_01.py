import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import warnings
import os

# Suppress runtime warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Geometry & Mesh
AIRFOIL_NAME = "0018"
N_PANELS = 320       # Number of panels
N_PORES = 32         # Number of pores on surface

# Porous Region
X_START = 0.01     
X_END = 0.99       
PORE_RADIUS = 2000e-6   # 2mm

# Physics
REYNOLDS_NUM = 250000   
ANGLE_OF_ATTACK = 6.0
RHO = 1.225             
MU = 1.78e-5            
P_INF = 0.0
CHORD = 1.0             

# Velocity (calculated once based on Re)
V_INF = (REYNOLDS_NUM * MU) / (RHO * CHORD)

# Solver
MAX_ITER = 100
RELAXATION = 0.01
CONVERGENCE_TOL = 1e-8

# ==============================================================================
# 2. GEOMETRY GENERATION
# ==============================================================================
def naca4(number, n_panels=160):
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0

    # Cosine clustering
    beta = np.linspace(0, np.pi, n_panels // 2 + 1)
    x = (1 - np.cos(beta)) / 2

    # Thickness distribution (Modified coeff 0.1036 for closed TE)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    
    # Camber lines (Symmetric 00xx -> yc=0)
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

    # Force TE closed
    xu[-1] = 1.0; yu[-1] = 0.0
    xl[-1] = 1.0; yl[-1] = 0.0

    # Concatenate: Trailing (Top) -> Leading -> Trailing (Bottom)
    X = np.concatenate((xu[::-1], xl[1:]))
    Y = np.concatenate((yu[::-1], yl[1:]))

    return X, Y

# ==============================================================================
# 3. AERODYNAMIC SOLVER (Panel Method)
# ==============================================================================
class PanelMethod:
    def __init__(self, X, Y, alpha_deg):
        self.X, self.Y = X, Y
        self.alpha = np.radians(alpha_deg)
        self.N = len(X) - 1

        # Panel Geometry
        self.XC = (X[:-1] + X[1:]) / 2
        self.YC = (Y[:-1] + Y[1:]) / 2
        self.dx = X[1:] - X[:-1]
        self.dy = Y[1:] - Y[:-1]
        self.L = np.sqrt(self.dx**2 + self.dy**2)

        # Normals and Tangents
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
                    self.Is_t[i, j] = 0.0
                    self.Iv_n[i, j] = 0.0
                    self.Iv_t[i, j] = 0.5 * np.pi
                    continue

                dx = self.XC[i] - self.X[j]
                dy = self.YC[i] - self.Y[j]
                x_local =  dx * self.tx[j] + dy * self.ty[j]
                y_local = -dx * self.ty[j] + dy * self.tx[j]

                r1_sq = x_local**2 + y_local**2
                r2_sq = (x_local - self.L[j])**2 + y_local**2
                
                theta1 = np.arctan2(y_local, x_local)
                theta2 = np.arctan2(y_local, x_local - self.L[j])
                dtheta = theta2 - theta1

                if dtheta > np.pi: dtheta -= 2*np.pi
                elif dtheta < -np.pi: dtheta += 2*np.pi

                us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
                vs_loc =  1.0 / np.pi * dtheta
                uv_loc, vv_loc = -vs_loc, us_loc

                us_glob = us_loc * self.tx[j] - vs_loc * self.ty[j]
                vs_glob = us_loc * self.ty[j] + vs_loc * self.tx[j]
                uv_glob = uv_loc * self.tx[j] - vv_loc * self.ty[j]
                vv_glob = uv_loc * self.ty[j] + vv_loc * self.tx[j]

                self.Is_n[i, j] = us_glob * self.nx[i] + vs_glob * self.ny[i]
                self.Is_t[i, j] = us_glob * self.tx[i] + vs_glob * self.ty[i]
                self.Iv_n[i, j] = uv_glob * self.nx[i] + vv_glob * self.ny[i]
                self.Iv_t[i, j] = uv_glob * self.tx[i] + vv_glob * self.ty[i]

    def solve(self, V_leakage=None):
        if V_leakage is None: V_leakage = np.zeros(self.N)
        
        Vinf_x = V_INF * np.cos(self.alpha)
        Vinf_y = V_INF * np.sin(self.alpha)
        Vinf_n = Vinf_x * self.nx + Vinf_y * self.ny
        Vinf_t = Vinf_x * self.tx + Vinf_y * self.ty

        A = np.zeros((self.N + 1, self.N + 1))
        b = np.zeros(self.N + 1)

        # Flow Tangency
        A[:self.N, :self.N] = self.Is_n
        A[:self.N, self.N] = np.sum(self.Iv_n, axis=1) 
        b[:self.N] = V_leakage - Vinf_n

        # Kutta Condition
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
        Cp = 1.0 - (Vt / V_INF)**2
        return Cp

    def compute_velocity_field(self, X_grid, Y_grid):
        u = np.zeros_like(X_grid)
        v = np.zeros_like(Y_grid)
        u += V_INF * np.cos(self.alpha)
        v += V_INF * np.sin(self.alpha)

        for j in range(self.N):
            dx = X_grid - self.X[j]
            dy = Y_grid - self.Y[j]
            x_loc =  dx * self.tx[j] + dy * self.ty[j]
            y_loc = -dx * self.ty[j] + dy * self.tx[j]
            
            r1_sq = x_loc**2 + y_loc**2
            r2_sq = (x_loc - self.L[j])**2 + y_loc**2
            
            theta1 = np.arctan2(y_loc, x_loc)
            theta2 = np.arctan2(y_loc, x_loc - self.L[j])
            dtheta = theta2 - theta1
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            
            us_loc = -0.5 / np.pi * np.log(r2_sq / (r1_sq + 1e-12))
            vs_loc =  1.0 / np.pi * dtheta
            uv_loc, vv_loc = -vs_loc, us_loc
            
            u_ind = (us_loc * self.q[j] + uv_loc * self.gamma) * self.tx[j] - \
                    (vs_loc * self.q[j] + vv_loc * self.gamma) * self.ty[j]
            v_ind = (us_loc * self.q[j] + uv_loc * self.gamma) * self.ty[j] + \
                    (vs_loc * self.q[j] + vv_loc * self.gamma) * self.tx[j]
            u += u_ind
            v += v_ind
        return u, v

# ==============================================================================
# 4. STRUCTURED POROUS MESH GENERATION
# ==============================================================================
def generate_tangential_mesh(xc, yc, tx, ty, pore_radius, mu):
    """
    Generates a High-Efficiency network connecting Mid-Chord Pressure side
    to Aft-Chord Suction side using Tangential Injection logic.
    """
    G = nx.Graph()
    porous_indices = []
    
    # --- 1. Define Zones (The "Smart" Selection) ---
    # Inlet: Bottom surface (Pressure), Mid-Chord
    inlet_candidates = []
    for i in range(len(xc)):
        # Check if on Bottom (yc < 0) and within 40-70% Chord
        if yc[i] < 0 and 0.40 <= xc[i] <= 0.70:
            inlet_candidates.append(i)

    # Outlet: Top surface (Suction), Aft-Chord
    outlet_candidates = []
    for i in range(len(xc)):
        # Check if on Top (yc > 0) and within 75-95% Chord
        if yc[i] > 0 and 0.75 <= xc[i] <= 0.95:
            outlet_candidates.append(i)

    # Consolidate active indices for the solver
    porous_indices = list(set(inlet_candidates + outlet_candidates))
    
    # Add nodes to Graph
    for idx in porous_indices:
        G.add_node(idx, pos=(xc[idx], yc[idx]), type='boundary', panel_idx=idx)

    # --- 2. Vector-Based Connection Logic ---
    # We brute-force check every Inlet-Outlet pair and filter by angle.
    
    connections_made = 0
    
    for i in inlet_candidates:
        p_in = np.array([xc[i], yc[i]])
        
        for j in outlet_candidates:
            p_out = np.array([xc[j], yc[j]])
            
            # Vector of the pore (Internal Flow Direction)
            vec_pore = p_out - p_in
            len_pore = np.linalg.norm(vec_pore)
            dir_pore = vec_pore / len_pore
            
            # Vector of the surface flow at outlet (External Flow Direction)
            # Note: In panel methods, 'tx, ty' usually point CW. 
            # On top surface, CW is Trailing->Leading (Backwards).
            # So the flow direction is -tx, -ty.
            vec_surf = np.array([-tx[j], -ty[j]]) 
            
            # Calculate Alignment (Dot Product)
            # cos(theta) = dot(a, b)
            dot_prod = np.dot(dir_pore, vec_surf)
            
            # Clamp for numerical safety
            dot_prod = max(min(dot_prod, 1.0), -1.0)
            angle_deg = np.degrees(np.arccos(dot_prod))
            
            # --- 3. The Filter (The "Smooth Merge" Check) ---
            # We strictly enforce shallow angles (Tangential Injection).
            # We also ensure the pore isn't absurdly long (optional check).
            if angle_deg < 25.0: 
                # Calculate Conductance (Hagen-Poiseuille)
                cond = (np.pi * pore_radius**4) / (8 * mu * len_pore)
                
                G.add_edge(i, j, length=len_pore, cond=cond, type='standard')
                connections_made += 1

    # Remove isolated nodes (panels selected in zones but found no valid partner)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    
    # Update porous_indices to only include connected nodes
    final_indices = [n for n in G.nodes()]
    
    print(f"   -> Tangential Logic: Created {connections_made} connections.")
    
    return G, final_indices

# ==============================================================================
# 5. INTERNAL FLOW SOLVER
# ==============================================================================
def solve_internal_flow(G, P_boundary):
    nodes = list(G.nodes())
    n = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    A = sp.lil_matrix((n, n))
    b = np.zeros(n)

    boundary_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'boundary']

    for node in nodes:
        idx = node_map[node]
        if node in boundary_nodes:
            pid = G.nodes[node]['panel_idx']
            if pid in P_boundary:
                A[idx, idx] = 1.0
                b[idx] = P_boundary[pid]
            else:
                A[idx, idx] = 1.0
                b[idx] = 0.0
        else:
            sigma_cond = 0.0
            for nbr in G.neighbors(node):
                c = G[node][nbr]['cond']
                nbr_idx = node_map[nbr]
                A[idx, nbr_idx] = -c
                sigma_cond += c
            A[idx, idx] = sigma_cond

    try:
        P_nodes = scipy.sparse.linalg.spsolve(A.tocsr(), b)
    except:
        return {}, np.zeros(n)

    velocities = {}
    Pore_Area = np.pi * PORE_RADIUS**2

    for node in boundary_nodes:
        pid = G.nodes[node]['panel_idx']
        idx = node_map[node]
        Q_net = 0
        for nbr in G.neighbors(node):
            c = G[node][nbr]['cond']
            nbr_idx = node_map[nbr]
            Q_net += c * (P_nodes[idx] - P_nodes[nbr_idx])
        velocities[pid] = -Q_net / Pore_Area

    return velocities, P_nodes

def plot_results(aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid, G, output_dir="porous_airfoil_results"):
    
    # --- CHANGED: DETERMINE PATH RELATIVE TO SCRIPT LOCATION ---
    try:
        # If running as a script, use the script's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # If running in Jupyter/Interactive, fallback to CWD
        base_dir = os.getcwd()

    full_output_dir = os.path.join(base_dir, output_dir)

    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        print(f"-> Created output directory: {full_output_dir}")
    else:
        print(f"-> Saving images to existing directory: {full_output_dir}")

    # ==========================================================================
    # SAVE RESULTS TO CSV
    # ==========================================================================
    csv_path = os.path.join(output_dir, 'simulation_data.csv')
    print(f"-> Exporting numerical data to: {csv_path}")
    
    try:
        with open(csv_path, 'w') as f:
            # 1. Write Global Aerodynamic Coefficients
            f.write("--- GLOBAL RESULTS ---\n")
            f.write("Metric,Solid_Baseline,Porous_Result,Change_Percent\n")
            
            cl_change = ((CL - CL_solid) / (abs(CL_solid) + 1e-9)) * 100
            cd_change = ((CD - CD_solid) / (abs(CD_solid) + 1e-9)) * 100
            
            f.write(f"CL (Lift Coeff),{CL_solid:.6f},{CL:.6f},{cl_change:.2f}%\n")
            f.write(f"CD (Drag Coeff),{CD_solid:.6f},{CD:.6f},{cd_change:.2f}%\n")
            f.write("\n") # Empty line for separation
            
            # 2. Write Detailed Panel Distribution Data
            f.write("--- PANEL DISTRIBUTION DATA ---\n")
            # Header
            f.write("Panel_ID,XC,YC,Nx,Ny,Panel_Length,Cp_Solid,Cp_Porous,Cp_Difference,V_leakage\n")
            
            # Data Rows
            for i in range(aero.N):
                cp_diff = Cp[i] - Cp_solid[i]
                line = (f"{i},"
                        f"{aero.XC[i]:.6f},{aero.YC[i]:.6f},"
                        f"{aero.nx[i]:.6f},{aero.ny[i]:.6f},"
                        f"{aero.L[i]:.6f},"
                        f"{Cp_solid[i]:.6f},{Cp[i]:.6f},"
                        f"{cp_diff:.6f},{V_leakage[i]:.6f}\n")
                f.write(line)
                
    except IOError as e:
        print(f"Error saving CSV file: {e}")

    # ==========================================================================
    # PLOTTING LOGIC
    # ==========================================================================

    # --- FIGURE 1: Original Geometry & Network ---
    fig1 = plt.figure(figsize=(12, 12))
    gs1 = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])

    ax1 = fig1.add_subplot(gs1[0])
    ax1.plot(aero.X, aero.Y, 'k-', linewidth=2, label="Airfoil Surface")
    ax1.fill(aero.X, aero.Y, 'whitesmoke')

    pos = nx.get_node_attributes(G, 'pos')
    std_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'standard']
    rescue_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'rescue']
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=std_edges, edge_color='cyan', alpha=0.6, width=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=rescue_edges, edge_color='blue', alpha=0.8, width=1.5, style=':')

    boundary_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'boundary']
    internal_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'internal']
    nx.draw_networkx_nodes(G, pos, nodelist=boundary_nodes, ax=ax1, node_size=30, node_color='black', label='Surface Pores')
    nx.draw_networkx_nodes(G, pos, nodelist=internal_nodes, ax=ax1, node_size=60, node_color='orange', edgecolors='black', label='Internal Nodes')

    ax1.set_title("1. Geometry & Porous Network Structure", fontsize=14)
    ax1.set_xlabel("x/c"); ax1.set_ylabel("y/c"); ax1.axis('equal'); ax1.legend(loc='lower right')

    # Subplot 2: Cp
    ax2 = fig1.add_subplot(gs1[1])
    ax2.plot(aero.XC, Cp_solid, 'k--', label=f'Solid Wall ($C_L = {CL_solid:.3f}$)')
    ax2.plot(aero.XC, Cp, 'b-', label=f'Porous ($C_L = {CL:.3f}$)')
    ax2.invert_yaxis()
    ax2.set_title("2. Pressure Coefficient ($C_p$)", fontsize=14)
    ax2.set_xlabel("x/c"); ax2.set_ylabel("Cp"); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    txt = f"CL Solid: {CL_solid:.4f}\nCL Porous: {CL:.4f}\nCL Change: {((CL-CL_solid)/(abs(CL_solid)+1e-9))*100:.1f}%"
    ax2.text(0.02, 0.05, txt, transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, '01_Geometry_and_Cp.png'), dpi=300, bbox_inches='tight')

    # --- FIGURE 3: Pressure Vectors ---
    fig3 = plt.figure(figsize=(10, 6))
    ax5 = fig3.add_subplot(111)
    
    ax5.plot(aero.X, aero.Y, 'k-', linewidth=2, label='Airfoil')
    ax5.fill(aero.X, aero.Y, 'whitesmoke')
    
    U_p = -Cp * aero.nx * 0.15
    V_p = -Cp * aero.ny * 0.15
    q = ax5.quiver(aero.XC, aero.YC, U_p, V_p, Cp, cmap='coolwarm_r', scale=1, scale_units='xy', width=0.004)
    plt.colorbar(q, ax=ax5, label='Cp')
    
    suction_idx = [i for i in range(aero.N) if V_leakage[i] < -1e-4]
    blowing_idx = [i for i in range(aero.N) if V_leakage[i] > 1e-4]
    
    if suction_idx:
        ax5.scatter(aero.XC[suction_idx], aero.YC[suction_idx], color='green', marker='v', s=60, zorder=5, label='Inflow')
    if blowing_idx:
        ax5.scatter(aero.XC[blowing_idx], aero.YC[blowing_idx], color='magenta', marker='^', s=60, zorder=5, label='Outflow')

    ax5.set_title("3. Surface Pressure Vectors & Flow Direction", fontsize=14)
    ax5.axis('equal'); ax5.grid(True, linestyle=':', alpha=0.6); ax5.legend(loc='lower right')
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, '03_Pressure_Vectors.png'), dpi=300, bbox_inches='tight')

    # --- FIGURE 4: Lift Loss & Flow Map (UPDATED: Removed Breathing Profile) ---
    fig4 = plt.figure(figsize=(10, 8)) # Reduced height slightly
    gs4 = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Lift Loss Visualization
    ax6 = fig4.add_subplot(gs4[0])
    ax6.plot(aero.XC, Cp_solid, 'k--', linewidth=1, label='Solid')
    ax6.plot(aero.XC, Cp, 'b-', linewidth=2, label='Porous')
    ax6.fill_between(aero.XC, Cp_solid, Cp, color='red', alpha=0.15, hatch='///', label='Lift Loss')
    ax6.invert_yaxis()
    ax6.set_title("4. Lift Loss Visualization", fontsize=14)
    ax6.set_ylabel("Cp"); ax6.legend(); ax6.grid(True, alpha=0.3)

    # Map of Inflow/Outflow Zones
    ax7 = fig4.add_subplot(gs4[1])
    ax7.plot(aero.X, aero.Y, 'k-', linewidth=2)
    ax7.fill(aero.X, aero.Y, 'whitesmoke')
    if suction_idx:
        ax7.scatter(aero.XC[suction_idx], aero.YC[suction_idx], color='green', marker='v', s=80, label='Inflow')
    if blowing_idx:
        ax7.scatter(aero.XC[blowing_idx], aero.YC[blowing_idx], color='magenta', marker='^', s=80, label='Outflow')
    ax7.set_title("Map of Inflow/Outflow Zones", fontsize=14)
    ax7.axis('equal'); ax7.grid(True, linestyle=':', alpha=0.6); ax7.legend(loc='lower right')
    
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, '04_Lift_Loss_Map.png'), dpi=300, bbox_inches='tight')

    # --- FIGURE 5: Simplified Flow Visualization ---
    print("-> Calculating Streamlines for visualization...")
    
    # Setup Globals safely
    v_inf_local = globals().get('V_INF', 1.0)
    reynolds_local = globals().get('REYNOLDS_NUM', 100000)

    # External Flow Field
    x_min, x_max = -0.5, 1.5; y_min, y_max = -0.6, 0.6
    Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Ug, Vg = aero.compute_velocity_field(Xg, Yg)
    vel_mag = np.sqrt(Ug**2 + Vg**2)

    fig5 = plt.figure(figsize=(12, 7))
    ax9 = fig5.add_subplot(111)
    
    # Background Contour
    levels = np.linspace(0, 1.5*v_inf_local, 50)
    cf = ax9.contourf(Xg, Yg, vel_mag, levels=levels, cmap='viridis', extend='both', alpha=0.9)
    plt.colorbar(cf, ax=ax9, label='Velocity Magnitude [m/s]', pad=0.02)
    
    # Streamlines
    seed_points = np.column_stack((np.ones(40) * x_min, np.linspace(y_min, y_max, 40)))
    ax9.streamplot(Xg, Yg, Ug, Vg, color='white', linewidth=0.8, arrowsize=0.8, density=2, start_points=seed_points)
    
    # Airfoil Geometry (Black fill)
    ax9.fill(aero.X, aero.Y, color='black', zorder=3)

    # Internal Network Structure (Faint lines for context)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edges(G, pos, ax=ax9, edge_color='gray', alpha=0.3, width=0.8)

    # Highlight Inflow/Outflow
    suction_mask = V_leakage < -1e-4
    blowing_mask = V_leakage > 1e-4
    scale_arrow = 0.08
    
    if np.any(suction_mask):
        ax9.quiver(aero.XC[suction_mask], aero.YC[suction_mask], 
                   aero.nx[suction_mask]*V_leakage[suction_mask]*scale_arrow, 
                   aero.ny[suction_mask]*V_leakage[suction_mask]*scale_arrow, 
                   color='lime', scale=1, scale_units='xy', width=0.006, zorder=6, label='Inflow (Suction)')
                   
    if np.any(blowing_mask):
        ax9.quiver(aero.XC[blowing_mask], aero.YC[blowing_mask], 
                   aero.nx[blowing_mask]*V_leakage[blowing_mask]*scale_arrow, 
                   aero.ny[blowing_mask]*V_leakage[blowing_mask]*scale_arrow, 
                   color='magenta', scale=1, scale_units='xy', width=0.006, zorder=6, label='Outflow (Blowing)')

    ax9.set_title(f"5. Flow Field & Surface Pore Activity (Re={int(reynolds_local)})", fontsize=14, weight='bold')
    ax9.set_xlim(x_min, x_max); ax9.set_ylim(y_min, y_max); ax9.set_aspect('equal')
    
    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='lime', lw=2),
                    Line2D([0], [0], color='magenta', lw=2),
                    Line2D([0], [0], color='gray', lw=1, alpha=0.5)]
    ax9.legend(custom_lines, ['Inflow', 'Outflow', 'Porous Network'], loc='upper right')

    plt.tight_layout()
    fig5.savefig(os.path.join(output_dir, '05_Flow_Field.png'), dpi=300, bbox_inches='tight')

    plt.show()
    print(f"-> All figures and data saved to: {os.path.abspath(output_dir)}")

# ==============================================================================
# 7. MAIN LOOP
# ==============================================================================
def run_simulation():
    print(f"--- SIMULATION START: NACA {AIRFOIL_NAME} @ {ANGLE_OF_ATTACK} deg ---")
    print(f"--- Re={REYNOLDS_NUM}, V_inf={V_INF:.2f} m/s ---")

    X, Y = naca4(AIRFOIL_NAME, n_panels=N_PANELS)
    aero = PanelMethod(X, Y, ANGLE_OF_ATTACK)

    print("-> Solving Baseline (Solid)...")
    Cp_solid = aero.solve(np.zeros(aero.N))

    # [Force Calculation Code Omitted for Brevity - keep your existing lines here]
    fx_elem = -Cp_solid * aero.nx * aero.L
    fy_elem = -Cp_solid * aero.ny * aero.L
    Fx_solid = np.sum(fx_elem)
    Fy_solid = np.sum(fy_elem)
    CL_solid = Fy_solid * np.cos(aero.alpha) - Fx_solid * np.sin(aero.alpha)
    CD_solid = Fx_solid * np.cos(aero.alpha) + Fy_solid * np.sin(aero.alpha)
    print(f"   Baseline CL: {CL_solid:.4f}")

    # --- MODIFIED SECTION START ---
    print("-> Generating High-Efficiency Tangential Mesh...")
    
    # CALL THE NEW FUNCTION HERE
    # We pass tx, ty to calculate injection angles
    G, porous_pids = generate_tangential_mesh(
        aero.XC, aero.YC, aero.tx, aero.ty, 
        PORE_RADIUS, MU
    )
    
    print(f"   Generated network with {len(G.nodes())} nodes.")
    # --- MODIFIED SECTION END ---

    # [Solver Loop - keep your existing lines here]
    V_leakage = np.zeros(aero.N)
    for i in range(MAX_ITER):
        Cp = aero.solve(V_leakage)
        q_inf = 0.5 * RHO * V_INF**2
        P_ext = P_INF + q_inf * Cp
        P_map = {pid: P_ext[pid] for pid in porous_pids}
        V_calculated, _ = solve_internal_flow(G, P_map)

        max_diff = 0.0
        V_new = V_leakage.copy()
        for pid, v_calc in V_calculated.items():
            v_relaxed = RELAXATION * v_calc + (1 - RELAXATION) * V_leakage[pid]
            # Tweak: Increase limiter slightly for tangential jets
            v_relaxed = max(min(v_relaxed, 80.0), -80.0) 
            diff = abs(v_relaxed - V_leakage[pid])
            if diff > max_diff: max_diff = diff
            V_new[pid] = v_relaxed
        V_leakage = V_new

        if max_diff < CONVERGENCE_TOL and i > 5:
            print(f"-> Converged at Iter {i}")
            break
        elif i % 10 == 0:
             print(f"   Iter {i}: Max Resid = {max_diff:.6f}")

    # [Final Calculation Code - keep your existing lines here]
    fx_elem = -Cp * aero.nx * aero.L
    fy_elem = -Cp * aero.ny * aero.L
    Fx_porous = np.sum(fx_elem)
    Fy_porous = np.sum(fy_elem)
    CL = Fy_porous * np.cos(aero.alpha) - Fx_porous * np.sin(aero.alpha)
    CD = Fx_porous * np.cos(aero.alpha) + Fy_porous * np.sin(aero.alpha)

    print(f"-> Final Results: Solid CL={CL_solid:.4f}, Porous CL={CL:.4f}")
    plot_results(aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid, G)

if __name__ == "__main__":
    run_simulation()
    