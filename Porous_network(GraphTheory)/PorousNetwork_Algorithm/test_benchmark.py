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
from scipy.interpolate import griddata
import matplotlib.path as mpath

# Suppress runtime warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Geometry & Mesh
AIRFOIL_NAME = "0018"
N_PANELS = 320       # Number of panels
N_PORES = 50         # Number of pores on surface

# Porous Region
X_START = 0.01     
X_END = 0.99       
PORE_RADIUS = 5000e-6   # 2mm

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
def generate_tangential_mesh(xc, yc, tx, ty, cp_solid, n_target, pore_radius, mu):
    """
    Generates a 'Cross-Flow Plenum' network.
    
    Correction:
      - Strictly enforces Outlet location at x > 0.85 on Top Surface.
      - Uses Plenum model to decouple inlet/outlet counts.
    """
    import networkx as nx
    G = nx.Graph()
    
    plenum_id = 99999 
    plenum_pos = np.array([0.5, 0.0])
    G.add_node(plenum_id, pos=plenum_pos, type='internal')
    
    # --- Configuration ---
    N_INLETS = 40
    R_INLET  = 3000e-6
    
    N_OUTLETS = 15
    R_OUTLET = 4000e-6
    
    # 1. Define Zones (Strict Filtering)
    # Outlet: Top Surface (y > 0) AND Aft (x > 0.85)
    # sorting by x descending ensures we get the very trailing edge
    outlet_candidates = [i for i in range(len(xc)) if yc[i] > 0 and xc[i] >= 0.85]
    
    # Inlet: Bottom Surface (y < 0) AND Forward (0.02 < x < 0.20)
    inlet_candidates = [i for i in range(len(xc)) if yc[i] < 0 and 0.02 <= xc[i] <= 0.20]
    
    # 2. Select Inlets (Best Cp - Highest Pressure)
    inlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in inlet_candidates]
    inlet_scores.sort(key=lambda x: x['cp'], reverse=True)
    selected_inlets = [x['id'] for x in inlet_scores[:N_INLETS]]
    
    # 3. Select Outlets (Rearmost points)
    outlet_scores = [{'id': i, 'x': xc[i]} for i in outlet_candidates]
    outlet_scores.sort(key=lambda x: x['x'], reverse=True) # Max x = TE
    selected_outlets = [x['id'] for x in outlet_scores[:N_OUTLETS]]
    
    # 4. Connect
    for u in selected_inlets:
        length = np.linalg.norm(np.array([xc[u], yc[u]]) - plenum_pos)
        cond = (np.pi * R_INLET**4) / (8 * mu * length)
        if u not in G: G.add_node(u, pos=(xc[u], yc[u]), type='boundary', panel_idx=u)
        G.add_edge(u, plenum_id, length=length, cond=cond, type='plenum_in')
        
    for v in selected_outlets:
        length = np.linalg.norm(np.array([xc[v], yc[v]]) - plenum_pos)
        cond = (np.pi * R_OUTLET**4) / (8 * mu * length)
        if v not in G: G.add_node(v, pos=(xc[v], yc[v]), type='boundary', panel_idx=v)
        G.add_edge(plenum_id, v, length=length, cond=cond, type='plenum_out')
        
    print(f"   -> Cross-Flow Plenum: {len(selected_inlets)} Inlets -> {len(selected_outlets)} Outlets (TE).")
    
    return G, selected_inlets + selected_outlets
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

# ==============================================================================
# 6. PLOTTING RESULTS (COMPLETE: CSV + CONTOUR)
# ==============================================================================
def plot_results(aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid, G, P_nodes, output_dir="porous_airfoil_results"):
    
    # 1. Determine Output Path
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    full_output_dir = os.path.join(base_dir, output_dir)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        print(f"-> Created output directory: {full_output_dir}")
    else:
        print(f"-> Saving results to: {full_output_dir}")

    # --- 2. SAVE CSV DATA (RESTORED) ---
    csv_path = os.path.join(full_output_dir, 'simulation_data.csv')
    try:
        with open(csv_path, 'w') as f:
            f.write("--- GLOBAL RESULTS ---\n")
            f.write("Metric,Solid_Baseline,Porous_Result,Change_Percent\n")
            # Handle potential division by zero if CL/CD are 0
            cl_change = ((CL - CL_solid) / (abs(CL_solid) + 1e-9)) * 100
            cd_change = ((CD - CD_solid) / (abs(CD_solid) + 1e-9)) * 100
            f.write(f"CL,{CL_solid:.6f},{CL:.6f},{cl_change:.2f}%\n")
            f.write(f"CD,{CD_solid:.6f},{CD:.6f},{cd_change:.2f}%\n\n")
            
            f.write("--- PANEL DISTRIBUTION DATA ---\n")
            f.write("Panel_ID,XC,YC,Cp_Solid,Cp_Porous,V_leakage\n")
            for i in range(aero.N):
                f.write(f"{i},{aero.XC[i]:.6f},{aero.YC[i]:.6f},{Cp_solid[i]:.6f},{Cp[i]:.6f},{V_leakage[i]:.6f}\n")
        print(f"-> CSV Data saved successfully.")
    except Exception as e:
        print(f"ERROR saving CSV: {e}")

    # --- 3. PLOTTING ---
    # Use Agg backend to prevent GUI crashes
    import matplotlib
    matplotlib.use('Agg') 
    
    # --- FIGURE 1: Geometry & Cp ---
    print("-> Generating Figure 1...")
    fig1 = plt.figure(figsize=(12, 12))
    gs1 = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])

    ax1 = fig1.add_subplot(gs1[0])
    ax1.plot(aero.X, aero.Y, 'k-', linewidth=2, label="Airfoil")
    ax1.fill(aero.X, aero.Y, 'whitesmoke')

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='cyan', alpha=0.6, width=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), ax=ax1, node_size=30, node_color='black')
    ax1.set_title("1. Geometry & Porous Network", fontsize=14); ax1.axis('equal')

    ax2 = fig1.add_subplot(gs1[1])
    ax2.plot(aero.XC, Cp_solid, 'k--', label=f'Solid ($C_L = {CL_solid:.3f}$)')
    ax2.plot(aero.XC, Cp, 'b-', label=f'Porous ($C_L = {CL:.3f}$)')
    ax2.invert_yaxis(); ax2.grid(True, alpha=0.3); ax2.legend()
    ax2.set_title("2. Pressure Coefficient ($C_p$)", fontsize=14)
    
    fig1.savefig(os.path.join(full_output_dir, '01_Geometry_and_Cp.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- FIGURE 3: Surface Vectors ---
    print("-> Generating Figure 3...")
    fig3 = plt.figure(figsize=(10, 6))
    ax5 = fig3.add_subplot(111)
    ax5.plot(aero.X, aero.Y, 'k-', linewidth=2)
    ax5.fill(aero.X, aero.Y, 'whitesmoke')
    
    U_p = -Cp * aero.nx * 0.15
    V_p = -Cp * aero.ny * 0.15
    ax5.quiver(aero.XC, aero.YC, U_p, V_p, Cp, cmap='coolwarm_r', scale=1, scale_units='xy', width=0.004)
    ax5.set_title("3. Surface Pressure & Leakage", fontsize=14); ax5.axis('equal')
    
    fig3.savefig(os.path.join(full_output_dir, '03_Pressure_Vectors.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # --- FIGURE 5: External Flow Field ---
    print("-> Generating Figure 5...")
    x_min, x_max = -0.5, 1.5; y_min, y_max = -0.6, 0.6
    Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Ug, Vg = aero.compute_velocity_field(Xg, Yg)
    vel_mag = np.sqrt(Ug**2 + Vg**2)

    fig5 = plt.figure(figsize=(12, 7))
    ax9 = fig5.add_subplot(111)
    ax9.contourf(Xg, Yg, vel_mag, levels=50, cmap='viridis', extend='both', alpha=0.9)
    if np.max(vel_mag) > 1e-6:
        seed_points = np.column_stack((np.ones(40) * x_min, np.linspace(y_min, y_max, 40)))
        ax9.streamplot(Xg, Yg, Ug, Vg, color='white', linewidth=0.8, density=2, start_points=seed_points)
    ax9.fill(aero.X, aero.Y, color='black', zorder=3)
    ax9.set_title("5. External Flow Field", fontsize=14); ax9.axis('equal')
    
    fig5.savefig(os.path.join(full_output_dir, '05_Flow_Field.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # --- FIGURE 6: INTERNAL FLOW CONTOUR (NEW) ---
    print("-> Generating Figure 6 (Internal Flow Contour)...")
    
    # 1. Gather Data Points from the Graph Edges
    points = []
    values = []
    
    # Map nodes to indices for Pressure lookup
    node_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    
    max_v_found = 0.0

    for u, v, data in G.edges(data=True):
        idx_u, idx_v = node_map[u], node_map[v]
        pos_u, pos_v = np.array(pos[u]), np.array(pos[v])
        
        Pu, Pv = P_nodes[idx_u], P_nodes[idx_v]
        delta_P = abs(Pu - Pv)
        length = np.linalg.norm(pos_v - pos_u)
        
        # Velocity Magnitude
        velocity_mag = (PORE_RADIUS**2 * delta_P) / (8 * MU * length)
        if velocity_mag > max_v_found: max_v_found = velocity_mag

        # We sample points along the edge to "fill" the space for the contour
        # Interpolate 5 points along every pipe
        num_samples = 5
        for t in np.linspace(0, 1, num_samples):
            pt = pos_u + t * (pos_v - pos_u)
            points.append(pt)
            values.append(velocity_mag)

    points = np.array(points)
    values = np.array(values)

    fig6 = plt.figure(figsize=(14, 8))
    ax_int = fig6.add_subplot(111)

    # Draw Airfoil Shape
    ax_int.plot(aero.X, aero.Y, 'k-', linewidth=3, color='#333333', zorder=10)

    if len(points) > 0:
        # 2. Create Grid for Contour
        # Define grid bounds based on airfoil
        min_x, max_x = min(aero.X), max(aero.X)
        min_y, max_y = min(aero.Y), max(aero.Y)
        
        grid_x, grid_y = np.mgrid[min_x:max_x:200j, min_y:max_y:200j]
        
        # 3. Interpolate discrete pipe data onto the grid
        # 'linear' works best here to connect the pipes visually
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)

        # 4. Mask the grid (keep only points INSIDE the airfoil)
        # Create a path from the airfoil coordinates
        airfoil_poly = np.column_stack((aero.X, aero.Y))
        path = mpath.Path(airfoil_poly)
        
        # Flatten the grid to check points
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        mask = path.contains_points(grid_points)
        mask = mask.reshape(grid_x.shape)
        
        # Apply mask (NaN out points outside)
        grid_z[~mask] = np.nan

        # 5. Plot the Contour
        # Use 100 levels for smoothness
        contour = ax_int.contourf(grid_x, grid_y, grid_z, levels=100, cmap='plasma', zorder=1)
        
        # Add Colorbar
        cbar = plt.colorbar(contour, ax=ax_int, pad=0.02)
        cbar.set_label('Internal Pore Velocity [m/s]', fontsize=12)

    # Overlay the network faintly so we see the structure
    nx.draw_networkx_edges(G, pos, ax=ax_int, edge_color='white', alpha=0.3, width=0.5)

    ax_int.set_title(f"6. Internal Flow Distribution\nMax Velocity: {max_v_found:.2f} m/s", fontsize=16)
    ax_int.set_xlabel("x/c"); ax_int.set_ylabel("y/c")
    ax_int.axis('equal')
    ax_int.set_xlim(0.0, 1.05)
    
    fig6.savefig(os.path.join(full_output_dir, '06_Internal_Flow_Map_Contour.png'), dpi=300, bbox_inches='tight')
    plt.close(fig6)

    print(f"-> All figures saved to: {full_output_dir}")

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

    # --- 1. Calculate SOLID Forces ---
    fx_elem = -Cp_solid * aero.nx * aero.L
    fy_elem = -Cp_solid * aero.ny * aero.L
    Fx_solid = np.sum(fx_elem)
    Fy_solid = np.sum(fy_elem)
    CL_solid = Fy_solid * np.cos(aero.alpha) - Fx_solid * np.sin(aero.alpha)
    CD_solid = Fx_solid * np.cos(aero.alpha) + Fy_solid * np.sin(aero.alpha)
    print(f"   Baseline CL: {CL_solid:.4f}")

    # Generate Mesh
    print("-> Generating Asymmetric Plenum Mesh...")
    G, porous_pids = generate_tangential_mesh(
        aero.XC, aero.YC, aero.tx, aero.ty, Cp_solid, 
        50, PORE_RADIUS, MU
    )
    
    print(f"   Generated network with {len(G.nodes())} nodes.")

    V_leakage = np.zeros(aero.N)
    final_P_nodes = np.zeros(len(G.nodes()))  # Initialize

    # --- Iteration Loop ---
    for i in range(MAX_ITER):
        Cp = aero.solve(V_leakage)
        q_inf = 0.5 * RHO * V_INF**2
        P_ext = P_INF + q_inf * Cp
        P_map = {pid: P_ext[pid] for pid in porous_pids}
        
        # Internal Flow Solver
        V_calculated, P_nodes_iter = solve_internal_flow(G, P_map)
        final_P_nodes = P_nodes_iter # Store for plotting

        # Relaxation
        max_diff = 0.0
        V_new = V_leakage.copy()
        for pid, v_calc in V_calculated.items():
            v_relaxed = RELAXATION * v_calc + (1 - RELAXATION) * V_leakage[pid]
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

    # --- 2. Calculate POROUS Forces (THIS WAS MISSING) ---
    fx_elem = -Cp * aero.nx * aero.L
    fy_elem = -Cp * aero.ny * aero.L
    Fx_porous = np.sum(fx_elem)
    Fy_porous = np.sum(fy_elem)
    
    # Calculate Lift and Drag
    CL = Fy_porous * np.cos(aero.alpha) - Fx_porous * np.sin(aero.alpha)
    CD = Fx_porous * np.cos(aero.alpha) + Fy_porous * np.sin(aero.alpha)

    print(f"-> Final Results: Solid CL={CL_solid:.4f}, Porous CL={CL:.4f}")
    
    # Pass everything to the plotter
    plot_results(aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid, G, final_P_nodes)

if __name__ == "__main__":
    run_simulation()

