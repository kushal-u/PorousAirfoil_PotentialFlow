import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import test  # Imports your base solver
from tqdm import tqdm

# ==============================================================================
# 1. NEW MESH GENERATOR (Short-Path Booster)
# ==============================================================================
def generate_optimized_mesh(xc, yc, tx, ty, n_target, pore_radius, mu):
    """
    Generates a 'Short-Path' network to minimize internal friction drag.
    
    Strategy:
      1. Source: Bottom Mid (0.30-0.60).
      2. Target: Top Mid-Aft (0.65-0.85).
      3. Physics: Connects 'Neutral' Bottom to 'High Suction' Top.
         - High Driving Force (Delta P).
         - Short Tube Length (Lower Friction).
    """
    import networkx as nx
    G = nx.Graph()
    
    # Define Zones
    # Target: Top Surface, Suction Zone (High driving force region)
    outlet_candidates = [i for i in range(len(xc)) if yc[i] > 0 and 0.65 <= xc[i] <= 0.85]
    
    # Source: Bottom Surface, Mid-Chord
    inlet_candidates = [i for i in range(len(xc)) if yc[i] < 0 and 0.30 <= xc[i] <= 0.60]

    connections_made = 0
    
    # We want to limit the total pores to n_target exactly if possible
    # So we'll collect all valid edges and then select the best ones
    valid_edges = []

    for j in outlet_candidates:
        p_out = np.array([xc[j], yc[j]])
        target_dir = np.array([-tx[j], -ty[j]])
        
        best_inlet = -1
        best_angle = 999.0
        
        for i in inlet_candidates:
            p_in = np.array([xc[i], yc[i]])
            vec_pore = p_out - p_in
            len_pore = np.linalg.norm(vec_pore)
            dir_pore = vec_pore / len_pore
            
            dot_prod = np.dot(dir_pore, target_dir)
            angle = np.degrees(np.arccos(max(min(dot_prod, 1.0), -1.0)))
            
            if angle < best_angle:
                best_angle = angle
                best_inlet = i
        
        # Filter: Allow slightly steeper angles (45 deg) for shorter paths
        if best_inlet != -1 and best_angle < 45.0:
            cond = (np.pi * pore_radius**4) / (8 * mu * np.linalg.norm(np.array([xc[best_inlet], yc[best_inlet]]) - p_out))
            valid_edges.append({
                'u': best_inlet, 'v': j, 
                'len': np.linalg.norm(np.array([xc[best_inlet], yc[best_inlet]]) - p_out),
                'cond': cond, 'angle': best_angle
            })

    # Sort edges by alignment (angle) to pick the most efficient ones
    valid_edges.sort(key=lambda x: x['angle'])
    
    # Select top N edges
    selected_edges = valid_edges[:n_target]
    
    for edge in selected_edges:
        u, v = edge['u'], edge['v']
        if u not in G: G.add_node(u, pos=(xc[u], yc[u]), type='boundary', panel_idx=u)
        if v not in G: G.add_node(v, pos=(xc[v], yc[v]), type='boundary', panel_idx=v)
        G.add_edge(u, v, length=edge['len'], cond=edge['cond'], type='standard')
        connections_made += 1

    return G, [n for n in G.nodes()]

# ==============================================================================
# 2. PORE DENSITY SWEEP
# ==============================================================================
def run_pore_sweep():
    # Sweep Parameters
    pore_counts = [10, 20, 30, 40, 50, 60]
    pore_radius = 3500e-6  # Keep large 3.5mm radius
    
    results = []
    
    print(f"--- STARTING EFFICIENCY SWEEP (Radius={pore_radius*1000}mm) ---")
    
    # 1. Run Solid Baseline Once
    X, Y = test.naca4(test.AIRFOIL_NAME, n_panels=test.N_PANELS)
    aero = test.PanelMethod(X, Y, 5.0) # Fixed Alpha = 5 deg
    Cp_solid = aero.solve(np.zeros(aero.N))
    
    # Calculate Solid Forces
    fx = -Cp_solid * aero.nx * aero.L
    fy = -Cp_solid * aero.ny * aero.L
    Fx = np.sum(fx); Fy = np.sum(fy)
    CL_solid = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD_solid = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    LD_solid = CL_solid / (CD_solid + 1e-9)
    
    print(f"Baseline (Solid): CL={CL_solid:.4f}, CD={CD_solid:.5f}, L/D={LD_solid:.2f}")

    # 2. Sweep Loop
    for n_pores in pore_counts:
        # Generate Mesh
        G, porous_pids = generate_optimized_mesh(
            aero.XC, aero.YC, aero.tx, aero.ty, 
            n_pores, pore_radius, test.MU
        )
        
        # Solve Coupled System
        V_leakage = np.zeros(aero.N)
        for i in range(50): # Fewer iters for speed
            Cp = aero.solve(V_leakage)
            q_inf = 0.5 * test.RHO * test.V_INF**2
            P_ext = test.P_INF + q_inf * Cp
            P_map = {pid: P_ext[pid] for pid in porous_pids}
            V_calc, _ = test.solve_internal_flow(G, P_map)
            
            # Relax
            for pid, v in V_calc.items():
                V_leakage[pid] = 0.1 * v + 0.9 * V_leakage[pid]
        
        # Calculate Porous Forces
        fx = -Cp * aero.nx * aero.L
        fy = -Cp * aero.ny * aero.L
        Fx = np.sum(fx); Fy = np.sum(fy)
        CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
        CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
        LD = CL / (CD + 1e-9)
        
        results.append({
            'N_Pores': n_pores,
            'CL': CL, 'CD': CD, 'L/D': LD,
            'Delta_LD_Pct': (LD - LD_solid)/LD_solid * 100
        })
        
        print(f"   N={n_pores}: CL={CL:.4f}, CD={CD:.5f}, L/D={LD:.2f} ({results[-1]['Delta_LD_Pct']:+.1f}%)")

    return pd.DataFrame(results), LD_solid

# ==============================================================================
# 3. PLOTTING
# ==============================================================================
if __name__ == "__main__":
    df, ld_base = run_pore_sweep()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Efficiency vs Pores
    ax1.plot(df['N_Pores'], df['L/D'], 'b-o', label='Porous L/D')
    ax1.axhline(ld_base, color='k', linestyle='--', label='Solid Baseline')
    ax1.set_xlabel('Number of Pores')
    ax1.set_ylabel('Lift-to-Drag Ratio (L/D)')
    ax1.set_title('Effect of Pore Density on Efficiency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Breakout of CL and CD
    ax2_cl = ax2.twinx()
    l1 = ax2.plot(df['N_Pores'], df['CD'], 'r-s', label='Drag (CD)')
    l2 = ax2_cl.plot(df['N_Pores'], df['CL'], 'g-^', label='Lift (CL)')
    
    ax2.set_xlabel('Number of Pores')
    ax2.set_ylabel('Drag Coefficient', color='r')
    ax2_cl.set_ylabel('Lift Coefficient', color='g')
    ax2.set_title('Forces Breakdown')
    
    # Legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='center right')
    
    plt.tight_layout()
    plt.show()