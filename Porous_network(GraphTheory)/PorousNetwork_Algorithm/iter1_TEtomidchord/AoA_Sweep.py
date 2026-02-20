import numpy as np
import matplotlib.pyplot as plt
import test  # This imports your new Class-based script
from tqdm import tqdm
import warnings
import os
from dataclasses import dataclass, field
from typing import List, Dict

# Suppress runtime warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================
@dataclass
class SweepResult:
    """Organized container for simulation results."""
    name: str
    radius: float
    angles: List[float] = field(default_factory=list)
    cl_solid: List[float] = field(default_factory=list)
    cd_solid: List[float] = field(default_factory=list)
    cl_porous: List[float] = field(default_factory=list)
    cd_porous: List[float] = field(default_factory=list)
    delta_cl: List[float] = field(default_factory=list)
    delta_r_cl: List[float] = field(default_factory=list)
    cp_data: Dict[float, dict] = field(default_factory=dict)

# ==============================================================================
# 2. CORE SIMULATION LOGIC
# ==============================================================================
def run_oo_sweep(angles, pore_radius, label, capture_angles=[]):
    """
    Runs the AoA sweep using the NEW Object-Oriented test.py structure.
    """
    print(f"\n--- STARTING SWEEP: {label} (Radius={pore_radius*1000:.1f}mm) ---")
    
    # 1. Setup Configuration
    # We create a Config object and overwrite the defaults with our sweep parameters
    cfg = test.Config()
    cfg.PORE_RADIUS_INLET = pore_radius  # Set the radius for this run
    cfg.PORE_RADIUS_OUTLET = pore_radius
    
    # Initialize Result Container
    res = SweepResult(name=label, radius=pore_radius)

    # 2. Generate Geometry & Solver ONCE (Optimization)
    # Note: We use AirfoilGenerator class now, not a global function
    X, Y = test.AirfoilGenerator.generate_naca4(cfg.AIRFOIL_NAME, cfg.N_PANELS)
    aero = test.PanelMethod(X, Y, cfg)

    # Progress bar
    pbar = tqdm(angles, desc=f"Simulating {label}", unit="deg")

    for alpha in pbar:
        # 3. Update Solver State for new Alpha
        cfg.ANGLE_OF_ATTACK = alpha
        aero.alpha = np.radians(alpha) 

        # 4. Solve Baseline (Solid)
        Cp_solid = aero.solve(np.zeros(aero.N))

        # Calculate Solid Forces
        fx_s = -Cp_solid * aero.nx * aero.L
        fy_s = -Cp_solid * aero.ny * aero.L
        Fx_s, Fy_s = np.sum(fx_s), np.sum(fy_s)
        CL_s = Fy_s * np.cos(aero.alpha) - Fx_s * np.sin(aero.alpha)
        CD_s = Fx_s * np.cos(aero.alpha) + Fy_s * np.sin(aero.alpha)

        # 5. Build Porous Network 
        # (This replaces generate_tangential_mesh. The class handles it internally)
        net = test.PorousNetwork(aero, Cp_solid, cfg)

        # 6. Iteration Loop (Coupled Solver)
        V_leakage = np.zeros(aero.N)
        Cp_porous = Cp_solid.copy() 

        for i in range(cfg.MAX_ITER):
            # External Flow
            Cp_porous = aero.solve(V_leakage)
            
            # Internal Flow (Using the Network Class)
            q_inf = 0.5 * cfg.RHO * cfg.V_INF**2
            P_ext = cfg.P_INF + q_inf * Cp_porous
            P_map = {pid: P_ext[pid] for pid in net.active_pores}
            
            # Solve Network
            V_calc, _ = net.solve_flow(P_map)
            
            # Relaxation
            max_diff = 0.0
            V_new = V_leakage.copy()
            for pid, v in V_calc.items():
                v_rel = cfg.RELAXATION * v + (1 - cfg.RELAXATION) * V_leakage[pid]
                v_rel = max(min(v_rel, 80.0), -80.0) # Limiter
                if abs(v_rel - V_leakage[pid]) > max_diff: max_diff = abs(v_rel - V_leakage[pid])
                V_new[pid] = v_rel
            V_leakage = V_new
            
            if max_diff < cfg.CONVERGENCE_TOL:
                break

        # 7. Calculate Porous Forces
        fx_p = -Cp_porous * aero.nx * aero.L
        fy_p = -Cp_porous * aero.ny * aero.L
        Fx_p, Fy_p = np.sum(fx_p), np.sum(fy_p)
        CL_p = Fy_p * np.cos(aero.alpha) - Fx_p * np.sin(aero.alpha)
        CD_p = Fx_p * np.cos(aero.alpha) + Fy_p * np.sin(aero.alpha)

        # 8. Store Data
        res.angles.append(alpha)
        res.cl_solid.append(CL_s)
        res.cd_solid.append(CD_s)
        res.cl_porous.append(CL_p)
        res.cd_porous.append(CD_p)
        res.delta_cl.append(CL_p - CL_s)
        res.delta_r_cl.append((CL_p - CL_s) / (abs(CL_s) + 1e-9))

        pbar.set_postfix({"CL_Solid": f"{CL_s:.2f}", "CL_Porous": f"{CL_p:.2f}"})

        if alpha in capture_angles:
            res.cp_data[alpha] = {
                'xc': aero.XC,
                'cp_solid': Cp_solid,
                'cp_porous': Cp_porous,
                'leakage': V_leakage
            }

    return res

# ==============================================================================
# 3. SAVE & PLOT (Unchanged logic, just data handling)
# ==============================================================================
def save_sweep_data(r1: SweepResult, r2: SweepResult, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # --- POLAR SUMMARY ---
    polar_path = os.path.join(output_dir, "polar_summary.csv")
    print(f"-> Exporting Polar Summary to: {polar_path}")
    
    with open(polar_path, 'w') as f:
        f.write("--- POLAR SUMMARY ---\n")
        header = (f"Alpha_deg,CL_Solid,CD_Solid,"
                  f"{r1.name}_CL,{r1.name}_CD,{r1.name}_DeltaCL,{r1.name}_PctChange,"
                  f"{r2.name}_CL,{r2.name}_CD,{r2.name}_DeltaCL,{r2.name}_PctChange\n")
        f.write(header)
        
        for i in range(len(r1.angles)):
            alpha = r1.angles[i]
            line = (f"{alpha:.2f},{r1.cl_solid[i]:.6f},{r1.cd_solid[i]:.6f},"
                    f"{r1.cl_porous[i]:.6f},{r1.cd_porous[i]:.6f},{r1.delta_cl[i]:.6f},{r1.delta_r_cl[i]*100:.2f},"
                    f"{r2.cl_porous[i]:.6f},{r2.cd_porous[i]:.6f},{r2.delta_cl[i]:.6f},{r2.delta_r_cl[i]*100:.2f}\n")
            f.write(line)

    # --- CP DATA ---
    cp_path = os.path.join(output_dir, "cp_distributions.csv")
    print(f"-> Exporting Cp Distributions to: {cp_path}")
    
    with open(cp_path, 'w') as f:
        f.write("Case_Name,Alpha_deg,Panel_XC,Cp_Solid,Cp_Porous,Leakage_Velocity\n")
        def write_block(res_obj):
            for ang in sorted(res_obj.cp_data.keys()):
                d = res_obj.cp_data[ang]
                for k in range(len(d['xc'])):
                    row = (f"{res_obj.name},{ang:.1f},{d['xc'][k]:.6f},"
                           f"{d['cp_solid'][k]:.6f},{d['cp_porous'][k]:.6f},{d['leakage'][k]:.6f}\n")
                    f.write(row)
        write_block(r1)
        write_block(r2)

def plot_full_comparison(r1: SweepResult, r2: SweepResult, output_dir):
    print("-> Generating plots...")
    style_solid = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.5, 'label': 'Solid Baseline'}
    style_case1 = {'color': 'black', 'linestyle': '-', 'marker': 'o', 'markersize': 4, 'label': r1.name}
    style_case2 = {'color': 'red',   'linestyle': '-', 'marker': 's', 'markersize': 4, 'label': r2.name}

    # Polars
    fig1 = plt.figure(figsize=(14, 10))
    gs1 = fig1.add_gridspec(2, 2)
    fig1.suptitle(f"Aerodynamic Polars Comparison (NACA {test.Config.AIRFOIL_NAME})", fontsize=16)

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(r1.angles, r1.cl_solid, **style_solid)
    ax1.plot(r1.angles, r1.cl_porous, **style_case1)
    ax1.plot(r2.angles, r2.cl_porous, **style_case2)
    ax1.set_ylabel("$C_L$"); ax1.grid(True, alpha=0.5); ax1.legend()
    ax1.set_title("Lift Coefficient")

    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.plot(r1.cd_solid, r1.cl_solid, **style_solid)
    ax2.plot(r1.cd_porous, r1.cl_porous, **style_case1)
    ax2.plot(r2.cd_porous, r2.cl_porous, **style_case2)
    ax2.set_xlabel("$C_D$"); ax2.set_ylabel("$C_L$"); ax2.grid(True, alpha=0.5)
    ax2.set_title("Drag Polar")

    ax3 = fig1.add_subplot(gs1[1, :])
    ld_s = np.array(r1.cl_solid) / (np.array(r1.cd_solid) + 1e-9)
    ld_1 = np.array(r1.cl_porous) / (np.array(r1.cd_porous) + 1e-9)
    ld_2 = np.array(r2.cl_porous) / (np.array(r2.cd_porous) + 1e-9)
    ax3.plot(r1.angles, ld_s, **style_solid)
    ax3.plot(r1.angles, ld_1, **style_case1)
    ax3.plot(r2.angles, ld_2, **style_case2)
    ax3.set_ylabel("$L/D$"); ax3.set_xlabel("Alpha (deg)"); ax3.grid(True, alpha=0.5)
    ax3.set_title("Efficiency ($L/D$)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "01_Polars.png"), dpi=200)
    plt.close(fig1)

    # Deltas
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 5))
    ax4.plot(r1.angles, np.array(r1.delta_r_cl)*100, **style_case1)
    ax4.plot(r2.angles, np.array(r2.delta_r_cl)*100, **style_case2)
    ax4.set_ylabel("Change (%)"); ax4.set_xlabel("Alpha (deg)"); ax4.grid(True, alpha=0.5); ax4.axhline(0, color='gray')
    ax4.set_title("Relative Lift Change")

    ax5.plot(r1.angles, r1.delta_cl, **style_case1)
    ax5.plot(r2.angles, r2.delta_cl, **style_case2)
    ax5.set_ylabel(r"$\Delta C_L$"); ax5.set_xlabel("Alpha (deg)"); ax5.grid(True, alpha=0.5); ax5.axhline(0, color='gray')
    ax5.set_title("Absolute Lift Increment")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "02_Deltas.png"), dpi=200)
    plt.close(fig2)

    # Cp Plots
    if r1.cp_data:
        angles = sorted(r1.cp_data.keys())
        fig3, axes = plt.subplots(1, len(angles), figsize=(5*len(angles), 6))
        if len(angles) == 1: axes = [axes]
        
        for i, ang in enumerate(angles):
            d1 = r1.cp_data[ang]
            d2 = r2.cp_data[ang]
            ax = axes[i]
            ax.plot(d1['xc'], d1['cp_solid'], 'gray', ls='--', label='Solid')
            ax.plot(d1['xc'], d1['cp_porous'], 'k-', label=r1.name)
            ax.plot(d2['xc'], d2['cp_porous'], 'r-.', label=r2.name)
            ax.invert_yaxis()
            ax.set_title(f"Alpha = {ang}°")
            ax.set_xlabel("x/c")
            if i==0: ax.set_ylabel("$C_p$")
            ax.legend()
        fig3.tight_layout()
        fig3.savefig(os.path.join(output_dir, "03_Cp_Distributions.png"), dpi=200)
        plt.close(fig3)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Settings
    CONFIG = {
        'AOA_RANGE': np.arange(-5.0, 10.1, 1.0),
        'CP_ANGLES': [-5.0, 5.0, 10.0],
        'OUT_DIR': os.path.join(os.path.dirname(os.path.abspath(__file__)), "aoa_sweep_results"),
        'R1': 4000e-6,
        'R2': 2000e-6
    }

    # Run
    res1 = run_oo_sweep(CONFIG['AOA_RANGE'], CONFIG['R1'], "Large Pores", CONFIG['CP_ANGLES'])
    res2 = run_oo_sweep(CONFIG['AOA_RANGE'], CONFIG['R2'], "Small Pores", CONFIG['CP_ANGLES'])

    # Output
    save_sweep_data(res1, res2, CONFIG['OUT_DIR'])
    plot_full_comparison(res1, res2, CONFIG['OUT_DIR'])
    
    print(f"\n-> Completed. Results saved to: {CONFIG['OUT_DIR']}")