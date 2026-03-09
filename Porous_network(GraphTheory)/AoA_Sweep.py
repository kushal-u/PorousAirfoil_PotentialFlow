import numpy as np
import matplotlib.pyplot as plt
import porous_network_panel_method 
from tqdm import tqdm
import warnings
import os

# Suppress runtime warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def calculate_forces(aero, Cp):
    """
    Calculates Lift and Drag Coefficients from Cp.
    Correctly transforms body forces to wind axes.
    """
    fx_elem = -Cp * aero.nx * aero.L
    fy_elem = -Cp * aero.ny * aero.L
    
    Fx_body = np.sum(fx_elem)
    Fy_body = np.sum(fy_elem)
    
    CL = Fy_body * np.cos(aero.alpha) - Fx_body * np.sin(aero.alpha)
    CD = Fx_body * np.cos(aero.alpha) + Fy_body * np.sin(aero.alpha)
    
    return CL, CD

def run_sweep(aoa_range, pore_radius, label_name, capture_angles=[0, 5, 10]):
    print(f"\n--- STARTING SWEEP: {label_name} (Radius={pore_radius*1000:.1f}mm) ---")
    
    porous_network_panel_method.PORE_RADIUS = pore_radius
    
    # Pre-Generate Mesh
    X, Y = porous_network_panel_method.naca4(porous_network_panel_method.AIRFOIL_NAME, n_panels=porous_network_panel_method.N_PANELS)
    temp_aero = porous_network_panel_method.PanelMethod(X, Y, 0.0)
    G, porous_pids = porous_network_panel_method.generate_structured_mesh(
        X, Y, temp_aero.XC, temp_aero.YC, 
        porous_network_panel_method.X_START, porous_network_panel_method.X_END, porous_network_panel_method.N_PORES,
        pore_radius, porous_network_panel_method.MU
    )
    
    results = {
        'alpha': [],
        'cl_solid': [], 'cd_solid': [],
        'cl_porous': [], 'cd_porous': [],
        'delta_cl': [], 'delta_r_cl': [],
        'cp_data': {}  # New storage for Cp distributions
    }

    for alpha_deg in tqdm(aoa_range):
        results['alpha'].append(alpha_deg)
        
        # 1. Initialize Solver
        aero = porous_network_panel_method.PanelMethod(X, Y, alpha_deg)
        
        # 2. Solid Case
        Cp_solid = aero.solve(np.zeros(aero.N))
        cl_s, cd_s = calculate_forces(aero, Cp_solid)
        results['cl_solid'].append(cl_s)
        results['cd_solid'].append(cd_s)
        
        # 3. Porous Case
        V_leakage = np.zeros(aero.N)
        for i in range(50): 
            Cp = aero.solve(V_leakage)
            q_inf = 0.5 * porous_network_panel_method.RHO * porous_network_panel_method.V_INF**2
            P_ext = porous_network_panel_method.P_INF + q_inf * Cp
            P_map = {pid: P_ext[pid] for pid in porous_pids}
            V_calculated, _ = porous_network_panel_method.solve_internal_flow(G, P_map)
            
            V_new = V_leakage.copy()
            max_diff = 0.0
            for pid, v_calc in V_calculated.items():
                v_relaxed = porous_network_panel_method.RELAXATION * v_calc + (1 - porous_network_panel_method.RELAXATION) * V_leakage[pid]
                v_relaxed = max(min(v_relaxed, 50.0), -50.0)
                if abs(v_relaxed - V_leakage[pid]) > max_diff: max_diff = abs(v_relaxed - V_leakage[pid])
                V_new[pid] = v_relaxed
            V_leakage = V_new
            if max_diff < porous_network_panel_method.CONVERGENCE_TOL: break
        
        cl_p, cd_p = calculate_forces(aero, Cp)
        results['cl_porous'].append(cl_p)
        results['cd_porous'].append(cd_p)
        
        # Calculate Deltas
        d_cl = cl_p - cl_s
        if abs(cl_s) > 1e-4: d_r_cl = (cl_p - cl_s) / abs(cl_s)
        else: d_r_cl = 0.0
            
        results['delta_cl'].append(d_cl)
        results['delta_r_cl'].append(d_r_cl)

        # 4. CAPTURE CP DATA (For specific angles)
        if alpha_deg in capture_angles:
            results['cp_data'][alpha_deg] = {
                'xc': aero.XC,
                'cp_solid': Cp_solid.copy(),
                'cp_porous': Cp.copy(),
                'leakage': V_leakage.copy()
            }

    return results

# ==============================================================================
# 2. SAVE RESULTS TO CSV (SPLIT FILES)
# ==============================================================================
def save_sweep_data(res1, res2, name1, name2, output_dir="aoa_sweep_results"):
    """
    Saves results into two separate CSV files in the output directory.
    1. polar_summary.csv (Now includes % changes)
    2. cp_distributions.csv
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- FILE 1: POLAR SUMMARY ---
    polar_path = os.path.join(output_dir, "polar_summary.csv")
    print(f"-> Exporting Polar Summary to: {polar_path}")
    
    try:
        with open(polar_path, 'w') as f:
            f.write("--- POLAR SUMMARY ---\n")
            header = (f"Alpha_deg,CL_Solid,CD_Solid,"
                      f"{name1}_CL,{name1}_CD,{name1}_DeltaCL,{name1}_PctChange,"
                      f"{name2}_CL,{name2}_CD,{name2}_DeltaCL,{name2}_PctChange\n")
            f.write(header)
            
            count = len(res1['alpha'])
            for i in range(count):
                alpha = res1['alpha'][i]
                
                # Baseline
                cl_s = res1['cl_solid'][i]
                cd_s = res1['cd_solid'][i]
                
                # Case 1
                cl_1 = res1['cl_porous'][i]
                cd_1 = res1['cd_porous'][i]
                dcl_1 = res1['delta_cl'][i]
                pct_1 = res1['delta_r_cl'][i] * 100
                
                # Case 2
                cl_2 = res2['cl_porous'][i]
                cd_2 = res2['cd_porous'][i]
                dcl_2 = res2['delta_cl'][i]
                pct_2 = res2['delta_r_cl'][i] * 100
                
                line = (f"{alpha:.2f},{cl_s:.6f},{cd_s:.6f},"
                        f"{cl_1:.6f},{cd_1:.6f},{dcl_1:.6f},{pct_1:.2f},"
                        f"{cl_2:.6f},{cd_2:.6f},{dcl_2:.6f},{pct_2:.2f}\n")
                f.write(line)
    except IOError as e:
        print(f"Error saving Polar CSV: {e}")

    # --- FILE 2: DETAILED CP DATA ---
    cp_path = os.path.join(output_dir, "cp_distributions.csv")
    print(f"-> Exporting Cp Distributions to: {cp_path}")
    
    try:
        with open(cp_path, 'w') as f:
            f.write("--- DETAILED CP DISTRIBUTIONS (CAPTURED ANGLES) ---\n")
            f.write("Case_Name,Alpha_deg,Panel_XC,Cp_Solid,Cp_Porous,Leakage_Velocity\n")
            
            def write_cp_block(res_dict, label):
                # Iterate through captured angles
                for ang in sorted(res_dict['cp_data'].keys()):
                    data = res_dict['cp_data'][ang]
                    xc = data['xc']
                    cp_s = data['cp_solid']
                    cp_p = data['cp_porous']
                    v_l = data['leakage']
                    
                    for k in range(len(xc)):
                        row = (f"{label},{ang:.1f},{xc[k]:.6f},"
                               f"{cp_s[k]:.6f},{cp_p[k]:.6f},{v_l[k]:.6f}\n")
                        f.write(row)

            write_cp_block(res1, name1)
            write_cp_block(res2, name2)
    except IOError as e:
        print(f"Error saving Cp CSV: {e}")

# ==============================================================================
# 3. PLOTTING ROUTINE (SAVES IMAGES)
# ==============================================================================

def plot_full_comparison(res1, res2, name1, name2, output_dir="aoa_sweep_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alphas = np.array(res1['alpha'])
    
    style_solid = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.5, 'label': 'Solid Baseline'}
    style_case1 = {'color': 'black', 'linestyle': '-', 'marker': 'o', 'markersize': 4, 'label': name1}
    style_case2 = {'color': 'red',   'linestyle': '-', 'marker': 's', 'markersize': 4, 'label': name2}

    # --- PAGE 1: Polars ---
    fig1 = plt.figure(figsize=(14, 10))
    gs1 = fig1.add_gridspec(2, 2)
    fig1.suptitle(f"Page 1: Aerodynamic Polars Comparison (Re={int(porous_network_panel_method.REYNOLDS_NUM)})", fontsize=16)

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(alphas, res1['cl_solid'], **style_solid)
    ax1.plot(alphas, res1['cl_porous'], **style_case1)
    ax1.plot(alphas, res2['cl_porous'], **style_case2)
    ax1.set_title("1. Lift Coefficient ($C_L$)", fontsize=12)
    ax1.set_xlabel("Alpha (deg)"); ax1.set_ylabel("$C_L$"); ax1.grid(True, alpha=0.5); ax1.legend()

    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.plot(res1['cd_solid'], res1['cl_solid'], **style_solid)
    ax2.plot(res1['cd_porous'], res1['cl_porous'], **style_case1)
    ax2.plot(res2['cd_porous'], res2['cl_porous'], **style_case2)
    ax2.set_title("2. Drag Polar", fontsize=12)
    ax2.set_xlabel("$C_D$"); ax2.set_ylabel("$C_L$"); ax2.grid(True, alpha=0.5)

    ld_solid = np.array(res1['cl_solid']) / (np.array(res1['cd_solid']) + 1e-9)
    ld_case1 = np.array(res1['cl_porous']) / (np.array(res1['cd_porous']) + 1e-9)
    ld_case2 = np.array(res2['cl_porous']) / (np.array(res2['cd_porous']) + 1e-9)

    ax3 = fig1.add_subplot(gs1[1, :])
    ax3.plot(alphas, ld_solid, **style_solid)
    ax3.plot(alphas, ld_case1, **style_case1)
    ax3.plot(alphas, ld_case2, **style_case2)
    ax3.set_title("3. Efficiency ($L/D$)", fontsize=12)
    ax3.set_xlabel("Alpha (deg)"); ax3.set_ylabel("$L/D$"); ax3.grid(True, alpha=0.5)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "01_Polars_Comparison.png"), dpi=300, bbox_inches='tight')

    # --- PAGE 2: Lift Deltas ---
    fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f"Page 2: Detailed Impact of Porosity on Lift", fontsize=16)
    
    ax4.plot(alphas, res1['cl_solid'], **style_solid)
    ax4.plot(alphas, res1['cl_porous'], **style_case1)
    ax4.plot(alphas, res2['cl_porous'], **style_case2)
    ax4.set_title("4. Reference Lift Curve", fontsize=12)
    ax4.set_xlabel("Alpha (deg)"); ax4.set_ylabel("$C_L$"); ax4.grid(True, alpha=0.5); ax4.legend()

    ax5.plot(alphas, np.array(res1['delta_r_cl'])*100, **style_case1)
    ax5.plot(alphas, np.array(res2['delta_r_cl'])*100, **style_case2)
    ax5.set_title(r"5. Relative Change % ($\Delta_r C_L$)", fontsize=12)
    ax5.set_xlabel("Alpha (deg)"); ax5.set_ylabel("Change (%)"); ax5.grid(True, alpha=0.5); ax5.axhline(0, color='gray')

    ax6.plot(alphas, res1['delta_cl'], **style_case1)
    ax6.plot(alphas, res2['delta_cl'], **style_case2)
    ax6.set_title(r"6. Absolute Change ($\Delta C_L$)", fontsize=12)
    ax6.set_xlabel("Alpha (deg)"); ax6.set_ylabel(r"$\Delta C_L$"); ax6.grid(True, alpha=0.5); ax6.axhline(0, color='gray')
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "02_Lift_Deltas.png"), dpi=300, bbox_inches='tight')

    # --- PAGE 3: Cp Distributions ---
    captured_angles = list(res1['cp_data'].keys())
    if captured_angles:
        fig3, axes = plt.subplots(1, len(captured_angles), figsize=(5*len(captured_angles), 6))
        if len(captured_angles) == 1: axes = [axes]
        
        fig3.suptitle("Page 3: Pressure Coefficient ($C_p$) Distributions", fontsize=16)
        for i, ang in enumerate(captured_angles):
            ax = axes[i]
            d1 = res1['cp_data'][ang]
            d2 = res2['cp_data'][ang]
            ax.plot(d1['xc'], d1['cp_solid'], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Solid')
            ax.plot(d1['xc'], d1['cp_porous'], color='black', linewidth=1.5, label=name1)
            ax.plot(d2['xc'], d2['cp_porous'], color='red', linewidth=1.5, linestyle='-.', label=name2)
            ax.invert_yaxis() 
            ax.set_title(f"Alpha = {ang}°", fontsize=12)
            ax.set_xlabel("x/c")
            if i == 0: ax.set_ylabel("$C_p$")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, "03_Cp_Distributions.png"), dpi=300, bbox_inches='tight')



    plt.show()
    print(f"-> All plots saved to: {os.path.abspath(output_dir)}")

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # ==========================================
    AOA_START = -5.0   # Start Angle (degrees)
    AOA_END   = 10.0    # End Angle (degrees)
    AOA_STEP  = 1.0     # Step size (degrees)
    
    TARGET_CP_ANGLES = [-5, 5, 10] 

    OUTPUT_FOLDER = "aoa_sweep_results"
    RADIUS_1 = 2000e-6  # 1.0mm
    RADIUS_2 = 5000e-6   # 0.5mm
    
    # ==========================================
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created directory: {OUTPUT_FOLDER}")

    angles = np.arange(AOA_START, AOA_END + AOA_STEP/100, AOA_STEP)
    
    
    label1 = f"Radius {RADIUS_1*1000:.1f}mm"
    label2 = f"Radius {RADIUS_2*1000:.1f}mm"

    res1 = run_sweep(angles, RADIUS_1, label1, capture_angles=TARGET_CP_ANGLES)
    res2 = run_sweep(angles, RADIUS_2, label2, capture_angles=TARGET_CP_ANGLES)
    
    save_sweep_data(res1, res2, label1, label2, output_dir=OUTPUT_FOLDER)
    plot_full_comparison(res1, res2, label1, label2, output_dir=OUTPUT_FOLDER)