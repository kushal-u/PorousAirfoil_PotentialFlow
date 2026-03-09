import numpy as np
import matplotlib.pyplot as plt
import t1  # Imports the base OOP script you provided
from tqdm import tqdm
import warnings
import os
from dataclasses import dataclass, field
from typing import List
from PIL import Image  # Required for stacking the images

# Suppress runtime warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================
@dataclass
class SweepResult:
    """Organized container for simulation results."""
    name: str
    radius_inlet: float
    radius_outlet: float
    angles: List[float] = field(default_factory=list)
    cl_solid: List[float] = field(default_factory=list)
    cd_solid: List[float] = field(default_factory=list)
    cl_porous: List[float] = field(default_factory=list)
    cd_porous: List[float] = field(default_factory=list)
    delta_cl: List[float] = field(default_factory=list)
    delta_r_cl: List[float] = field(default_factory=list)
    delta_r_cd: List[float] = field(default_factory=list) 
    capture_image_paths: List[str] = field(default_factory=list)
# ==============================================================================
# 2. CORE SIMULATION LOGIC
# ==============================================================================
def run_oo_sweep(angles, r_inlet, r_outlet, label, base_out_dir, capture_angles=None):
    """
    Runs the AoA sweep, prints aerodynamic stats for specific angles, 
    and triggers t1.Visualizer.
    """
    if capture_angles is None:
        capture_angles = []
        
    print(f"\n--- STARTING SWEEP: {label} (R_in={r_inlet*1000:.1f}mm, R_out={r_outlet*1000:.1f}mm) ---")
    
    cfg = t1.Config()
    cfg.PORE_RADIUS_INLET = r_inlet
    cfg.PORE_RADIUS_OUTLET = r_outlet
    
    res = SweepResult(name=label, radius_inlet=r_inlet, radius_outlet=r_outlet)

    X, Y = t1.AirfoilGenerator.generate_naca4(cfg.AIRFOIL_NAME, cfg.N_PANELS)
    aero = t1.PanelMethod(X, Y, cfg)

    pbar = tqdm(angles, desc=f"Simulating {label}", unit="deg")

    for alpha in pbar:
        cfg.ANGLE_OF_ATTACK = alpha
        aero.alpha = np.radians(alpha) 

        # Baseline (Solid)
        Cp_solid = aero.solve(np.zeros(aero.N))

        fx_s = -Cp_solid * aero.nx * aero.L
        fy_s = -Cp_solid * aero.ny * aero.L
        Fx_s, Fy_s = np.sum(fx_s), np.sum(fy_s)
        CL_s = Fy_s * np.cos(aero.alpha) - Fx_s * np.sin(aero.alpha)
        CD_s = Fx_s * np.cos(aero.alpha) + Fy_s * np.sin(aero.alpha)

        # Porous Network 
        net = t1.PorousNetwork(aero, Cp_solid, cfg)

        V_leakage = np.zeros(aero.N)
        Cp_porous = Cp_solid.copy() 
        P_nodes_final = None  

        for i in range(cfg.MAX_ITER):
            Cp_porous = aero.solve(V_leakage)
            
            q_inf = 0.5 * cfg.RHO * cfg.V_INF**2
            P_ext = cfg.P_INF + q_inf * Cp_porous
            P_map = {pid: P_ext[pid] for pid in net.active_pores}
            
            V_calc, P_nodes_final = net.solve_flow(P_map)
            
            max_diff = 0.0
            V_new = V_leakage.copy()
            for pid, v in V_calc.items():
                v_rel = cfg.RELAXATION * v + (1 - cfg.RELAXATION) * V_leakage[pid]
                v_rel = max(min(v_rel, 80.0), -80.0) 
                if abs(v_rel - V_leakage[pid]) > max_diff: 
                    max_diff = abs(v_rel - V_leakage[pid])
                V_new[pid] = v_rel
            V_leakage = V_new
            
            if max_diff < cfg.CONVERGENCE_TOL:
                break

        # Porous Forces
        fx_p = -Cp_porous * aero.nx * aero.L
        fy_p = -Cp_porous * aero.ny * aero.L
        Fx_p, Fy_p = np.sum(fx_p), np.sum(fy_p)
        CL_p = Fy_p * np.cos(aero.alpha) - Fx_p * np.sin(aero.alpha)
        CD_p = Fx_p * np.cos(aero.alpha) + Fy_p * np.sin(aero.alpha)

        # Store Data
        res.angles.append(alpha)
        res.cl_solid.append(CL_s)
        res.cd_solid.append(CD_s)
        res.cl_porous.append(CL_p)
        res.cd_porous.append(CD_p)
        res.delta_cl.append(CL_p - CL_s)
        res.delta_r_cl.append((CL_p - CL_s) / (abs(CL_s) + 1e-9))
        res.delta_r_cd.append((CD_p - CD_s) / (abs(CD_s) + 1e-9))  

        pbar.set_postfix({"CL_Solid": f"{CL_s:.2f}", "CL_Porous": f"{CL_p:.2f}"})

        # --- TERMINAL PRINTOUT & PLOT GENERATION FOR SPECIFIC ANGLES ---
        if alpha in capture_angles:
            # 1. Terminal Printout
            cl_change = ((CL_p - CL_s) / (abs(CL_s) + 1e-9)) * 100
            cd_change = ((CD_p - CD_s) / (abs(CD_s) + 1e-9)) * 100
            
            msg = (f"\n  >>> RESULTS FOR AoA = {alpha}° <<<\n"
                   f"      CL: {CL_s:.5f} (Solid) -> {CL_p:.5f} (Porous) | Change: {cl_change:+.2f}%\n"
                   f"      CD: {CD_s:.5f} (Solid) -> {CD_p:.5f} (Porous) | Change: {cd_change:+.2f}%")
            pbar.write(msg) # pbar.write prevents the text from glitching the progress bar

            # 2. Trigger t1.Visualizer
            safe_label = label.replace(" ", "_")
            sub_folder = os.path.join(base_out_dir, f"{safe_label}_AoA_{alpha}")
            cfg.OUTPUT_DIR = sub_folder
            
            viz = t1.Visualizer(cfg)
            viz.save_csv(aero, Cp_porous, Cp_solid, V_leakage, CL_p, CL_s, CD_p, CD_s)
            viz.plot_all(aero, net, Cp_porous, Cp_solid, P_nodes_final)
            export_cp_distribution_csv(
                alpha_deg=alpha,
                Cp_solid=Cp_solid,
                Cp_porous=Cp_porous,
                X=X, Y=Y,
                aero=aero,
                out_dir=sub_folder,
                fname="cp_distribution.csv"
            )

            # Store the path to the geometry/cp image for the stacker
            target_img = os.path.join(sub_folder, '01_Geometry_Cp.png')
            res.capture_image_paths.append(target_img)

    return res
import csv

import csv
import numpy as np
import os

def export_cp_distribution_csv(alpha_deg, Cp_solid, Cp_porous, X, Y, aero, out_dir, fname="cp_distribution.csv"):
    """
    Export Cp distribution (solid + porous) to CSV with panel midpoints computed from X,Y.

    Columns:
      panel_id, x0, y0, x1, y1, x_mid, y_mid, s, L, nx, ny, Cp_solid, Cp_porous, dCp
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()

    # If geometry is closed, it might have N+1 points for N panels (common)
    # We’ll assume panels are consecutive point pairs.
    Np = len(X) - 1
    if len(Cp_solid) != Np:
        # fallback: if Cp has aero.N, use that and truncate/align
        Np = min(len(Cp_solid), len(X) - 1)

    x0, y0 = X[:Np], Y[:Np]
    x1, y1 = X[1:Np+1], Y[1:Np+1]
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)

    # Panel properties
    L = np.asarray(aero.L).flatten()[:Np] if hasattr(aero, "L") else np.sqrt((x1-x0)**2 + (y1-y0)**2)
    nx = np.asarray(aero.nx).flatten()[:Np] if hasattr(aero, "nx") else np.full(Np, np.nan)
    ny = np.asarray(aero.ny).flatten()[:Np] if hasattr(aero, "ny") else np.full(Np, np.nan)

    # Arc-length coordinate (cumulative along surface)
    s = np.zeros(Np)
    if len(L) == Np:
        s = np.cumsum(L) - 0.5 * L  # midpoint arc-length

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# AoA_deg={alpha_deg}"])
        w.writerow([
            "panel_id",
            "x0","y0","x1","y1",
            "x_mid","y_mid",
            "s","L","nx","ny",
            "Cp_solid","Cp_porous","dCp_porous_minus_solid"
        ])

        for i in range(Np):
            w.writerow([
                i,
                float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i]),
                float(x_mid[i]), float(y_mid[i]),
                float(s[i]), float(L[i]),
                float(nx[i]), float(ny[i]),
                float(Cp_solid[i]), float(Cp_porous[i]),
                float(Cp_porous[i] - Cp_solid[i])
            ])

    return out_path


# ==============================================================================
# 3. IMAGE STACKER & OUTPUT LOGIC
# ==============================================================================
def stack_case_images(res: SweepResult, out_dir: str):
    """Reads the individual 01_Geometry_Cp.png files and stacks them vertically."""
    if not res.capture_image_paths: return
    
    images = []
    for path in res.capture_image_paths:
        if os.path.exists(path):
            images.append(Image.open(path))
            
    if not images: return
    
    # Calculate dimensions for the new stacked image
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    max_height = sum(heights)
    
    # Create blank canvas and paste images
    stacked_im = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    y_offset = 0
    for im in images:
        stacked_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
        
    # Save the final page
    safe_name = res.name.replace(" ", "_")
    out_file = os.path.join(out_dir, f"{safe_name}_Stacked_Cp_Summary.png")
    stacked_im.save(out_file)
    print(f"-> Stacked Cp summary created: {out_file}")

def save_sweep_summary(r1: SweepResult, r2: SweepResult, output_dir):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    polar_path = os.path.join(output_dir, "polar_summary.csv")
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

def plot_polars(r1: SweepResult, r2: SweepResult, output_dir):
    style_solid = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.5, 'label': 'Solid Baseline'}
    style_case1 = {'color': 'black', 'linestyle': '-', 'marker': 'o', 'markersize': 4, 'label': r1.name}
    style_case2 = {'color': 'red',   'linestyle': '-', 'marker': 's', 'markersize': 4, 'label': r2.name}

    # --- FIGURE 1: Main Aerodynamic Polars (Now a 2x2 Grid) ---
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f"Aerodynamic Polars Comparison (NACA {t1.Config.AIRFOIL_NAME})", fontsize=16)

    # Top Left: Lift Coefficient vs Alpha
    ax1 = axes[0, 0]
    ax1.plot(r1.angles, r1.cl_solid, **style_solid)
    ax1.plot(r1.angles, r1.cl_porous, **style_case1)
    ax1.plot(r2.angles, r2.cl_porous, **style_case2)
    ax1.set_ylabel("$C_L$"); ax1.set_xlabel("Alpha (deg)")
    ax1.grid(True, alpha=0.5); ax1.legend()
    ax1.set_title("Lift Coefficient vs Alpha")

    # Top Right: Drag Coefficient vs Alpha (NEW)
    ax2 = axes[0, 1]
    ax2.plot(r1.angles, r1.cd_solid, **style_solid)
    ax2.plot(r1.angles, r1.cd_porous, **style_case1)
    ax2.plot(r2.angles, r2.cd_porous, **style_case2)
    ax2.set_ylabel("$C_D$"); ax2.set_xlabel("Alpha (deg)")
    ax2.grid(True, alpha=0.5)
    ax2.set_title("Drag Coefficient vs Alpha")

    # Bottom Left: Drag Polar (CL vs CD)
    ax3 = axes[1, 0]
    ax3.plot(r1.cd_solid, r1.cl_solid, **style_solid)
    ax3.plot(r1.cd_porous, r1.cl_porous, **style_case1)
    ax3.plot(r2.cd_porous, r2.cl_porous, **style_case2)
    ax3.set_xlabel("$C_D$"); ax3.set_ylabel("$C_L$")
    ax3.grid(True, alpha=0.5)
    ax3.set_title("Drag Polar")

    # Bottom Right: Efficiency (L/D) vs Alpha
    ax4 = axes[1, 1]
    ld_s = np.array(r1.cl_solid) / (np.array(r1.cd_solid) + 1e-9)
    ld_1 = np.array(r1.cl_porous) / (np.array(r1.cd_porous) + 1e-9)
    ld_2 = np.array(r2.cl_porous) / (np.array(r2.cd_porous) + 1e-9)
    ax4.plot(r1.angles, ld_s, **style_solid)
    ax4.plot(r1.angles, ld_1, **style_case1)
    ax4.plot(r2.angles, ld_2, **style_case2)
    ax4.set_ylabel("$L/D$"); ax4.set_xlabel("Alpha (deg)")
    ax4.grid(True, alpha=0.5)
    ax4.set_title("Efficiency ($L/D$)")

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(os.path.join(output_dir, "01_Polars.png"), dpi=200)
    plt.close(fig1)

    # --- FIGURE 2: Percentage Change vs AoA ---
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Relative Performance vs. Solid Baseline", fontsize=14)
    
    # Left Plot: % Change in Lift
    ax5.plot(r1.angles, np.array(r1.delta_r_cl)*100, **style_case1)
    ax5.plot(r2.angles, np.array(r2.delta_r_cl)*100, **style_case2)
    ax5.set_ylabel(r"$\Delta C_L$ (%)"); ax5.set_xlabel("Alpha (deg)")
    ax5.grid(True, alpha=0.5); ax5.axhline(0, color='gray', linestyle='--')
    ax5.set_title("Percentage Change in Lift")
    ax5.legend()

    # Right Plot: % Change in Drag
    ax6.plot(r1.angles, np.array(r1.delta_r_cd)*100, **style_case1)
    ax6.plot(r2.angles, np.array(r2.delta_r_cd)*100, **style_case2)
    ax6.set_ylabel(r"$\Delta C_D$ (%)"); ax6.set_xlabel("Alpha (deg)")
    ax6.grid(True, alpha=0.5); ax6.axhline(0, color='gray', linestyle='--')
    ax6.set_title("Percentage Change in Drag")
    ax6.legend()

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "02_Percentage_Changes.png"), dpi=200)
    plt.close(fig2)
# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Ensure output strictly uses the folder the script is in
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SWEEP_OUT_DIR = os.path.join(SCRIPT_DIR, "aoa_sweep_results")

    CONFIG = {
        'AOA_RANGE': np.arange(-5.0, 10.1, 1.0),
        'CP_ANGLES': [-5.0, 5.0, 10.0],
        
        # Case 1 Radii Config
        'C1_RIN': 10000e-6,
        'C1_ROUT': 12000e-6,
        
        # Case 2 Radii Config
        'C2_RIN': 10000e-6,
        'C2_ROUT':8000e-6
    }

    # Run sweeps 
    res1 = run_oo_sweep(
        CONFIG['AOA_RANGE'], 
        CONFIG['C1_RIN'], 
        CONFIG['C1_ROUT'], 
        "Large Ports", 
        SWEEP_OUT_DIR,
        CONFIG['CP_ANGLES']
    )
    
    res2 = run_oo_sweep(
        CONFIG['AOA_RANGE'], 
        CONFIG['C2_RIN'], 
        CONFIG['C2_ROUT'], 
        "Small Ports", 
        SWEEP_OUT_DIR,
        CONFIG['CP_ANGLES']
    )

    # Output global comparison data
    save_sweep_summary(res1, res2, SWEEP_OUT_DIR)
    plot_polars(res1, res2, SWEEP_OUT_DIR)
    
    # Stack the generated Cp images into single pages
    print("\n-> Stacking captured Cp images...")
    stack_case_images(res1, SWEEP_OUT_DIR)
    stack_case_images(res2, SWEEP_OUT_DIR)
    
    print(f"\n-> Completed. Main results saved to: {SWEEP_OUT_DIR}")