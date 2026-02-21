"""
aoa_sweep.py

Rewrite of your AoA sweep to match the *new* code structure:

- Uses:
    from input import Config, AirfoilGenerator
    from solver import PanelMethod, PorousNetwork, MonolithicCoupledSolverAnderson, compute_forces
    from plotter import Visualizer

- Coupling is done via MonolithicCoupledSolverAnderson (no manual MAX_ITER / RELAXATION loop).
- Creates a fresh PanelMethod for each AoA so freestream projections stay consistent.
- Captures plots for selected angles into per-case subfolders.

Run:
    python aoa_sweep.py
"""

import os
import csv
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from input import Config, AirfoilGenerator
from solver import PanelMethod, PorousNetwork, MonolithicCoupledSolverAnderson, compute_forces
from plotter import Visualizer

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================
@dataclass
class SweepResult:
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
# 2. UTILS
# ==============================================================================
def export_cp_distribution_csv(
    alpha_deg: float,
    Cp_solid: np.ndarray,
    Cp_porous: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    aero: PanelMethod,
    out_dir: str,
    fname: str = "cp_distribution.csv",
) -> str:
    """
    Export Cp distribution (solid + porous) to CSV with panel midpoints computed from X,Y.

    Columns:
      panel_id, x0, y0, x1, y1, x_mid, y_mid, s, L, nx, ny, Cp_solid, Cp_porous, dCp
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()

    Np = len(X) - 1
    if len(Cp_solid) != Np:
        Np = min(len(Cp_solid), len(X) - 1)

    x0, y0 = X[:Np], Y[:Np]
    x1, y1 = X[1 : Np + 1], Y[1 : Np + 1]
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)

    L = np.asarray(aero.L).flatten()[:Np]
    nx = np.asarray(aero.nx).flatten()[:Np]
    ny = np.asarray(aero.ny).flatten()[:Np]

    s = np.cumsum(L) - 0.5 * L

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# AoA_deg={alpha_deg}"])
        w.writerow(
            [
                "panel_id",
                "x0",
                "y0",
                "x1",
                "y1",
                "x_mid",
                "y_mid",
                "s",
                "L",
                "nx",
                "ny",
                "Cp_solid",
                "Cp_porous",
                "dCp_porous_minus_solid",
            ]
        )

        for i in range(Np):
            w.writerow(
                [
                    i,
                    float(x0[i]),
                    float(y0[i]),
                    float(x1[i]),
                    float(y1[i]),
                    float(x_mid[i]),
                    float(y_mid[i]),
                    float(s[i]),
                    float(L[i]),
                    float(nx[i]),
                    float(ny[i]),
                    float(Cp_solid[i]),
                    float(Cp_porous[i]),
                    float(Cp_porous[i] - Cp_solid[i]),
                ]
            )

    return out_path


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s).strip("_")


# ==============================================================================
# 3. CORE SWEEP
# ==============================================================================
def run_aoa_sweep(
    angles: Sequence[float],
    r_inlet: float,
    r_outlet: float,
    label: str,
    base_out_dir: str,
    capture_angles: Optional[Sequence[float]] = None,
) -> SweepResult:
    """
    Runs AoA sweep with your new solver structure:
    - Solid baseline via PanelMethod.solve(V=0)
    - Porous coupled via PorousNetwork + MonolithicCoupledSolverAnderson
    - Optional plot capture at chosen AoAs
    """
    if capture_angles is None:
        capture_angles = []

    print(f"\n--- STARTING SWEEP: {label} (R_in={r_inlet*1000:.1f}mm, R_out={r_outlet*1000:.1f}mm) ---")

    cfg = Config()
    cfg.PORE_RADIUS_INLET = float(r_inlet)
    cfg.PORE_RADIUS_OUTLET = float(r_outlet)

    res = SweepResult(name=label, radius_inlet=r_inlet, radius_outlet=r_outlet)

    # Geometry is independent of AoA; generate once
    X, Y = AirfoilGenerator.generate_naca4(cfg.AIRFOIL_NAME, cfg.N_PANELS)

    # We will create a fresh PanelMethod per AoA so freestream projections are consistent.
    # (Angle is read in PanelMethod.__init__)
    pbar = tqdm(list(angles), desc=f"Simulating {label}", unit="deg")

    for alpha in pbar:
        cfg.ANGLE_OF_ATTACK = float(alpha)

        # --- SOLID BASELINE ---
        aero_solid = PanelMethod(X, Y, cfg)
        Cp_solid = aero_solid.solve(np.zeros(aero_solid.N, dtype=float))
        CL_s, CD_s = compute_forces(aero_solid, Cp_solid)

        # --- POROUS COUPLED ---
        # Use a second aero instance for porous (so q/gamma correspond to porous solution)
        aero_porous = PanelMethod(X, Y, cfg)

        topology = getattr(cfg, "NETWORK_TOPOLOGY", "spine")
        net = PorousNetwork(aero_porous, Cp_solid, cfg, topology=topology)

        coupled = MonolithicCoupledSolverAnderson(aero_porous, net, cfg, v_clip=80.0)
        v0 = np.zeros(len(net.active_pores), dtype=float)

        v_active, Cp_porous, P_nodes = coupled.solve(
            v0=v0,
            tol=cfg.CONVERGENCE_TOL,
            maxiter=cfg.ANDERSON_MAXITER,
            m=cfg.ANDERSON_M,
            beta=cfg.ANDERSON_BETA,
            verbose=False,
        )

        # Expand leakage velocities to panel array for saving/plotting
        V_leakage = np.zeros(aero_porous.N, dtype=float)
        if len(net.active_pores) > 0:
            V_leakage[np.array(net.active_pores, dtype=int)] = np.clip(v_active, -80.0, 80.0)

        CL_p, CD_p = compute_forces(aero_porous, Cp_porous)

        # --- STORE ---
        res.angles.append(float(alpha))
        res.cl_solid.append(float(CL_s))
        res.cd_solid.append(float(CD_s))
        res.cl_porous.append(float(CL_p))
        res.cd_porous.append(float(CD_p))
        res.delta_cl.append(float(CL_p - CL_s))
        res.delta_r_cl.append(float((CL_p - CL_s) / (abs(CL_s) + 1e-12)))
        res.delta_r_cd.append(float((CD_p - CD_s) / (abs(CD_s) + 1e-12)))

        pbar.set_postfix({"CL_Solid": f"{CL_s:.2f}", "CL_Porous": f"{CL_p:.2f}"})

        # --- CAPTURE PLOTS ---
        if float(alpha) in [float(a) for a in capture_angles]:
            cl_change = ((CL_p - CL_s) / (abs(CL_s) + 1e-12)) * 100
            cd_change = ((CD_p - CD_s) / (abs(CD_s) + 1e-12)) * 100

            msg = (
                f"\n  >>> RESULTS FOR AoA = {alpha}° <<<\n"
                f"      CL: {CL_s:.5f} (Solid) -> {CL_p:.5f} (Porous) | Change: {cl_change:+.2f}%\n"
                f"      CD: {CD_s:.5f} (Solid) -> {CD_p:.5f} (Porous) | Change: {cd_change:+.2f}%"
            )
            pbar.write(msg)

            safe_label = _safe_name(label)
            sub_folder = os.path.join(base_out_dir, f"{safe_label}_AoA_{alpha:g}")
            os.makedirs(sub_folder, exist_ok=True)

            # Important: Visualizer uses cfg.OUTPUT_DIR relative to script dir,
            # so we set it to a relative subfolder name rather than an absolute path.
            # If you prefer absolute output, modify Visualizer accordingly.
            cfg_cap = Config()
            cfg_cap.__dict__.update(cfg.__dict__)
            cfg_cap.OUTPUT_DIR = os.path.relpath(sub_folder, start=os.path.dirname(os.path.abspath(__file__)))

            viz = Visualizer(cfg_cap)
            viz.save_csv(aero_porous, Cp_porous, Cp_solid, V_leakage, CL_p, CL_s, CD_p, CD_s)
            viz.plot_all(aero_solid, aero_porous, net, Cp_porous, Cp_solid, P_nodes)

            export_cp_distribution_csv(
                alpha_deg=float(alpha),
                Cp_solid=Cp_solid,
                Cp_porous=Cp_porous,
                X=X,
                Y=Y,
                aero=aero_porous,
                out_dir=sub_folder,
                fname="cp_distribution.csv",
            )

            target_img = os.path.join(sub_folder, "01_Geometry_Cp.png")
            res.capture_image_paths.append(target_img)

    return res


# ==============================================================================
# 4. IMAGE STACKER & OUTPUT
# ==============================================================================
def stack_case_images(res: SweepResult, out_dir: str) -> None:
    if not res.capture_image_paths:
        return

    images = []
    for path in res.capture_image_paths:
        if os.path.exists(path):
            images.append(Image.open(path))

    if not images:
        return

    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)

    stacked_im = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for im in images:
        stacked_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    safe_name = _safe_name(res.name)
    out_file = os.path.join(out_dir, f"{safe_name}_Stacked_Cp_Summary.png")
    stacked_im.save(out_file)
    print(f"-> Stacked Cp summary created: {out_file}")


def save_sweep_summary(cases: List[SweepResult], output_dir: str, filename: str = "polar_summary.csv"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # Assume same angles list across cases
    a = np.asarray(cases[0].angles, float)

    with open(path, "w") as f:
        f.write("--- POLAR SUMMARY ---\n")

        header = "Alpha_deg,CL_Solid,CD_Solid"
        for c in cases:
            header += f",{c.name}_CL,{c.name}_CD,{c.name}_DeltaCL,{c.name}_PctChangeCL,{c.name}_PctChangeCD"
        header += "\n"
        f.write(header)

        for i in range(len(a)):
            cl_s = cases[0].cl_solid[i]
            cd_s = cases[0].cd_solid[i]
            line = f"{a[i]:.2f},{cl_s:.6f},{cd_s:.6f}"

            for c in cases:
                clp = c.cl_porous[i]
                cdp = c.cd_porous[i]
                dcl = clp - cl_s
                pcl = 100.0 * dcl / (abs(cl_s) + 1e-12)
                pcd = 100.0 * (cdp - cd_s) / (abs(cd_s) + 1e-12)
                line += f",{clp:.6f},{cdp:.6f},{dcl:.6f},{pcl:.2f},{pcd:.2f}"
            line += "\n"
            f.write(line)

    print(f"-> Saved: {path}")


def plot_polars(cases: List[SweepResult], output_dir: str, filename_prefix: str = "01"):
    os.makedirs(output_dir, exist_ok=True)

    style_solid = dict(color="gray", linestyle="--", linewidth=1.8, label="Solid Baseline")
    markers = ["o", "s", "D", "^", "v", "x"]

    aoa = np.asarray(cases[0].angles, float)
    cl_solid = np.asarray(cases[0].cl_solid, float)
    cd_solid = np.asarray(cases[0].cd_solid, float)

    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle("Aerodynamic Polars Comparison", fontsize=16)

    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # CL vs AoA
    ax1.plot(aoa, cl_solid, **style_solid)
    for k, c in enumerate(cases):
        ax1.plot(aoa, c.cl_porous, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
    ax1.set_title("Lift coefficient vs AoA")
    ax1.set_xlabel("AoA (deg)")
    ax1.set_ylabel("CL")
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    # CD vs AoA
    ax2.plot(aoa, cd_solid, **style_solid)
    for k, c in enumerate(cases):
        ax2.plot(aoa, c.cd_porous, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
    ax2.set_title("Drag coefficient vs AoA")
    ax2.set_xlabel("AoA (deg)")
    ax2.set_ylabel("CD")
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    # Drag polar
    ax3.plot(cd_solid, cl_solid, **style_solid)
    for k, c in enumerate(cases):
        ax3.plot(c.cd_porous, c.cl_porous, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
    ax3.set_title("Drag polar")
    ax3.set_xlabel("CD")
    ax3.set_ylabel("CL")
    ax3.grid(True, alpha=0.4)

    # L/D vs AoA
    ld_s = cl_solid / (cd_solid + 1e-12)
    ax4.plot(aoa, ld_s, **style_solid)
    for k, c in enumerate(cases):
        clp = np.asarray(c.cl_porous, float)
        cdp = np.asarray(c.cd_porous, float)
        ldp = clp / (cdp + 1e-12)
        ax4.plot(aoa, ldp, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
    ax4.set_title("Efficiency (L/D) vs AoA")
    ax4.set_xlabel("AoA (deg)")
    ax4.set_ylabel("CL/CD")
    ax4.grid(True, alpha=0.4)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(os.path.join(output_dir, f"{filename_prefix}_Polars.png"), dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Percent change plots
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Relative Performance vs Solid Baseline", fontsize=14)

    for k, c in enumerate(cases):
        dcl_pct = 100.0 * np.asarray(c.delta_r_cl, float)
        dcd_pct = 100.0 * np.asarray(c.delta_r_cd, float)
        ax5.plot(aoa, dcl_pct, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
        ax6.plot(aoa, dcd_pct, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)

    ax5.set_title("Percentage change in CL")
    ax5.set_xlabel("AoA (deg)")
    ax5.set_ylabel("ΔCL (%)")
    ax5.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax5.grid(True, alpha=0.4)
    ax5.legend()

    ax6.set_title("Percentage change in CD")
    ax6.set_xlabel("AoA (deg)")
    ax6.set_ylabel("ΔCD (%)")
    ax6.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax6.grid(True, alpha=0.4)
    ax6.legend()

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"{filename_prefix}_Percentage_Changes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)


# ==============================================================================
# 5. MAIN
# ==============================================================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SWEEP_OUT_DIR = os.path.join(SCRIPT_DIR, "aoa_sweep_results")
    os.makedirs(SWEEP_OUT_DIR, exist_ok=True)

    CONFIG = {
        "AOA_RANGE": np.arange(-5.0, 10.1, 1.0),
        "CP_ANGLES": [-5.0, 5.0, 10.0],
        "C1_RIN": 10000e-6,
        "C1_ROUT": 12000e-6,
        "C2_RIN": 10000e-6,
        "C2_ROUT": 8000e-6,
    }

    res1 = run_aoa_sweep(
        CONFIG["AOA_RANGE"],
        CONFIG["C1_RIN"],
        CONFIG["C1_ROUT"],
        "Large Ports",
        SWEEP_OUT_DIR,
        capture_angles=CONFIG["CP_ANGLES"],
    )

    res2 = run_aoa_sweep(
        CONFIG["AOA_RANGE"],
        CONFIG["C2_RIN"],
        CONFIG["C2_ROUT"],
        "Small Ports",
        SWEEP_OUT_DIR,
        capture_angles=CONFIG["CP_ANGLES"],
    )

    cases = [res1, res2]
    save_sweep_summary(cases, SWEEP_OUT_DIR)
    plot_polars(cases, SWEEP_OUT_DIR)

    print("\n-> Stacking captured Cp images...")
    stack_case_images(res1, SWEEP_OUT_DIR)
    stack_case_images(res2, SWEEP_OUT_DIR)

    print(f"\n-> Completed. Main results saved to: {SWEEP_OUT_DIR}")