import numpy as np

from input import Config, AirfoilGenerator
from solver import PanelMethod, PorousNetwork
from plotter import Visualizer


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

    # Snapshot object for solid external field plotting
    aero_solid = PanelMethod(X, Y, cfg)
    _ = aero_solid.solve(np.zeros(aero_solid.N))

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
        # External Aerodynamics
        Cp = aero.solve(V_leakage)
        P_ext = cfg.P_INF + (0.5 * cfg.RHO * cfg.V_INF**2) * Cp
        P_map = {pid: P_ext[pid] for pid in net.active_pores}

        # Internal Pipe Network
        vel_calc, P_nodes = net.solve_flow(P_map)

        # Update & Relax
        max_diff = 0.0
        for pid, v_target in vel_calc.items():
            v_relaxed = cfg.RELAXATION * v_target + (1 - cfg.RELAXATION) * V_leakage[pid]
            diff = abs(v_relaxed - V_leakage[pid])
            max_diff = max(max_diff, diff)
            V_leakage[pid] = v_relaxed

        if max_diff < cfg.CONVERGENCE_TOL and i > 5:
            print(f"   Converged at Iter {i}")
            break
        if i % 10 == 0:
            print(f"   Iter {i}: Resid={max_diff:.6f}")

    # 5. Final Forces
    fx = -Cp * aero.nx * aero.L
    fy = -Cp * aero.ny * aero.L
    Fx, Fy = np.sum(fx), np.sum(fy)
    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)

    print(f"-> Result: CL Solid={CL_solid:.4f} -> Porous={CL:.4f}")

    # Physical Sanity Check
    max_v_transpiration = np.max(np.abs(V_leakage))
    if max_v_transpiration > cfg.V_INF:
        print(f"\n   [WARNING] Max transpiration velocity ({max_v_transpiration:.2f} m/s) exceeds freestream ({cfg.V_INF:.2f} m/s).")
        print("             Small-perturbation potential flow assumptions may be invalid.")

    # 6. Output
    viz.save_csv(aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid)
    viz.plot_all(aero_solid, aero, net, Cp, Cp_solid, P_nodes)


if __name__ == "__main__":
    run_simulation()