# run.py
import warnings
warnings.filterwarnings("ignore")
import os
from input import Config, AirfoilGenerator
from plotter import Visualizer


from solver import run_core_anderson  # <-- Anderson version

def run_simulation():
    cfg = Config()
    print(f"--- SIMULATION START: NACA {cfg.AIRFOIL_NAME} ---")
    print(f"   Network topology: {getattr(cfg, 'NETWORK_TOPOLOGY', 'spine')}")

    X, Y = AirfoilGenerator.generate_naca4(cfg.AIRFOIL_NAME, cfg.N_PANELS)

    results = run_core_anderson(X, Y, cfg)

    print(f"-> Result: CL Solid={results['CL_solid']:.4f} -> Porous={results['CL']:.4f}")

    viz = Visualizer(cfg)
    viz.save_csv(
        results["aero"], results["Cp"], results["Cp_solid"], results["V_leakage"],
        results["CL"], results["CL_solid"], results["CD"], results["CD_solid"]
    )
    viz.plot_all(
        results["aero_solid"], results["aero"], results["net"],
        results["Cp"], results["Cp_solid"], results["P_nodes"]
    )

    


if __name__ == "__main__":
    run_simulation()
