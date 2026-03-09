import numpy as np
import warnings

from input import Config, AirfoilGenerator
from solver import PanelMethod
from network import PorousNetwork
from plotter import Visualizer

warnings.filterwarnings("ignore")

# ==============================================================================
# ANDERSON ACCELERATOR 
# ==============================================================================
class AndersonAccelerator:
    """
    Implements Anderson Acceleration (Mixing) to speed up fixed-point iterations.
    Finds coefficients that minimize the history of residuals.
    """
    def __init__(self, window_size=5, beta=0.5):
        self.m = window_size
        self.beta = beta
        self.X = []  # History of states
        self.F = []  # History of residuals: f(x) = G(x) - x

    def apply(self, x_current, g_evaluated):
        f = g_evaluated - x_current
        self.X.append(x_current.copy())
        self.F.append(f.copy())

        # Keep history within window size
        if len(self.X) > self.m:
            self.X.pop(0)
            self.F.pop(0)

        k = len(self.X)
        if k == 1:
            return x_current + self.beta * f

        # Setup Least Squares constraint: ||F * alpha|| -> min, s.t. sum(alpha) = 1
        F_mat = np.column_stack(self.F)
        X_mat = np.column_stack(self.X)

        A = np.zeros((k + 1, k + 1))
        A[:k, :k] = F_mat.T @ F_mat
        A[:k, k] = 1.0
        A[k, :k] = 1.0

        b = np.zeros(k + 1)
        b[k] = 1.0

        try:
            res = np.linalg.solve(A, b)
            alpha = res[:k]
        except np.linalg.LinAlgError:
            # Fallback to simple relaxation if matrix is singular
            return x_current + self.beta * f

        # Next iterate calculation
        x_next = (X_mat @ alpha) + self.beta * (F_mat @ alpha)
        return x_next

# ==============================================================================
# MAIN SIMULATION LOOP
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

    aero_solid = PanelMethod(X, Y, cfg)
    _ = aero_solid.solve(np.zeros(aero_solid.N)) 

    fx = -Cp_solid * aero.nx * aero.L
    fy = -Cp_solid * aero.ny * aero.L
    Fx, Fy = np.sum(fx), np.sum(fy)
    CL_solid = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD_solid = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    print(f"   Baseline CL: {CL_solid:.4f}")

    # 3. Build Network
    print("-> Building Porous Network...")
    net = PorousNetwork(aero, Cp_solid, cfg)
    
    # 4. Iteration Loop with Anderson Acceleration
    V_leakage = np.zeros(aero.N)
    P_nodes = None
    
    anderson = AndersonAccelerator(window_size=4, beta=cfg.RELAXATION)
    
    print(f"-> Iterating (Max {cfg.MAX_ITER})...")
    for i in range(cfg.MAX_ITER):
        # External Aerodynamics
        Cp = aero.solve(V_leakage)
        P_ext = cfg.P_INF + (0.5 * cfg.RHO * cfg.V_INF**2) * Cp
        P_map = {pid: P_ext[pid] for pid in net.active_pores}
        
        # Internal Pipe Network
        vel_calc, P_nodes = net.solve_flow(P_map)
        
        # Map calculated network velocities back to a full aero panel array
        G_V = V_leakage.copy()
        for pid, v_target in vel_calc.items():
            G_V[pid] = v_target
            
        # Apply Anderson Acceleration
        v_next = anderson.apply(V_leakage, G_V)
        
        # Assess convergence
        max_diff = np.max(np.abs(v_next - V_leakage))
        V_leakage = v_next
            
        if max_diff < cfg.CONVERGENCE_TOL and i > 5:
            print(f"   Converged at Iter {i}")
            break
        if i % 10 == 0: 
            print(f"   Iter {i}: Resid={max_diff:.6e}")

    # 5. Final Forces
    fx = -Cp * aero.nx * aero.L
    fy = -Cp * aero.ny * aero.L
    Fx, Fy = np.sum(fx), np.sum(fy)
    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    
    print(f"-> Result: CL Solid={CL_solid:.4f} -> Porous={CL:.4f}")

    max_v_transpiration = np.max(np.abs(V_leakage))
    if max_v_transpiration > cfg.V_INF:
        print(f"\n   Max Leakage velocity ({max_v_transpiration:.2f} m/s) ,freestream ({cfg.V_INF:.2f} m/s).")
        

    # 6. Output
    viz.save_csv(aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid)
    viz.plot_all(aero_solid, aero, net, Cp, Cp_solid, P_nodes)

if __name__ == "__main__":
    run_simulation()