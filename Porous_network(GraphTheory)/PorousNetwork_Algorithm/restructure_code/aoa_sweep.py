import os
import numpy as np

from input import Config, AirfoilGenerator
from solver import PanelMethod, PorousNetwork, MonolithicCoupledSolverAnderson, compute_forces
from h5_logger import H5AoASweepLogger


def run_sweep_to_h5(h5_path: str, aoa_values):
    cfg = Config()

    # geometry once
    X, Y = AirfoilGenerator.generate_naca4(cfg.AIRFOIL_NAME, cfg.N_PANELS)

    with H5AoASweepLogger(h5_path, overwrite=True, compression="gzip", comp_level=1) as log:
        for alpha in aoa_values:
            cfg.ANGLE_OF_ATTACK = float(alpha)

            aero = PanelMethod(X, Y, cfg)

            # solid baseline
            Cp_solid = aero.solve(np.zeros(aero.N))
            CL_solid, CD_solid = compute_forces(aero, Cp_solid)

            # build porous network
            net = PorousNetwork(aero, Cp_solid, cfg)

            # coupled Anderson solve
            coupled = MonolithicCoupledSolverAnderson(aero, net, cfg, v_clip=80.0)
            v0 = np.zeros(len(net.active_pores), dtype=float)

            gname = log.init_aoa_group(alpha_deg=float(alpha), aero=aero, net=net)

            def cb(k, v_active, g_inf, q, gamma, Cp, P_nodes):
                if P_nodes is None:
                    P_nodes = np.zeros(len(list(net.G.nodes())), dtype=np.float64)
                log.append_iter(
                    gname=gname,
                    k=int(k),
                    g_inf=float(g_inf),
                    v_active=v_active,
                    q=q,
                    gamma=float(gamma),
                    Cp=Cp,
                    P_nodes=P_nodes
                )

            print(f"\n[AoA {alpha:+.2f}] -> Solving coupled system with Anderson + HDF5 logging...")
            v_active, Cp, P_nodes = coupled.solve(v0=v0, verbose=True, callback=cb)

            # final convenience write
            log.write_final(gname, v_active=v_active, q=aero.q, gamma=aero.gamma, Cp=Cp, P_nodes=P_nodes)

            CL, CD = compute_forces(aero, Cp)
            print(f"[AoA {alpha:+.2f}] CL solid={CL_solid:.4f} -> porous={CL:.4f} | CD solid={CD_solid:.5f} -> porous={CD:.5f}")

    print(f"\nSaved HDF5 sweep history to:\n  {h5_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(here, "aoa_sweep_history.h5")
    aoa = np.arange(-5.0, 10.0 + 1e-9, 1.0)
    run_sweep_to_h5(h5_path, aoa)


if __name__ == "__main__":
    main()
