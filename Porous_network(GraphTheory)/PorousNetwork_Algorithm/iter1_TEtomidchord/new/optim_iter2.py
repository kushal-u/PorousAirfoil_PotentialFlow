#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import math
import csv
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np

# IMPORTANT: iter2.py must be in the same directory (or in PYTHONPATH)
import iter2


# ==============================================================================
# 1) Small helper: internal solver with tunable AREA_MULT (no changes to iter2.py)
# ==============================================================================

class InternalFlowSolverWithArea(iter2.InternalFlowSolverSparseLU):
    def __init__(self, *args, area_mult: float = 50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._area_mult = float(area_mult)

    def area_surface(self, panel_id: int) -> float:
        # same idea as your iter2, but tunable
        return float(self._area_mult * np.pi * self.surface_pore_radius**2)


# ==============================================================================
# 2) One design evaluation (fast): run only a few AoAs, no plotting
# ==============================================================================

def evaluate_design(
    cfg_base: iter2.SimConfig,
    design: iter2.DesignRadii,
    area_mult: float,
    aoas: List[float] = [0.0, 5.0, 10.0],
    max_iter: int = 80,
) -> Dict[str, float]:
    """
    Returns metrics + objective for a single candidate design.

    Objective (minimize):
      J = -mean(CL/CD over aoas) + penalties

    Penalties:
      + 5 if any AoA doesn't converge
      + small penalty if porous CD is worse than solid on average
    """

    # clone cfg with new design + smaller max_iter for optimizer speed
    cfg = iter2.SimConfig(
        airfoil=cfg_base.airfoil,
        flow=cfg_base.flow,
        porous=cfg_base.porous,
        design=design,
        coupling=iter2.CouplingConfig(
            max_iter=int(max_iter),
            relax=cfg_base.coupling.relax,
            tol=cfg_base.coupling.tol,
            clip=cfg_base.coupling.clip,
            min_iters=cfg_base.coupling.min_iters,
        ),
        plot=iter2.PlotConfig(make_figures=False),
        output=cfg_base.output,
        sweep=iter2.SweepConfig(do_sweep=False),
    )

    # geometry + aero cached per evaluation (fast enough for minimal optimizer)
    geom = iter2.make_geometry(cfg)
    aero = iter2.PanelMethodAero(geom, cfg.flow)

    # accumulators
    eff_list = []
    conv_all = True
    cd_delta_list = []

    for alpha in aoas:
        aero.set_alpha_deg(alpha)

        V_INF = aero.V_INF
        q_inf = 0.5 * cfg.flow.rho * (V_INF**2)

        # Solid baseline
        Cp_solid = aero.solve(np.zeros(aero.N))
        CL_solid, CD_solid, _, _ = iter2.compute_forces_like_iter1(Cp_solid, aero)

        # Network uses baseline pressure
        P_solid = cfg.flow.p_inf + q_inf * Cp_solid
        builder = iter2.SpinePipeBuilder(cfg.porous, cfg.design, cfg.flow.mu, k_spine=iter2.SPINE_NODES)
        build_res = builder.build(aero, P_solid)

        internal = InternalFlowSolverWithArea(
            build_res.G,
            mu=cfg.flow.mu,
            surface_pore_radius=cfg.porous.surface_pore_radius,
            area_mult=area_mult,
        )

        coupling = iter2.AdaptiveRelaxedClippedCoupling(
            relax_init=cfg.coupling.relax,
            clip=cfg.coupling.clip,
            relax_min=1e-5,
            relax_max=0.02,
            grow=1.03,
            shrink=0.5,
        )

        # Coupling loop
        V_leak = np.zeros(aero.N)
        Cp = Cp_solid.copy()
        converged = False

        for it in range(int(cfg.coupling.max_iter)):
            Cp = aero.solve(V_leak)
            P_ext = coupling.external_pressure(Cp, q_inf=q_inf, p_inf=cfg.flow.p_inf)
            P_map = coupling.boundary_map(P_ext, build_res.porous_panels)
            V_calc, _ = internal.solve(P_map)
            V_leak, max_diff = coupling.update_leakage(V_leak, V_calc, build_res.porous_panels)

            if (max_diff < cfg.coupling.tol) and (it >= cfg.coupling.min_iters):
                converged = True
                break

        if not converged:
            conv_all = False

        CL_p, CD_p, _, _ = iter2.compute_forces_like_iter1(Cp, aero)
        eff = CL_p / (CD_p + 1e-12)

        eff_list.append(float(eff))
        cd_delta_list.append(float(CD_p - CD_solid))

    mean_eff = float(np.mean(eff_list))
    mean_cd_delta = float(np.mean(cd_delta_list))

    # Objective: maximize mean_eff => minimize -mean_eff
    J = -mean_eff

    # penalty if not converged for any AoA
    if not conv_all:
        J += 5.0

    # gentle penalty if porous increases drag vs solid
    J += 2.0 * max(0.0, mean_cd_delta)

    return {
        "objective": float(J),
        "mean_eff": mean_eff,
        "mean_cd_delta": mean_cd_delta,
        "converged_all": float(1.0 if conv_all else 0.0),
        "area_mult": float(area_mult),
        "r_spine": float(design.r_spine),
        "r_branch_in": float(design.r_branch_in),
        "r_branch_out": float(design.r_branch_out),
    }


# ==============================================================================
# 3) Minimal but effective optimizer
#    - random sampling in log space around your baseline
#    - keeps code short and robust
# ==============================================================================

def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Sample from [lo, hi] log-uniform."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def optimize(
    cfg_base: iter2.SimConfig,
    n_trials: int = 40,
    seed: int = 7,
    aoas: List[float] = [0.0, 5.0, 10.0],
    out_csv: str = "optim_results.csv",
) -> Dict[str, float]:
    """
    Runs a small random/log search and writes a CSV.
    Returns the best row dict.
    """
    rng = np.random.default_rng(seed)

    base = cfg_base.design
    base_area_mult_guess = 50.0  # your current hardcoded value

    # Search ranges (reasonable starting ranges)
    # AREA_MULT has huge effect; keep broad
    area_lo, area_hi = 1.0, 300.0

    # Radii search ranges around baseline (log-uniform scaling)
    # Keep conservative to avoid extreme r^4 explosions.
    spine_lo, spine_hi = max(1e-4, base.r_spine * 0.4), base.r_spine * 1.6
    bin_lo, bin_hi = max(1e-4, base.r_branch_in * 0.4), base.r_branch_in * 1.6
    bout_lo, bout_hi = max(1e-4, base.r_branch_out * 0.4), base.r_branch_out * 1.6

    rows: List[Dict[str, float]] = []
    best = None

    # Ensure output path
    out_path = os.path.join(cfg_base.output.out_dir, out_csv)
    os.makedirs(cfg_base.output.out_dir, exist_ok=True)

    t0 = time.time()
    for k in range(n_trials):
        # sample candidate
        area_mult = log_uniform(rng, area_lo, area_hi)
        r_spine = log_uniform(rng, spine_lo, spine_hi)

        # optional: also vary branches
        r_bin = log_uniform(rng, bin_lo, bin_hi)
        r_bout = log_uniform(rng, bout_lo, bout_hi)

        design = iter2.DesignRadii(r_branch_in=r_bin, r_branch_out=r_bout, r_spine=r_spine)

        try:
            metrics = evaluate_design(
                cfg_base=cfg_base,
                design=design,
                area_mult=area_mult,
                aoas=aoas,
                max_iter=min(80, int(cfg_base.coupling.max_iter)),
            )
        except Exception as e:
            # fail-safe: assign very bad score
            metrics = {
                "objective": 1e9,
                "mean_eff": -1e9,
                "mean_cd_delta": 1e9,
                "converged_all": 0.0,
                "area_mult": float(area_mult),
                "r_spine": float(r_spine),
                "r_branch_in": float(r_bin),
                "r_branch_out": float(r_bout),
            }

        metrics["trial"] = float(k)

        rows.append(metrics)
        if (best is None) or (metrics["objective"] < best["objective"]):
            best = metrics

        print(
            f"[{k+1:03d}/{n_trials}] J={metrics['objective']:.4f} "
            f"eff={metrics['mean_eff']:.3f} conv={int(metrics['converged_all'])} "
            f"AREA={metrics['area_mult']:.2f} "
            f"sp={metrics['r_spine']:.5f} bin={metrics['r_branch_in']:.5f} bout={metrics['r_branch_out']:.5f}"
        )

    # Write CSV
    headers = ["trial", "objective", "mean_eff", "mean_cd_delta", "converged_all",
               "area_mult", "r_spine", "r_branch_in", "r_branch_out"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})

    dt = time.time() - t0
    print(f"\nSaved optimization history: {out_path}")
    print(f"Elapsed: {dt:.1f} s")

    assert best is not None
    print("\n=== BEST DESIGN FOUND ===")
    print(f"Objective: {best['objective']:.6f}")
    print(f"Mean efficiency (CL/CD): {best['mean_eff']:.6f}")
    print(f"Converged all AoAs: {bool(int(best['converged_all']))}")
    print("Params:")
    print(f"  AREA_MULT    = {best['area_mult']:.6f}")
    print(f"  r_spine      = {best['r_spine']:.6f}")
    print(f"  r_branch_in  = {best['r_branch_in']:.6f}")
    print(f"  r_branch_out = {best['r_branch_out']:.6f}")

    return best


# ==============================================================================
# 4) Main entry
# ==============================================================================

def main():
    # Use your same baseline config defaults, but turn off plots/output heavy work
    cfg = iter2.SimConfig(
        design=iter2.DesignRadii(
            r_branch_in=0.0019590275837019517,
            r_branch_out=0.005243981588587115,
            r_spine=0.008,
        ),
        plot=iter2.PlotConfig(make_figures=False),
        output=iter2.OutputConfig(out_dir="porous_airfoil_results_iter2"),
        coupling=iter2.CouplingConfig(
            max_iter=100,
            relax=0.01,
            tol=1e-8,
            clip=80.0,
            min_iters=6,
        ),
        sweep=iter2.SweepConfig(do_sweep=False),
    )

    # Minimal set of AoAs to optimize against (fast)
    aoas = [0.0, 5.0, 10.0]

    # Run optimizer
    optimize(
        cfg_base=cfg,
        n_trials=40,
        seed=7,
        aoas=aoas,
        out_csv="optim_results.csv",
    )


if __name__ == "__main__":
    main()
