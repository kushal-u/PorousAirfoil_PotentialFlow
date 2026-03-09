# =========================
# pipe_network_optim_stage1.py
# =========================
"""
Run-occasionally script:
- Searches over PipeDesign + (optionally) CouplingParams/porous_k to find good settings
- Writes a small JSON config (default: configs/best_design.json)
- Does NOT need to run every time

Communication contract:
- Produces JSON that pipe_network_report_case.py can read.

No circular imports:
- Imports iter1 (core)
- Re-implements a small sweep loop locally (so optimizer doesn't have to import report script)
  (If you prefer, you CAN import report.run_sweep, but keeping optimizer standalone is safer.)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

import iter1


def sweep_for_design(
    aoa_list: np.ndarray,
    airfoil: Tuple[float, float, float, int],
    ref_aoa: float,
    porous_k: int,
    design: iter1.PipeDesign,
    flow: iter1.FlowParams,
    coupling: iter1.CouplingParams,
) -> pd.DataFrame:
    m, p, t, n = airfoil

    X, Y = iter1.naca4(m=m, p=p, t=t, n=n)
    aero = iter1.PanelMethod(X, Y, aoa_deg=ref_aoa, flow=flow)

    solid_ref = aero.solve(v_leak=np.zeros(aero.N))
    cp_ref = np.asarray(solid_ref["cp"])
    x_mid = np.asarray(solid_ref["x_mid"])
    y_mid = aero.y_mid

    net, porous_idx = iter1.generate_spine_manifold_network_fixed_ports(
        x_mid=x_mid, y_mid=y_mid, cp_ref=cp_ref, porous_k=porous_k, design=design, flow=flow
    )
    internal = iter1.InternalFlowSolver(net)

    rows = []
    warm_v = None
    for aoa in aoa_list:
        case, warm_v = iter1.run_coupled_case(
            aoa_deg=float(aoa),
            aero=aero,
            internal=internal,
            porous_idx=porous_idx,
            flow=flow,
            coupling=coupling,
            warm_v_ports=warm_v,
        )
        rows.append(case.to_summary_row())

    return pd.DataFrame(rows).sort_values("aoa_deg").reset_index(drop=True)


def objective(df: pd.DataFrame, target_aoa: float, w_cd: float, w_leak: float) -> float:
    # minimize: -CL + w_cd*CD + w_leak*|leak|
    aoa = df["aoa_deg"].to_numpy()
    i = int(np.argmin(np.abs(aoa - target_aoa)))
    cl = float(df.loc[i, "cl_porous"])
    cd = float(df.loc[i, "cd_porous"])
    leak = float(abs(df.loc[i, "total_leak"]))
    return (-cl) + w_cd * cd + w_leak * leak


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", type=str, default="results_optim_stage1")
    ap.add_argument("--save_best", type=str, default="configs/best_design.json")

    # sweep
    ap.add_argument("--aoa_min", type=float, default=0.0)
    ap.add_argument("--aoa_max", type=float, default=10.0)
    ap.add_argument("--aoa_n", type=int, default=11)
    ap.add_argument("--ref_aoa", type=float, default=2.0)

    # airfoil
    ap.add_argument("--m", type=float, default=0.02)
    ap.add_argument("--p", type=float, default=0.4)
    ap.add_argument("--t", type=float, default=0.12)
    ap.add_argument("--n_pts", type=int, default=240)

    # flow
    ap.add_argument("--re", type=float, default=1e6)
    ap.add_argument("--rho", type=float, default=1.225)
    ap.add_argument("--mu", type=float, default=1.8e-5)
    ap.add_argument("--chord", type=float, default=1.0)

    # coupling (kept fixed in stage-1, but you can grid these too)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--relax", type=float, default=0.03)
    ap.add_argument("--v_clip_min", type=float, default=0.0)
    ap.add_argument("--v_clip_max", type=float, default=5.0)

    # porous selection
    ap.add_argument("--porous_k", type=int, default=24)

    # optimization objective
    ap.add_argument("--target_aoa", type=float, default=6.0)
    ap.add_argument("--w_cd", type=float, default=2.0)
    ap.add_argument("--w_leak", type=float, default=0.5)

    # design search grid
    ap.add_argument("--r_in_min", type=float, default=1.5e-3)
    ap.add_argument("--r_in_max", type=float, default=4.0e-3)
    ap.add_argument("--r_in_n", type=int, default=6)

    ap.add_argument("--r_spine_min", type=float, default=2.5e-3)
    ap.add_argument("--r_spine_max", type=float, default=6.0e-3)
    ap.add_argument("--r_spine_n", type=int, default=6)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_best) or ".", exist_ok=True)

    aoa_list = np.linspace(args.aoa_min, args.aoa_max, args.aoa_n)
    airfoil = (args.m, args.p, args.t, args.n_pts)

    flow = iter1.FlowParams(rho=args.rho, mu=args.mu, re=args.re, chord=args.chord)
    coupling = iter1.CouplingParams(
        max_iter=args.max_iter,
        tol=args.tol,
        relax=args.relax,
        v_clip_min=args.v_clip_min,
        v_clip_max=args.v_clip_max,
    )

    r_in_grid = np.linspace(args.r_in_min, args.r_in_max, args.r_in_n)
    r_sp_grid = np.linspace(args.r_spine_min, args.r_spine_max, args.r_spine_n)

    records: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for r_in in r_in_grid:
        for r_sp in r_sp_grid:
            design = iter1.PipeDesign(r_in=float(r_in), r_out=float(r_in), r_spine=float(r_sp))
            df = sweep_for_design(
                aoa_list=aoa_list,
                airfoil=airfoil,
                ref_aoa=args.ref_aoa,
                porous_k=args.porous_k,
                design=design,
                flow=flow,
                coupling=coupling,
            )

            J = objective(df, args.target_aoa, args.w_cd, args.w_leak)

            # store a few diagnostics at target aoa
            aoa = df["aoa_deg"].to_numpy()
            i = int(np.argmin(np.abs(aoa - args.target_aoa)))
            rec = {
                "r_in": float(r_in),
                "r_spine": float(r_sp),
                "objective": float(J),
                "cl_target": float(df.loc[i, "cl_porous"]),
                "cd_target": float(df.loc[i, "cd_porous"]),
                "leak_target": float(df.loc[i, "total_leak"]),
            }
            records.append(rec)

            if best is None or J < best["objective"]:
                best = {
                    "objective": float(J),
                    "design": {"r_in": float(r_in), "r_out": float(r_in), "r_spine": float(r_sp)},
                    "porous_k": int(args.porous_k),
                    "coupling": {
                        "max_iter": int(args.max_iter),
                        "tol": float(args.tol),
                        "relax": float(args.relax),
                        "v_clip_min": float(args.v_clip_min),
                        "v_clip_max": float(args.v_clip_max),
                    },
                    "meta": {
                        "target_aoa": float(args.target_aoa),
                        "w_cd": float(args.w_cd),
                        "w_leak": float(args.w_leak),
                        "saved_utc": datetime.now(timezone.utc).isoformat(),
                    },
                }

    rank = pd.DataFrame(records).sort_values("objective").reset_index(drop=True)
    rank.to_csv(os.path.join(args.outdir, "ranking.csv"), index=False)

    if best is not None:
        with open(args.save_best, "w") as f:
            json.dump(best, f, indent=2)
        with open(os.path.join(args.outdir, "best_design.json"), "w") as f:
            json.dump(best, f, indent=2)

    print(f"Saved ranking to: {os.path.join(args.outdir, 'ranking.csv')}")
    if best is not None:
        print(f"Saved best config to: {args.save_best}")
        print("Best objective:", best["objective"])
        print("Best design:", best["design"])


if __name__ == "__main__":
    main()
