#!/usr/bin/env python3
"""
pipe_network_report_case.py  (SWEEP-ONLY VERSION)

What this script does (only sweep-level outputs)
------------------------------------------------
1) Builds ONE fixed pipe-manifold network (NO plenum) from a reference AoA solid Cp.
2) Sweeps AoA from aoa_min..aoa_max.
3) For each AoA, computes:
   - Solid baseline: Cp_solid, CL_solid, CD_solid
   - Porous (coupled): Cp_porous, CL_porous, CD_porous, leakage stats
4) Saves ONLY sweep-level files (no per-AoA folders):
   - sweep_summary.csv
   - sweep_CL_vs_AoA.png              (solid vs porous)
   - sweep_CD_vs_AoA.png              (solid vs porous)
   - sweep_CLCD_vs_AoA.png            (solid vs porous)
   - sweep_leak_rms_vs_AoA.png        (porous only)
   - sweep_leak_max_vs_AoA.png        (porous only)
   - Cp_comparison_3AoA.png           (Cp solid vs porous at 3 AoAs)

Reuses your existing code (iter1.py)
------------------------------------
- iter1.naca4
- iter1.PanelMethod
- iter1.InternalFlowSolver
- constants: MU, PORE_RADIUS, RHO, V_INF, P_INF, AIRFOIL_NAME, N_PANELS, RELAXATION, CONVERGENCE_TOL

Usage (PowerShell)
------------------
& C:/Users/kusha/anaconda3/python.exe `
"c:/.../pipe_network_report_case.py" `
--r_branch_in 0.0019590275837019517 `
--r_branch_out 0.005243981588587115 `
--r_spine 0.008 `
--out porous_pipe_sweep_only `
--aoa_min -5 --aoa_max 10 --aoa_step 1 `
--ref_aoa 6 `
--cp_aoas "-5,2,10"

Notes
-----
- Network topology (which panels are porous ports + spine layout) is FIXED using ref_aoa.
  That makes the sweep comparable and much faster.
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iter1  # your existing script


# ==============================================================================
# Pipe-network design
# ==============================================================================
@dataclass(frozen=True)
class PipeDesign:
    r_branch_in: float
    r_branch_out: float
    r_spine: float


# ==============================================================================
# Small utilities
# ==============================================================================
def _poiseuille_cond(radius: float, mu: float, length: float) -> float:
    return (math.pi * radius**4) / (8.0 * mu * (length + 1e-15))


def forces_from_cp(aero: "iter1.PanelMethod", Cp: np.ndarray) -> Tuple[float, float]:
    fx = -Cp * aero.nx * aero.L
    fy = -Cp * aero.ny * aero.L
    Fx = float(np.sum(fx))
    Fy = float(np.sum(fy))
    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    return CL, CD


def make_aoa_list(aoa_min: float, aoa_max: float, aoa_step: float) -> List[float]:
    n = int(round((aoa_max - aoa_min) / aoa_step)) + 1
    return [aoa_min + i * aoa_step for i in range(n)]


def parse_cp_aoas(s: str) -> List[float]:
    # accepts "-5,2,10" or "-5 2 10"
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    return [float(p) for p in parts]


def save_xy_plot(
    x: List[float],
    y1: List[float],
    y2: List[float] | None,
    *,
    label1: str,
    label2: str | None,
    xlabel: str,
    ylabel: str,
    title: str,
    path: str,
):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, marker="o", label=label1)
    if y2 is not None and label2 is not None:
        plt.plot(x, y2, marker="o", label=label2)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if y2 is not None and label2 is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# ==============================================================================
# Network generator (pipe manifold, no plenum) with fixed-port selection
# ==============================================================================
def generate_spine_manifold_network_fixed_ports(
    xc: np.ndarray,
    yc: np.ndarray,
    cp_solid_ref: np.ndarray,
    mu: float,
    design: PipeDesign,
    *,
    # topology (same as your stage1 unless you change it)
    n_spine: int = 9,
    spine_x0: float = 0.15,
    spine_x1: float = 0.95,
    spine_y: float = 0.0,
    # port selection (same as stage1)
    n_inlets: int = 30,
    n_outlets: int = 12,
    inlet_x0: float = 0.02,
    inlet_x1: float = 0.25,
    outlet_x0: float = 0.80,
) -> Tuple[nx.Graph, List[int]]:
    """
    Builds a manifold network and selects porous ports ONCE using cp_solid_ref.
    """
    G = nx.Graph()

    # spine nodes
    spine_ids: List[int] = []
    xs = np.linspace(spine_x0, spine_x1, n_spine)
    for k, x in enumerate(xs):
        nid = 100000 + k
        spine_ids.append(nid)
        G.add_node(nid, pos=(float(x), float(spine_y)), type="internal")

    # spine edges
    for a, b in zip(spine_ids[:-1], spine_ids[1:]):
        pa = np.array(G.nodes[a]["pos"], float)
        pb = np.array(G.nodes[b]["pos"], float)
        L = float(np.linalg.norm(pa - pb))
        c = _poiseuille_cond(design.r_spine, mu, L)
        G.add_edge(a, b, length=L, cond=c, radius=float(design.r_spine), type="spine")

    # select ports ONCE from reference Cp
    inlet_candidates = [i for i in range(len(xc)) if (yc[i] < 0.0 and inlet_x0 <= xc[i] <= inlet_x1)]
    outlet_candidates = [i for i in range(len(xc)) if (yc[i] > 0.0 and xc[i] >= outlet_x0)]

    inlet_candidates.sort(key=lambda i: cp_solid_ref[i], reverse=True)
    outlet_candidates.sort(key=lambda i: xc[i], reverse=True)

    selected_in = inlet_candidates[:max(1, n_inlets)]
    selected_out = outlet_candidates[:max(1, n_outlets)]
    porous_pids = selected_in + selected_out

    spine_pos = np.array([G.nodes[n]["pos"] for n in spine_ids], float)

    def nearest_spine_node(xp: float, yp: float) -> int:
        p = np.array([xp, yp], float)
        d2 = np.sum((spine_pos - p) ** 2, axis=1)
        return spine_ids[int(np.argmin(d2))]

    def add_surface_port(pid: int):
        if not G.has_node(pid):
            G.add_node(pid, pos=(float(xc[pid]), float(yc[pid])), type="boundary", panel_idx=int(pid))

    def connect_branch(pid: int, spine_node: int, radius: float, etype: str):
        p_port = np.array(G.nodes[pid]["pos"], float)
        p_sp = np.array(G.nodes[spine_node]["pos"], float)
        L = float(np.linalg.norm(p_port - p_sp))
        c = _poiseuille_cond(radius, mu, L)
        G.add_edge(pid, spine_node, length=L, cond=c, radius=float(radius), type=etype)

    for pid in selected_in:
        add_surface_port(pid)
        spn = nearest_spine_node(xc[pid], yc[pid])
        connect_branch(pid, spn, design.r_branch_in, "branch_in")

    for pid in selected_out:
        add_surface_port(pid)
        spn = nearest_spine_node(xc[pid], yc[pid])
        connect_branch(pid, spn, design.r_branch_out, "branch_out")

    return G, porous_pids


# ==============================================================================
# One AoA solve: solid + porous coupled
# ==============================================================================
def run_one_aoa(
    aoa_deg: float,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    porous_pids: List[int],
    internal_solver: "iter1.InternalFlowSolver",
    max_iter: int,
    relax: float,
    tol: float,
    vclip: float,
    warm_V: np.ndarray | None,
) -> Dict[str, object]:
    aero = iter1.PanelMethod(X, Y, aoa_deg)

    # solid baseline
    Cp_solid = aero.solve(np.zeros(aero.N))
    CL_solid, CD_solid = forces_from_cp(aero, Cp_solid)

    # porous coupled
    V_leak = np.zeros(aero.N, float) if warm_V is None else warm_V.copy()
    q_inf = 0.5 * iter1.RHO * (iter1.V_INF ** 2)
    Cp_porous = Cp_solid.copy()

    for it in range(max_iter):
        Cp_porous = aero.solve(V_leak)

        P_ext = iter1.P_INF + q_inf * Cp_porous
        P_map = {pid: float(P_ext[pid]) for pid in porous_pids}

        V_calc, _P_nodes = internal_solver.solve(P_map)

        max_diff = 0.0
        for pid, vcalc in V_calc.items():
            vold = V_leak[pid]
            vnew = relax * float(vcalc) + (1.0 - relax) * vold
            vnew = max(min(vnew, vclip), -vclip)
            max_diff = max(max_diff, abs(vnew - vold))
            V_leak[pid] = vnew

        if max_diff < tol and it > 5:
            break

    CL_porous, CD_porous = forces_from_cp(aero, Cp_porous)

    if porous_pids:
        vals = np.array([V_leak[i] for i in porous_pids], float)
        leak_rms = float(np.sqrt(np.mean(vals**2)))
        leak_max = float(np.max(np.abs(vals)))
    else:
        leak_rms, leak_max = 0.0, 0.0

    return {
        "aero": aero,                  # kept in case you need geometry data
        "Cp_solid": Cp_solid,
        "Cp_porous": Cp_porous,
        "V_leak": V_leak,
        "CL_solid": float(CL_solid),
        "CD_solid": float(CD_solid),
        "CL_porous": float(CL_porous),
        "CD_porous": float(CD_porous),
        "leak_rms": float(leak_rms),
        "leak_max": float(leak_max),
    }


# ==============================================================================
# Cp comparison plot at 3 AoAs
# ==============================================================================
def save_cp_comparison_plot(
    *,
    out_path: str,
    aoas: List[float],
    XC: np.ndarray,
    cp_solid_by_aoa: Dict[float, np.ndarray],
    cp_porous_by_aoa: Dict[float, np.ndarray],
):
    plt.figure(figsize=(10, 8))
    for aoa in aoas:
        CpS = cp_solid_by_aoa[aoa]
        CpP = cp_porous_by_aoa[aoa]
        plt.plot(XC, CpS, linestyle="--", label=f"Solid Cp @ {aoa:+.1f}°")
        plt.plot(XC, CpP, linestyle="-",  label=f"Porous Cp @ {aoa:+.1f}°")

    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c (panel midpoints)")
    plt.ylabel("Cp")
    plt.title("Cp comparison (Solid vs Porous Pipe Network) at 3 AoAs")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


# ==============================================================================
# Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()

    # radii (required)
    ap.add_argument("--r_branch_in", type=float, required=True)
    ap.add_argument("--r_branch_out", type=float, required=True)
    ap.add_argument("--r_spine", type=float, required=True)

    # sweep
    ap.add_argument("--aoa_min", type=float, default=-5.0)
    ap.add_argument("--aoa_max", type=float, default=10.0)
    ap.add_argument("--aoa_step", type=float, default=1.0)

    # fixed network selection
    ap.add_argument("--ref_aoa", type=float, default=6.0)

    # coupling
    ap.add_argument("--max_iter", type=int, default=80)
    ap.add_argument("--relax", type=float, default=iter1.RELAXATION)
    ap.add_argument("--tol", type=float, default=iter1.CONVERGENCE_TOL)
    ap.add_argument("--vclip", type=float, default=80.0)

    # outputs
    ap.add_argument("--out", type=str, default="porous_pipe_sweep_only")

    # Cp comparisons at 3 AoAs
    ap.add_argument("--cp_aoas", type=str, default="-5,2,10",
                    help='Three AoAs for Cp comparison, e.g. "-5,2,10"')

    args = ap.parse_args()

    design = PipeDesign(
        r_branch_in=float(args.r_branch_in),
        r_branch_out=float(args.r_branch_out),
        r_spine=float(args.r_spine),
    )

    aoa_list = make_aoa_list(args.aoa_min, args.aoa_max, args.aoa_step)
    cp_aoas = parse_cp_aoas(args.cp_aoas)
    if len(cp_aoas) != 3:
        raise ValueError("--cp_aoas must contain exactly 3 AoAs, e.g. -5,2,10")

    # ensure cp_aoas are in the sweep (or at least computable)
    # We'll compute them even if not in aoa_list by just running them too.
    # But for convenience, include them if missing.
    extra_cp_aoas = [a for a in cp_aoas if a not in aoa_list]

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    print("\n=== PIPE NETWORK SWEEP (SWEEP-LEVEL OUTPUTS ONLY) ===")
    print(f"Design radii: {asdict(design)}")
    print(f"AoA sweep: {aoa_list}")
    if extra_cp_aoas:
        print(f"Extra Cp AoAs not in sweep (will still compute): {extra_cp_aoas}")
    print(f"Reference AoA for fixed ports: {args.ref_aoa}")
    print(f"Coupling: max_iter={args.max_iter}, relax={args.relax}, tol={args.tol}, vclip={args.vclip}")
    print(f"Output root: {os.path.abspath(out_root)}\n")

    # geometry once
    X, Y = iter1.naca4(iter1.AIRFOIL_NAME, n_panels=iter1.N_PANELS)

    # build fixed network from ref AoA solid Cp
    print("-> Building fixed pipe network from ref_aoa solid Cp ...")
    aero_ref = iter1.PanelMethod(X, Y, float(args.ref_aoa))
    Cp_solid_ref = aero_ref.solve(np.zeros(aero_ref.N))
    G, porous_pids = generate_spine_manifold_network_fixed_ports(
        aero_ref.XC, aero_ref.YC, Cp_solid_ref, iter1.MU, design
    )
    internal_solver = iter1.InternalFlowSolver(G, mu=iter1.MU, surface_pore_radius=iter1.PORE_RADIUS)
    print(f"   Network: nodes={G.number_of_nodes()} edges={G.number_of_edges()} porous_ports={len(porous_pids)}")

    # sweep storage
    rows: List[Dict[str, float]] = []
    warm_V = None

    aoas_to_run = aoa_list[:]  # main sweep
    # also compute missing cp_aoas if they aren't in sweep, but DON'T include them in sweep plots/csv
    # (they'll only be used for the Cp comparison plot)
    aoas_for_cp_only = extra_cp_aoas[:]

    # For Cp comparison plot we need Cp arrays for these 3 AoAs
    cp_solid_by_aoa: Dict[float, np.ndarray] = {}
    cp_porous_by_aoa: Dict[float, np.ndarray] = {}

    # ---- main sweep ----
    for aoa in tqdm(aoas_to_run, desc="AoA sweep", unit="deg"):
        res = run_one_aoa(
            float(aoa),
            X=X, Y=Y,
            porous_pids=porous_pids,
            internal_solver=internal_solver,
            max_iter=int(args.max_iter),
            relax=float(args.relax),
            tol=float(args.tol),
            vclip=float(args.vclip),
            warm_V=warm_V,
        )
        warm_V = res["V_leak"]

        CLs = float(res["CL_solid"]);  CDs = float(res["CD_solid"])
        CLp = float(res["CL_porous"]); CDp = float(res["CD_porous"])

        rows.append({
            "aoa": float(aoa),
            "CL_solid": CLs,
            "CD_solid": CDs,
            "CLCD_solid": float(CLs / (abs(CDs) + 1e-12)),
            "CL_porous": CLp,
            "CD_porous": CDp,
            "CLCD_porous": float(CLp / (abs(CDp) + 1e-12)),
            "dCL": float(CLp - CLs),
            "dCD": float(CDp - CDs),
            "dCLCD": float((CLp / (abs(CDp) + 1e-12)) - (CLs / (abs(CDs) + 1e-12))),
            "leak_rms": float(res["leak_rms"]),
            "leak_max": float(res["leak_max"]),
        })

        # store Cp if it's one of the requested Cp-compare AoAs
        if aoa in cp_aoas:
            cp_solid_by_aoa[aoa] = res["Cp_solid"]
            cp_porous_by_aoa[aoa] = res["Cp_porous"]

        print(
            f"   AoA {aoa:+.1f}: "
            f"Solid CL/CD={CLs/(abs(CDs)+1e-12):.3f} | "
            f"Porous CL/CD={CLp/(abs(CDp)+1e-12):.3f} | "
            f"leak_rms={res['leak_rms']:.3f}"
        )

    # ---- compute extra Cp-only AoAs (if any) ----
    for aoa in aoas_for_cp_only:
        res = run_one_aoa(
            float(aoa),
            X=X, Y=Y,
            porous_pids=porous_pids,
            internal_solver=internal_solver,
            max_iter=int(args.max_iter),
            relax=float(args.relax),
            tol=float(args.tol),
            vclip=float(args.vclip),
            warm_V=None,  # don't warm-start these
        )
        cp_solid_by_aoa[aoa] = res["Cp_solid"]
        cp_porous_by_aoa[aoa] = res["Cp_porous"]

    # ensure we have all 3 Cp AoAs stored
    missing = [a for a in cp_aoas if a not in cp_solid_by_aoa]
    if missing:
        raise RuntimeError(f"Failed to compute Cp for requested AoAs: {missing}")

    # ---- save sweep CSV ----
    csv_path = os.path.join(out_root, "sweep_summary.csv")
    cols = [
        "aoa",
        "CL_solid", "CD_solid", "CLCD_solid",
        "CL_porous", "CD_porous", "CLCD_porous",
        "dCL", "dCD", "dCLCD",
        "leak_rms", "leak_max",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n-> Saved: {csv_path}")

    # ---- build arrays for plots ----
    aoas = [r["aoa"] for r in rows]
    CLs = [r["CL_solid"] for r in rows]
    CDs = [r["CD_solid"] for r in rows]
    CLCDs = [r["CLCD_solid"] for r in rows]

    CLp = [r["CL_porous"] for r in rows]
    CDp = [r["CD_porous"] for r in rows]
    CLCDp = [r["CLCD_porous"] for r in rows]

    leak_rms = [r["leak_rms"] for r in rows]
    leak_max = [r["leak_max"] for r in rows]

    # ---- sweep plots (SOLID vs POROUS) ----
    save_xy_plot(
        aoas, CLs, CLp,
        label1="Solid", label2="Porous",
        xlabel="AoA [deg]", ylabel="CL",
        title="CL vs AoA (Solid vs Porous Pipe Network)",
        path=os.path.join(out_root, "sweep_CL_vs_AoA.png"),
    )
    save_xy_plot(
        aoas, CDs, CDp,
        label1="Solid", label2="Porous",
        xlabel="AoA [deg]", ylabel="CD",
        title="CD vs AoA (Solid vs Porous Pipe Network)",
        path=os.path.join(out_root, "sweep_CD_vs_AoA.png"),
    )
    save_xy_plot(
        aoas, CLCDs, CLCDp,
        label1="Solid", label2="Porous",
        xlabel="AoA [deg]", ylabel="CL/CD",
        title="CL/CD vs AoA (Solid vs Porous Pipe Network)",
        path=os.path.join(out_root, "sweep_CLCD_vs_AoA.png"),
    )

    # ---- leakage plots (POROUS only) ----
    save_xy_plot(
        aoas, leak_rms, None,
        label1="Porous leak_rms", label2=None,
        xlabel="AoA [deg]", ylabel="Leakage RMS [m/s]",
        title="Leakage RMS vs AoA (Porous Pipe Network)",
        path=os.path.join(out_root, "sweep_leak_rms_vs_AoA.png"),
    )
    save_xy_plot(
        aoas, leak_max, None,
        label1="Porous leak_max", label2=None,
        xlabel="AoA [deg]", ylabel="Leakage max |v| [m/s]",
        title="Leakage max vs AoA (Porous Pipe Network)",
        path=os.path.join(out_root, "sweep_leak_max_vs_AoA.png"),
    )

    # ---- Cp comparison plot at 3 AoAs ----
    cp_plot_path = os.path.join(out_root, "Cp_comparison_3AoA.png")
    save_cp_comparison_plot(
        out_path=cp_plot_path,
        aoas=cp_aoas,
        XC=aero_ref.XC,  # geometry is same across AoA
        cp_solid_by_aoa=cp_solid_by_aoa,
        cp_porous_by_aoa=cp_porous_by_aoa,
    )
    print(f"-> Saved: {cp_plot_path}")

    # ---- quick text summary ----
    txt_path = os.path.join(out_root, "sweep_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Pipe Network Sweep Summary (Solid vs Porous)\n")
        f.write("===========================================\n\n")
        f.write(f"Design radii: {asdict(design)}\n")
        f.write(f"AoA sweep: {aoa_list}\n")
        f.write(f"ref_aoa (fixed ports): {args.ref_aoa}\n")
        f.write(f"cp_aoas (Cp compare): {cp_aoas}\n\n")
        f.write(f"mean(CL/CD) solid:  {float(np.mean(CLCDs)):.6f}\n")
        f.write(f"mean(CL/CD) porous: {float(np.mean(CLCDp)):.6f}\n")
        f.write(f"mean(dCL)   : {float(np.mean([r['dCL'] for r in rows])):.6f}\n")
        f.write(f"mean(dCD)   : {float(np.mean([r['dCD'] for r in rows])):.6f}\n")
        f.write(f"mean(leak_rms): {float(np.mean(leak_rms)):.6f}\n")
    print(f"-> Saved: {txt_path}")

    print("\n✅ DONE. Sweep-level outputs are in:", os.path.abspath(out_root))


if __name__ == "__main__":
    main()
