# optimise.py
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import csv
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from input import Config, AirfoilGenerator
from solver import PanelMethod
from network import PorousNetwork
from plotter import Visualizer


# ==============================================================================
# GLOBAL SETTINGS
# ==============================================================================
AOA_SWEEP = list(range(-5, 13))   # -5, -4, ..., 12
REF_AOA = 6.0

CASE_2P1CH_DIR = "results_2p1ch_sweep_opt"
CASE_4P2CH_DIR = "results_4p2ch_sweep_opt"


# ==============================================================================
# HELPERS
# ==============================================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_history_csv(path: str, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_sweep_csv(path: str, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def get_naca_thickness(x, thickness_ratio=0.18):
    """Exact local thickness of NACA 0018-style 4-digit thickness law."""
    t = thickness_ratio
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    return 2.0 * yt


def soft_invalid_score(base=-2.0, extra=0.0):
    return base - extra


def compute_force_coefficients(aero, Cp):
    fx = -Cp * aero.nx * aero.L
    fy = -Cp * aero.ny * aero.L
    Fx = np.sum(fx)
    Fy = np.sum(fy)

    CL = Fy * np.cos(aero.alpha) - Fx * np.sin(aero.alpha)
    CD = Fx * np.cos(aero.alpha) + Fy * np.sin(aero.alpha)
    return CL, CD


def build_candidate_surface_ids(aero):
    """
    Candidate panel IDs on top and bottom surfaces.
    Avoid exact LE/TE closure corners.
    """
    top_ids = np.where((aero.YC > 0.0) & (aero.XC >= 0.03) & (aero.XC <= 0.98))[0]
    bot_ids = np.where((aero.YC < 0.0) & (aero.XC >= 0.03) & (aero.XC <= 0.98))[0]

    if len(top_ids) == 0 or len(bot_ids) == 0:
        raise RuntimeError("Failed to construct candidate surface panel lists.")

    return top_ids.astype(int), bot_ids.astype(int)


def decode_surface_and_panel(selector_val, surface_flag_val, top_ids, bot_ids):
    """
    surface_flag_val < 0.5  => top
    surface_flag_val >= 0.5 => bottom
    """
    use_top = surface_flag_val < 0.5
    surface_ids = top_ids if use_top else bot_ids

    max_idx = len(surface_ids) - 1
    sel_idx = int(np.clip(np.round(selector_val), 0, max_idx))
    pid = int(surface_ids[sel_idx])

    surface_name = "top" if use_top else "bottom"
    return surface_name, sel_idx, pid


def compute_efficiency(CL, CD):
    if not np.isfinite(CL) or not np.isfinite(CD) or CD <= 0.0:
        return np.nan
    return CL / (CD + 1e-12)

def de_objective_2p1ch(x):
    return -evaluate_2p1ch_design_over_sweep(x)


def de_objective_4p2ch(x):
    return -evaluate_4p2ch_design_over_sweep(x)

# ==============================================================================
# CACHED GEOMETRY (AOA-INDEPENDENT)
# ==============================================================================
@lru_cache(maxsize=1)
def get_geometry_base():
    cfg = Config()
    X, Y = AirfoilGenerator.generate_naca4(cfg.AIRFOIL_NAME, cfg.N_PANELS)
    # Build once to obtain XC/YC geometry and candidate surface ids
    aero = PanelMethod(X, Y, cfg)
    top_surface_ids, bot_surface_ids = build_candidate_surface_ids(aero)

    return {
        "X": X,
        "Y": Y,
        "top_surface_ids": top_surface_ids,
        "bot_surface_ids": bot_surface_ids,
    }


def make_config_for_aoa(aoa_deg: float, output_dir: str | None = None):
    cfg = Config()
    cfg.ANGLE_OF_ATTACK = float(aoa_deg)
    if output_dir is not None:
        cfg.OUTPUT_DIR = output_dir
    return cfg


def make_aero_for_aoa(aoa_deg: float, output_dir: str | None = None):
    geom = get_geometry_base()
    cfg = make_config_for_aoa(aoa_deg, output_dir=output_dir)
    aero = PanelMethod(geom["X"], geom["Y"], cfg)
    return cfg, aero


# ==============================================================================
# COUPLED SOLVER
# ==============================================================================
def run_coupled_case(aero, cfg, net, max_iter=None):
    if max_iter is None:
        max_iter = cfg.MAX_ITER

    V_leakage = np.zeros(aero.N)
    P_nodes = None
    fluxes = None
    net_flux = None
    converged = False

    q = None
    gamma = None
    Cp = None
    last_max_diff = np.inf

    for i in range(max_iter):
        try:
            q, gamma, Cp = aero.solve(V_leakage)
        except RuntimeError:
            return {"valid": False, "reason": "aero_solve_failed"}

        if Cp is None or not np.all(np.isfinite(Cp)):
            return {"valid": False, "reason": "invalid_cp"}

        P_ext = cfg.P_INF + 0.5 * cfg.RHO * (cfg.V_INF ** 2) * Cp
        P_map = {pid: P_ext[pid] for pid in net.active_pores}

        try:
            velocities, P_nodes, fluxes, net_flux = net.solve_flow(P_map)
        except RuntimeError:
            return {"valid": False, "reason": "network_solve_failed"}

        max_diff = 0.0
        for pid, v_target in velocities.items():
            if not np.isfinite(v_target):
                return {"valid": False, "reason": "nonfinite_network_velocity"}

            v_target = np.clip(v_target, -3.0 * cfg.V_INF, 3.0 * cfg.V_INF)
            v_relaxed = cfg.RELAXATION * v_target + (1.0 - cfg.RELAXATION) * V_leakage[pid]
            max_diff = max(max_diff, abs(v_relaxed - V_leakage[pid]))
            V_leakage[pid] = v_relaxed

        last_max_diff = max_diff
        if max_diff < cfg.CONVERGENCE_TOL:
            converged = True
            break

    if q is None or gamma is None or Cp is None:
        return {"valid": False, "reason": "solver_state_missing"}

    CL, CD = compute_force_coefficients(aero, Cp)
    if not np.isfinite(CL) or not np.isfinite(CD):
        return {"valid": False, "reason": "invalid_force_coefficients"}

    return {
        "valid": True,
        "converged": converged,
        "iterations": i + 1,
        "last_max_diff": last_max_diff,
        "q": q,
        "gamma": gamma,
        "Cp": Cp,
        "CL": CL,
        "CD": CD,
        "CL_CD": compute_efficiency(CL, CD),
        "V_leakage": V_leakage,
        "P_nodes": P_nodes,
        "fluxes": fluxes,
        "net_flux": net_flux,
    }


# ==============================================================================
# CASE EVALUATORS (ONE AOA)
# ==============================================================================
def evaluate_solid_at_aoa(aoa_deg: float):
    cfg, aero = make_aero_for_aoa(aoa_deg)
    q, gamma, Cp = aero.solve(np.zeros(aero.N))
    CL, CD = compute_force_coefficients(aero, Cp)
    return {
        "case": "solid",
        "aoa_deg": aoa_deg,
        "valid": True,
        "converged": True,
        "iterations": 1,
        "net_flux": 0.0,
        "CL": CL,
        "CD": CD,
        "CL_CD": compute_efficiency(CL, CD),
        "Cp": Cp,
        "q": q,
        "gamma": gamma,
        "aero": aero,
        "cfg": cfg,
        "V_leakage": np.zeros(aero.N),
        "P_nodes": None,
        "net": None,
    }


def evaluate_2p1ch_at_aoa(aoa_deg: float, design: dict):
    cfg, aero = make_aero_for_aoa(aoa_deg)
    net = PorousNetwork(aero, None, cfg, auto_build=False)
    pore_specs = [(p["panel_id"], p["radius"]) for p in design["pores"]]
    net.build_from_pores(pore_specs, design["x_plenum"])

    result = run_coupled_case(aero, cfg, net, max_iter=cfg.MAX_ITER)
    result["case"] = "2p1ch"
    result["aoa_deg"] = aoa_deg
    result["aero"] = aero
    result["cfg"] = cfg
    result["net"] = net
    return result


def evaluate_4p2ch_at_aoa(aoa_deg: float, design: dict):
    cfg, aero = make_aero_for_aoa(aoa_deg)
    net = PorousNetwork(aero, None, cfg, auto_build=False)
    pore_specs = [(p["panel_id"], p["radius"], p["chamber_id"]) for p in design["pores"]]
    net.build_from_two_chambers(pore_specs, design["xA"], design["xB"], design["r_link"])

    result = run_coupled_case(aero, cfg, net, max_iter=cfg.MAX_ITER)
    result["case"] = "4p2ch"
    result["aoa_deg"] = aoa_deg
    result["aero"] = aero
    result["cfg"] = cfg
    result["net"] = net
    return result


# ==============================================================================
# SWEEP RUNNER
# ==============================================================================
def run_aoa_sweep(case_name: str, design: dict | None, aoa_list=None, verbose=True):
    if aoa_list is None:
        aoa_list = AOA_SWEEP

    rows = []

    for aoa in aoa_list:
        if case_name == "solid":
            res = evaluate_solid_at_aoa(aoa)
        elif case_name == "2p1ch":
            res = evaluate_2p1ch_at_aoa(aoa, design)
        elif case_name == "4p2ch":
            res = evaluate_4p2ch_at_aoa(aoa, design)
        else:
            raise ValueError(f"Unknown case_name: {case_name}")

        row = {
            "case": case_name,
            "aoa_deg": aoa,
            "valid": bool(res.get("valid", False)),
            "converged": bool(res.get("converged", False)),
            "iterations": int(res.get("iterations", -1)),
            "net_flux": float(res.get("net_flux", np.nan)) if res.get("net_flux", None) is not None else np.nan,
            "CL": float(res.get("CL", np.nan)),
            "CD": float(res.get("CD", np.nan)),
            "CL_CD": float(res.get("CL_CD", np.nan)),
        }
        rows.append(row)

        if verbose:
            print(
                f"AoA = {aoa:>3} deg | Case = {case_name:6s} | "
                f"CL = {row['CL']:.6f} | CD = {row['CD']:.6f} | "
                f"CL/CD = {row['CL_CD']:.6f} | converged = {row['converged']}"
            )

    return rows


def summarize_sweep(rows: list[dict]):
    valid_rows = [r for r in rows if r["valid"] and r["converged"] and np.isfinite(r["CL_CD"])]
    n_total = len(rows)
    n_valid = len(valid_rows)

    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": 0,
            "coverage": 0.0,
            "mean_CL_CD": np.nan,
            "peak_CL_CD": np.nan,
            "CL_CD_at_ref": np.nan,
        }

    coverage = n_valid / n_total
    mean_eff = float(np.mean([r["CL_CD"] for r in valid_rows]))
    peak_eff = float(np.max([r["CL_CD"] for r in valid_rows]))

    ref_rows = [r for r in valid_rows if abs(r["aoa_deg"] - REF_AOA) < 1e-12]
    ref_eff = float(ref_rows[0]["CL_CD"]) if ref_rows else np.nan

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "coverage": coverage,
        "mean_CL_CD": mean_eff,
        "peak_CL_CD": peak_eff,
        "CL_CD_at_ref": ref_eff,
    }


def score_from_sweep(rows: list[dict]):
    summary = summarize_sweep(rows)
    if summary["n_valid"] == 0:
        return soft_invalid_score(extra=10.0)

    # mean efficiency penalized by sweep failures
    score = summary["mean_CL_CD"] - 2.0 * (1.0 - summary["coverage"])
    return score


def plot_sweep_curves(rows: list[dict], output_dir: str, prefix: str):
    ensure_dir(output_dir)
    aoa = [r["aoa_deg"] for r in rows]
    CL = [r["CL"] for r in rows]
    CD = [r["CD"] for r in rows]
    E = [r["CL_CD"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(aoa, CL, marker="o")
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("CL")
    plt.grid(alpha=0.3)
    plt.title(f"{prefix}: CL vs AoA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_CL_vs_AoA.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(aoa, CD, marker="o")
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("CD")
    plt.grid(alpha=0.3)
    plt.title(f"{prefix}: CD vs AoA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_CD_vs_AoA.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(aoa, E, marker="o")
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("CL/CD")
    plt.grid(alpha=0.3)
    plt.title(f"{prefix}: CL/CD vs AoA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_CLCD_vs_AoA.png"), dpi=150)
    plt.close()


# ==============================================================================
# CASE 1: 2 PORES / 1 CHAMBER
# ==============================================================================
def decode_2p1ch_vector(x):
    base = get_geometry_base()
    top_ids = base["top_surface_ids"]
    bot_ids = base["bot_surface_ids"]

    pores = []
    for i in range(2):
        off = 3 * i
        surface_flag = float(x[off + 0])
        selector_val = float(x[off + 1])
        radius = float(x[off + 2])

        surface_name, sel_idx, pid = decode_surface_and_panel(selector_val, surface_flag, top_ids, bot_ids)
        pores.append({
            "surface": surface_name,
            "selector_idx": sel_idx,
            "panel_id": pid,
            "radius": radius,
        })

    x_plenum = float(x[6])
    return {"type": "2p1ch", "pores": pores, "x_plenum": x_plenum}


def get_bounds_2p1ch():
    base = get_geometry_base()
    n_top = len(base["top_surface_ids"])
    n_bot = len(base["bot_surface_ids"])
    max_selector = max(n_top - 1, n_bot - 1)

    return [
        (0.0, 1.0),          # pore1 surface flag
        (0.0, max_selector), # pore1 selector
        (0.001, 0.015),      # pore1 radius
        (0.0, 1.0),          # pore2 surface flag
        (0.0, max_selector), # pore2 selector
        (0.001, 0.015),      # pore2 radius
        (0.10, 0.90),        # x_plenum
    ]


def evaluate_2p1ch_design_over_sweep(x):
    geom = get_geometry_base()
    aero_geom_cfg = make_config_for_aoa(REF_AOA)
    aero_geom = PanelMethod(geom["X"], geom["Y"], aero_geom_cfg)

    design = decode_2p1ch_vector(x)
    pores = design["pores"]
    x_plenum = design["x_plenum"]

    panel_ids = [p["panel_id"] for p in pores]
    radii = [p["radius"] for p in pores]

    if len(set(panel_ids)) < 2:
        return soft_invalid_score(extra=1.0)

    if any((not np.isfinite(r)) or (r <= 0.0) for r in radii):
        return soft_invalid_score(extra=1.0)

    # spacing on same surface
    same_surface = pores[0]["surface"] == pores[1]["surface"]
    if same_surface:
        if abs(aero_geom.XC[panel_ids[0]] - aero_geom.XC[panel_ids[1]]) < 0.03:
            return soft_invalid_score(extra=0.5)

    # thickness constraints
    margin = 0.002
    thickness_viol = 0.0
    for pid, r in zip(panel_ids, radii):
        x_loc = aero_geom.XC[pid]
        thickness_viol += max(0.0, (2.0 * r + margin) - get_naca_thickness(x_loc))
    if thickness_viol > 0.0:
        return soft_invalid_score(extra=200.0 * thickness_viol)

    rows = run_aoa_sweep("2p1ch", design, aoa_list=AOA_SWEEP, verbose=False)
    return score_from_sweep(rows)


# ==============================================================================
# CASE 2: 4 PORES / 2 CHAMBERS
# ==============================================================================
def decode_4p2ch_vector(x):
    base = get_geometry_base()
    top_ids = base["top_surface_ids"]
    bot_ids = base["bot_surface_ids"]

    pores = []
    for i in range(4):
        off = 4 * i
        surface_flag = float(x[off + 0])
        selector_val = float(x[off + 1])
        radius = float(x[off + 2])
        chamber_flag = float(x[off + 3])

        surface_name, sel_idx, pid = decode_surface_and_panel(selector_val, surface_flag, top_ids, bot_ids)
        chamber_id = 0 if chamber_flag < 0.5 else 1

        pores.append({
            "surface": surface_name,
            "selector_idx": sel_idx,
            "panel_id": pid,
            "radius": radius,
            "chamber_id": chamber_id,
        })

    xA = float(x[16])
    xB = float(x[17])
    r_link = float(x[18])

    return {
        "type": "4p2ch",
        "pores": pores,
        "xA": xA,
        "xB": xB,
        "r_link": r_link,
    }


def get_bounds_4p2ch():
    base = get_geometry_base()
    n_top = len(base["top_surface_ids"])
    n_bot = len(base["bot_surface_ids"])
    max_selector = max(n_top - 1, n_bot - 1)

    bounds = []
    for _ in range(4):
        bounds.extend([
            (0.0, 1.0),          # surface flag
            (0.0, max_selector), # selector
            (0.001, 0.015),      # pore radius
            (0.0, 1.0),          # chamber flag
        ])

    bounds.extend([
        (0.10, 0.70),  # xA
        (0.20, 0.90),  # xB
        (0.001, 0.020) # r_link
    ])
    return bounds


def evaluate_4p2ch_design_over_sweep(x):
    geom = get_geometry_base()
    aero_geom_cfg = make_config_for_aoa(REF_AOA)
    aero_geom = PanelMethod(geom["X"], geom["Y"], aero_geom_cfg)

    design = decode_4p2ch_vector(x)
    pores = design["pores"]
    xA = design["xA"]
    xB = design["xB"]
    r_link = design["r_link"]

    if xA >= xB - 0.05:
        return soft_invalid_score(extra=10.0 * (xA - (xB - 0.05) + 1e-12))

    if not np.isfinite(r_link) or r_link <= 0.0:
        return soft_invalid_score(extra=1.0)

    panel_ids = [p["panel_id"] for p in pores]
    radii = [p["radius"] for p in pores]
    chamber_ids = [p["chamber_id"] for p in pores]

    if len(set(panel_ids)) < 4:
        return soft_invalid_score(extra=1.0)

    if 0 not in chamber_ids or 1 not in chamber_ids:
        return soft_invalid_score(extra=1.0)

    # spacing
    min_sep_x = 0.02
    for surf in ("top", "bottom"):
        surf_panels = [p["panel_id"] for p in pores if p["surface"] == surf]
        surf_x = sorted([aero_geom.XC[pid] for pid in surf_panels])
        for i in range(len(surf_x) - 1):
            if abs(surf_x[i + 1] - surf_x[i]) < min_sep_x:
                return soft_invalid_score(extra=0.5)

    # thickness
    margin = 0.002
    thickness_viol = 0.0
    for pid, r in zip(panel_ids, radii):
        x_loc = aero_geom.XC[pid]
        thickness_viol += max(0.0, (2.0 * r + margin) - get_naca_thickness(x_loc))
    if thickness_viol > 0.0:
        return soft_invalid_score(extra=200.0 * thickness_viol)

    rows = run_aoa_sweep("4p2ch", design, aoa_list=AOA_SWEEP, verbose=False)
    return score_from_sweep(rows)


# ==============================================================================
# FINAL SAVERS
# ==============================================================================
def save_case_summary(output_dir: str, design: dict, sweep_rows: list[dict], ref_result: dict, case_label: str):
    ensure_dir(output_dir)

    sweep_summary = summarize_sweep(sweep_rows)

    summary_lines = [
        f"Case: {case_label}",
        f"Mean CL/CD over sweep: {sweep_summary['mean_CL_CD']:.6f}",
        f"Peak CL/CD over sweep: {sweep_summary['peak_CL_CD']:.6f}",
        f"CL/CD at {REF_AOA:.1f} deg: {sweep_summary['CL_CD_at_ref']:.6f}",
        f"Converged points: {sweep_summary['n_valid']} / {sweep_summary['n_total']}",
        "",
        f"Reference AoA = {REF_AOA:.1f} deg",
        f"Reference CL: {ref_result['CL']:.6f}",
        f"Reference CD: {ref_result['CD']:.6f}",
        f"Reference CL/CD: {ref_result['CL_CD']:.6f}",
        f"Reference converged: {ref_result['converged']}",
        f"Reference iterations: {ref_result['iterations']}",
    ]

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    save_json(os.path.join(output_dir, "best_design.json"), design)
    save_sweep_csv(os.path.join(output_dir, "aoa_sweep_results.csv"), sweep_rows)
    plot_sweep_curves(sweep_rows, output_dir, prefix=case_label)


def make_reference_plots_and_csv(output_dir: str, ref_result: dict):
    """
    Save the same style of plots you were already generating,
    at REF_AOA only.
    """
    cfg = ref_result["cfg"]
    aero = ref_result["aero"]
    net = ref_result["net"]
    Cp = ref_result["Cp"]
    q = ref_result["q"]
    gamma = ref_result["gamma"]
    V_leakage = ref_result["V_leakage"]

    # Solid baseline at same AoA for comparison
    solid = evaluate_solid_at_aoa(REF_AOA)
    aero_solid = solid["aero"]
    Cp_solid = solid["Cp"]
    q_solid = solid["q"]
    gamma_solid = solid["gamma"]
    CL_solid = solid["CL"]
    CD_solid = solid["CD"]

    cfg.OUTPUT_DIR = output_dir
    viz = Visualizer(cfg)
    viz.save_csv(aero, Cp, Cp_solid, V_leakage, ref_result["CL"], CL_solid, ref_result["CD"], CD_solid)
    if net is not None:
        viz.plot_all(
            aero_solid, aero, net,
            Cp, Cp_solid, ref_result["P_nodes"],
            q_solid, gamma_solid, q, gamma
        )


# ==============================================================================
# OPTIMIZATION DRIVERS
# ==============================================================================
def optimize_case_2p1ch():
    output_dir = CASE_2P1CH_DIR
    ensure_dir(output_dir)

    print("\n========================================================")
    print("--- STARTING SWEEP-BASED EFFICIENCY OPTIMIZATION: 2P / 1CH ---")
    print("Objective: maximize mean(CL/CD) over AoA = -5..12 deg")
    print(f"Output folder: {output_dir}")
    print("========================================================")

    bounds = get_bounds_2p1ch()
    history = []

    def callback(xk, convergence):
        design = decode_2p1ch_vector(xk)
        rows = run_aoa_sweep("2p1ch", design, aoa_list=AOA_SWEEP, verbose=False)
        summary = summarize_sweep(rows)
        score = score_from_sweep(rows)

        history.append({
            "score": score,
            "convergence_metric": convergence,
            "mean_CL_CD": summary["mean_CL_CD"],
            "peak_CL_CD": summary["peak_CL_CD"],
            "CL_CD_at_ref": summary["CL_CD_at_ref"],
            "coverage": summary["coverage"],
            "x_plenum": design["x_plenum"],
        })

        print("\n--------------------------------------------------------")
        print("-> DE generation update [2p / 1ch]")
        print(f"   convergence metric: {convergence:.6e}")
        print(f"   current sweep score: {score:.6f}")
        print(f"   mean(CL/CD): {summary['mean_CL_CD']:.6f}")
        print(f"   peak(CL/CD): {summary['peak_CL_CD']:.6f}")
        print(f"   CL/CD at {REF_AOA:.1f} deg: {summary['CL_CD_at_ref']:.6f}")
        print(f"   converged AoA points: {summary['n_valid']} / {summary['n_total']}")
        print(f"   x_plenum = {design['x_plenum']:.4f}")
        for i, p in enumerate(design["pores"], 1):
            print(
                f"   pore_{i}: {p['surface']:6s} | panel {p['panel_id']:5d} | "
                f"r = {p['radius']*1000:.2f} mm"
            )
        return False

    result = differential_evolution(
        func=de_objective_2p1ch,
        bounds=bounds,
        strategy="randtobest1bin",
        maxiter=60,
        popsize=10,
        tol=0.005,
        mutation=(0.5, 1.2),
        recombination=0.75,
        seed=42,
        callback=callback,
        polish=False,
        updating="deferred",
        workers=12,
        disp=True,
    )

    best_x = result.x
    best_design = decode_2p1ch_vector(best_x)
    best_score = -result.fun

    print("\n--- OPTIMIZATION COMPLETE [2p / 1ch] ---")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Best sweep score: {best_score:.6f}")

    # Final full sweep + reference AoA run
    sweep_rows = run_aoa_sweep("2p1ch", best_design, aoa_list=AOA_SWEEP, verbose=True)
    ref_result = evaluate_2p1ch_at_aoa(REF_AOA, best_design)

    save_case_summary(output_dir, best_design, sweep_rows, ref_result, "2p1ch")
    save_history_csv(os.path.join(output_dir, "optimization_history.csv"), history)
    make_reference_plots_and_csv(output_dir, ref_result)

    return best_design, sweep_rows


def optimize_case_4p2ch():
    output_dir = CASE_4P2CH_DIR
    ensure_dir(output_dir)

    print("\n========================================================")
    print("--- STARTING SWEEP-BASED EFFICIENCY OPTIMIZATION: 4P / 2CH ---")
    print("Objective: maximize mean(CL/CD) over AoA = -5..12 deg")
    print(f"Output folder: {output_dir}")
    print("========================================================")

    bounds = get_bounds_4p2ch()
    history = []

    def callback(xk, convergence):
        design = decode_4p2ch_vector(xk)
        rows = run_aoa_sweep("4p2ch", design, aoa_list=AOA_SWEEP, verbose=False)
        summary = summarize_sweep(rows)
        score = score_from_sweep(rows)

        history.append({
            "score": score,
            "convergence_metric": convergence,
            "mean_CL_CD": summary["mean_CL_CD"],
            "peak_CL_CD": summary["peak_CL_CD"],
            "CL_CD_at_ref": summary["CL_CD_at_ref"],
            "coverage": summary["coverage"],
            "xA": design["xA"],
            "xB": design["xB"],
            "r_link": design["r_link"],
        })

        print("\n--------------------------------------------------------")
        print("-> DE generation update [4p / 2ch]")
        print(f"   convergence metric: {convergence:.6e}")
        print(f"   current sweep score: {score:.6f}")
        print(f"   mean(CL/CD): {summary['mean_CL_CD']:.6f}")
        print(f"   peak(CL/CD): {summary['peak_CL_CD']:.6f}")
        print(f"   CL/CD at {REF_AOA:.1f} deg: {summary['CL_CD_at_ref']:.6f}")
        print(f"   converged AoA points: {summary['n_valid']} / {summary['n_total']}")
        print(f"   xA = {design['xA']:.4f}, xB = {design['xB']:.4f}, r_link = {design['r_link']*1000:.2f} mm")
        for i, p in enumerate(design["pores"], 1):
            ch = "A" if p["chamber_id"] == 0 else "B"
            print(
                f"   pore_{i}: {p['surface']:6s} | panel {p['panel_id']:5d} | "
                f"r = {p['radius']*1000:.2f} mm | chamber {ch}"
            )
        return False

    result = differential_evolution(
        func=de_objective_4p2ch,
        bounds=bounds,
        strategy="randtobest1bin",
        maxiter=80,
        popsize=12,
        tol=0.005,
        mutation=(0.5, 1.2),
        recombination=0.75,
        seed=42,
        callback=callback,
        polish=False,
        updating="deferred",
        workers=12,
        disp=True,
    )

    best_x = result.x
    best_design = decode_4p2ch_vector(best_x)
    best_score = -result.fun

    print("\n--- OPTIMIZATION COMPLETE [4p / 2ch] ---")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Best sweep score: {best_score:.6f}")

    sweep_rows = run_aoa_sweep("4p2ch", best_design, aoa_list=AOA_SWEEP, verbose=True)
    ref_result = evaluate_4p2ch_at_aoa(REF_AOA, best_design)

    save_case_summary(output_dir, best_design, sweep_rows, ref_result, "4p2ch")
    save_history_csv(os.path.join(output_dir, "optimization_history.csv"), history)
    make_reference_plots_and_csv(output_dir, ref_result)

    return best_design, sweep_rows


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("========================================================")
    print("SWEEP-BASED EFFICIENCY OPTIMIZATION")
    print(f"AoA sweep: {AOA_SWEEP[0]} deg to {AOA_SWEEP[-1]} deg")
    print(f"Reference AoA for detailed plots: {REF_AOA:.1f} deg")
    print("Cases:")
    print(f"  1. 2 pores / 1 chamber  -> {CASE_2P1CH_DIR}")
    print(f"  2. 4 pores / 2 chambers -> {CASE_4P2CH_DIR}")
    print("========================================================")

    best_2p1ch, rows_2p1ch = optimize_case_2p1ch()
    best_4p2ch, rows_4p2ch = optimize_case_4p2ch()

    s2 = summarize_sweep(rows_2p1ch)
    s4 = summarize_sweep(rows_4p2ch)

    print("\n========================================================")
    print("FINAL SUMMARY")
    print(f"2p / 1ch  -> mean(CL/CD) = {s2['mean_CL_CD']:.6f}, peak(CL/CD) = {s2['peak_CL_CD']:.6f}")
    print(f"4p / 2ch  -> mean(CL/CD) = {s4['mean_CL_CD']:.6f}, peak(CL/CD) = {s4['peak_CL_CD']:.6f}")
    winner = "2p / 1ch" if s2["mean_CL_CD"] > s4["mean_CL_CD"] else "4p / 2ch"
    print(f"Best mean efficiency winner: {winner}")
    print("========================================================")