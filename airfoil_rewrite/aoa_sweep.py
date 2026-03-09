# aoa_sweep.py
import os
import json
import csv

import matplotlib.pyplot as plt

from optimise import (
    AOA_SWEEP,
    REF_AOA,
    CASE_2P1CH_DIR,
    CASE_4P2CH_DIR,
    ensure_dir,
    run_aoa_sweep,
    summarize_sweep,
    save_sweep_csv,
)


COMPARE_DIR = "aoa_sweep_compare_results"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def plot_comparison(all_rows, output_dir):
    ensure_dir(output_dir)

    cases = sorted(set(r["case"] for r in all_rows))

    # CL vs AoA
    plt.figure(figsize=(8, 5))
    for case in cases:
        rows = [r for r in all_rows if r["case"] == case]
        plt.plot([r["aoa_deg"] for r in rows], [r["CL"] for r in rows], marker="o", label=case)
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("CL")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("CL vs AoA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CL_vs_AoA.png"), dpi=150)
    plt.close()

    # CD vs AoA
    plt.figure(figsize=(8, 5))
    for case in cases:
        rows = [r for r in all_rows if r["case"] == case]
        plt.plot([r["aoa_deg"] for r in rows], [r["CD"] for r in rows], marker="o", label=case)
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("CD")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("CD vs AoA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CD_vs_AoA.png"), dpi=150)
    plt.close()

    # CL/CD vs AoA
    plt.figure(figsize=(8, 5))
    for case in cases:
        rows = [r for r in all_rows if r["case"] == case]
        plt.plot([r["aoa_deg"] for r in rows], [r["CL_CD"] for r in rows], marker="o", label=case)
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("CL/CD")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("CL/CD vs AoA")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CLCD_vs_AoA.png"), dpi=150)
    plt.close()


def write_summary(path, summaries):
    lines = [
        "--- AOA SWEEP COMPARISON SUMMARY ---",
        f"AoA sweep: {AOA_SWEEP[0]} deg to {AOA_SWEEP[-1]} deg",
        f"Reference AoA: {REF_AOA:.1f} deg",
        "",
    ]

    for case, s in summaries.items():
        lines.append(f"Case: {case}")
        lines.append(f"  Converged points: {s['n_valid']} / {s['n_total']}")
        lines.append(f"  Mean CL/CD: {s['mean_CL_CD']:.6f}")
        lines.append(f"  Peak CL/CD: {s['peak_CL_CD']:.6f}")
        lines.append(f"  CL/CD at {REF_AOA:.1f} deg: {s['CL_CD_at_ref']:.6f}")
        lines.append("")

    # Winner by mean efficiency
    valid_cases = [(k, v["mean_CL_CD"]) for k, v in summaries.items()]
    valid_cases = [(k, v) for k, v in valid_cases if v == v]  # filter nan

    if valid_cases:
        winner = max(valid_cases, key=lambda kv: kv[1])[0]
        lines.append(f"Best mean efficiency winner: {winner}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    ensure_dir(COMPARE_DIR)

    print("========================================================")
    print("--- STARTING AOA SWEEP COMPARISON ---")
    print("Cases:")
    print("  1. solid baseline")
    print(f"  2. saved best 2p/1ch from {CASE_2P1CH_DIR}")
    print(f"  3. saved best 4p/2ch from {CASE_4P2CH_DIR}")
    print(f"Output folder: {COMPARE_DIR}")
    print("========================================================")

    design_2p1ch = load_json(os.path.join(CASE_2P1CH_DIR, "best_design.json"))
    design_4p2ch = load_json(os.path.join(CASE_4P2CH_DIR, "best_design.json"))

    rows_solid = run_aoa_sweep("solid", None, aoa_list=AOA_SWEEP, verbose=True)
    rows_2p1ch = run_aoa_sweep("2p1ch", design_2p1ch, aoa_list=AOA_SWEEP, verbose=True)
    rows_4p2ch = run_aoa_sweep("4p2ch", design_4p2ch, aoa_list=AOA_SWEEP, verbose=True)

    all_rows = rows_solid + rows_2p1ch + rows_4p2ch
    save_sweep_csv(os.path.join(COMPARE_DIR, "aoa_sweep_comparison.csv"), all_rows)

    summaries = {
        "solid": summarize_sweep(rows_solid),
        "2p1ch": summarize_sweep(rows_2p1ch),
        "4p2ch": summarize_sweep(rows_4p2ch),
    }

    plot_comparison(all_rows, COMPARE_DIR)
    write_summary(os.path.join(COMPARE_DIR, "summary.txt"), summaries)

    print("\n--- AOA SWEEP SUMMARY ---")
    for case, s in summaries.items():
        print(f"{case:6s} | mean CL/CD = {s['mean_CL_CD']:.6f} | peak CL/CD = {s['peak_CL_CD']:.6f} | "
              f"CL/CD at {REF_AOA:.1f} deg = {s['CL_CD_at_ref']:.6f} | converged = {s['n_valid']}/{s['n_total']}")

    valid_cases = [(k, v["mean_CL_CD"]) for k, v in summaries.items() if v["mean_CL_CD"] == v["mean_CL_CD"]]
    if valid_cases:
        winner = max(valid_cases, key=lambda kv: kv[1])[0]
        print(f"Best mean efficiency winner: {winner}")

    print(f"Results saved to: {COMPARE_DIR}")