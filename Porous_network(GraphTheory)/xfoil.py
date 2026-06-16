"""
xfoil.py

Utilities for running XFOIL and parsing its outputs.

This version is configured for:
- inviscid flow
- incompressible flow

Notes
-----
For inviscid, incompressible XFOIL runs:
- do NOT use VISC
- do NOT use MACH
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


@dataclass
class XFOILPolarPoint:
    """One XFOIL polar row."""

    alpha: float
    CL: float
    CD: float
    CM: float


def _validate_xfoil_executable(xfoil_exe_path: str | Path) -> Path:
    """Validate and return the XFOIL executable path."""
    exe = Path(xfoil_exe_path)
    if not exe.exists():
        raise FileNotFoundError(f"XFOIL executable not found: {exe}")
    return exe


def _run_xfoil(
    xfoil_exe_path: str | Path,
    commands: str,
    output_dir: str | Path,
    input_filename: str,
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run XFOIL using a temporary input script."""
    exe = _validate_xfoil_executable(xfoil_exe_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / input_filename
    input_path.write_text(commands, encoding="utf-8")

    try:
        with input_path.open("r", encoding="utf-8") as fin:
            completed = subprocess.run(
                [str(exe)],
                stdin=fin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=str(output_dir),
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"XFOIL timed out after {timeout} s.") from exc

    return completed


def _oper_block() -> str:
    """Return the OPER block for inviscid, incompressible XFOIL runs."""
    return "OPER\n"


def run_xfoil_polar(
    xfoil_exe_path: str | Path,
    airfoil_name: str,
    reynolds: float,
    aoa_deg: float,
    output_dir: str | Path,
    mach: float = 0.0,
    timeout: float = 60.0,
) -> Path:
    """Run XFOIL at one angle of attack and save a polar file."""
    del reynolds, mach

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    polar_filename = f"xfoil_polar_{airfoil_name}_a{aoa_deg:.3f}.txt"
    polar_path = output_dir / polar_filename
    if polar_path.exists():
        polar_path.unlink()

    commands = (
        f"NACA {airfoil_name}\n"
        f"PANE\n"
        f"{_oper_block()}"
        f"PACC\n"
        f"{polar_filename}\n\n"
        f"ALFA {aoa_deg}\n"
        f"PACC\n\n"
        f"QUIT\n"
    )

    completed = _run_xfoil(
        xfoil_exe_path=xfoil_exe_path,
        commands=commands,
        output_dir=output_dir,
        input_filename="xfoil_polar_input.in",
        timeout=timeout,
    )

    if not polar_path.exists():
        raise RuntimeError(
            "XFOIL finished but no polar file was created.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    return polar_path


def load_xfoil_polar(filepath: str | Path) -> list[XFOILPolarPoint]:
    """Load an XFOIL polar file."""
    filepath = Path(filepath)
    points: list[XFOILPolarPoint] = []

    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 5:
            try:
                float(parts[0])
                float(parts[1])
                float(parts[2])
                start = i
                break
            except ValueError:
                pass

    if start is None:
        raise ValueError("Could not find numeric XFOIL polar data.")

    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            alpha = float(parts[0])
            cl = float(parts[1])
            cd = float(parts[2])
            cm = float(parts[4])
            points.append(XFOILPolarPoint(alpha=alpha, CL=cl, CD=cd, CM=cm))
        except ValueError:
            continue

    return points


def nearest_xfoil_point(points: list[XFOILPolarPoint], aoa_deg: float) -> XFOILPolarPoint:
    """Return the XFOIL point nearest to the requested angle of attack."""
    if not points:
        raise ValueError("Empty XFOIL point list.")
    idx = int(np.argmin([abs(p.alpha - aoa_deg) for p in points]))
    return points[idx]


def run_and_load_xfoil_point(
    xfoil_exe_path: str | Path,
    airfoil_name: str,
    reynolds: float,
    aoa_deg: float,
    output_dir: str | Path,
    mach: float = 0.0,
    timeout: float = 60.0,
) -> tuple[Path, XFOILPolarPoint]:
    """Run XFOIL, load the polar file, and return the point nearest the target AoA."""
    polar_path = run_xfoil_polar(
        xfoil_exe_path=xfoil_exe_path,
        airfoil_name=airfoil_name,
        reynolds=reynolds,
        aoa_deg=aoa_deg,
        output_dir=output_dir,
        mach=mach,
        timeout=timeout,
    )
    points = load_xfoil_polar(polar_path)
    point = nearest_xfoil_point(points, aoa_deg)
    return polar_path, point


def run_xfoil_cp(
    xfoil_exe_path: str | Path,
    airfoil_name: str,
    aoa_deg: float,
    output_dir: str | Path,
    mach: float = 0.0,
    timeout: float = 60.0,
    reynolds: float | None = None,
) -> Path:
    """Run XFOIL and export the surface Cp distribution."""
    del mach, reynolds

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cp_filename = f"xfoil_cp_{airfoil_name}_a{aoa_deg:.3f}.txt"
    cp_path = output_dir / cp_filename
    if cp_path.exists():
        cp_path.unlink()

    commands = (
        f"NACA {airfoil_name}\n"
        f"PANE\n"
        f"{_oper_block()}"
        f"ALFA {aoa_deg}\n"
        f"CPWR {cp_filename}\n"
        f"QUIT\n"
    )

    completed = _run_xfoil(
        xfoil_exe_path=xfoil_exe_path,
        commands=commands,
        output_dir=output_dir,
        input_filename="xfoil_cp_input.in",
        timeout=timeout,
    )

    if not cp_path.exists():
        raise RuntimeError(
            "XFOIL finished but no Cp file was created.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    return cp_path


def load_xfoil_cp(filepath: str | Path) -> dict:
    """Load an XFOIL surface Cp file."""
    filepath = Path(filepath)

    data = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                cp = float(parts[2])
                data.append((x, y, cp))
            except ValueError:
                continue

    if not data:
        raise ValueError("No numeric Cp data found in XFOIL Cp file.")

    arr = np.array(data)
    x = arr[:, 0]
    y = arr[:, 1]
    cp = arr[:, 2]

    upper = y >= 0.0
    lower = y < 0.0

    x_upper = x[upper]
    cp_upper = cp[upper]
    x_lower = x[lower]
    cp_lower = cp[lower]

    upper_order = np.argsort(x_upper)
    lower_order = np.argsort(x_lower)

    return {
        "x": x,
        "y": y,
        "cp": cp,
        "x_upper": x_upper[upper_order],
        "cp_upper": cp_upper[upper_order],
        "x_lower": x_lower[lower_order],
        "cp_lower": cp_lower[lower_order],
    }


def run_xfoil_polar_sweep(
    xfoil_exe_path: str | Path,
    airfoil_name: str,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    output_dir: str | Path,
    mach: float = 0.0,
    timeout: float = 120.0,
    reynolds: float | None = None,
) -> Path:
    """Run an XFOIL angle-of-attack polar sweep for inviscid, incompressible flow."""
    del mach, reynolds

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    polar_filename = (
        f"xfoil_polar_sweep_{airfoil_name}_"
        f"a{alpha_start:.1f}_to_{alpha_end:.1f}_step_{alpha_step:.1f}.txt"
    )
    polar_path = output_dir / polar_filename
    if polar_path.exists():
        polar_path.unlink()

    commands = (
        f"NACA {airfoil_name}\n"
        f"PANE\n"
        f"{_oper_block()}"
        f"PACC\n"
        f"{polar_filename}\n\n"
        f"ASEQ {alpha_start} {alpha_end} {alpha_step}\n"
        f"PACC\n\n"
        f"QUIT\n"
    )

    completed = _run_xfoil(
        xfoil_exe_path=xfoil_exe_path,
        commands=commands,
        output_dir=output_dir,
        input_filename="xfoil_sweep_input.in",
        timeout=timeout,
    )

    if not polar_path.exists():
        raise RuntimeError(
            "XFOIL sweep finished but no polar file was created.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    return polar_path


def load_xfoil_polar_dataframe(filepath: str | Path) -> pd.DataFrame:
    """Load an XFOIL polar file into a pandas DataFrame."""
    points = load_xfoil_polar(filepath)
    return pd.DataFrame(
        {
            "alpha_deg": [p.alpha for p in points],
            "xfoil_CL": [p.CL for p in points],
            "xfoil_CD": [p.CD for p in points],
            "xfoil_CM": [p.CM for p in points],
        }
    )
