from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from solver import (
    CouplingConfig,
    FlowConfig,
    PanelGeometry,
    ReferenceConfig,
    SourceVortexPanelMethod,
    SurfacePoint,
    generate_naca4,
)
from plotter import (
    plot_airfoil_with_porous_network,
    plot_aoa_sweep_comparison,
    plot_cp_distribution_with_xfoil,
    plot_difference_contours,
    plot_pressure_contours_comparison,
    plot_velocity_contours_comparison,
)
from xfoil import (
    load_xfoil_cp,
    load_xfoil_polar_dataframe,
    run_xfoil_cp,
    run_xfoil_polar_sweep,
)
from porous_config import (
    AIRFOIL_NAME,
    AOA_DEG,
    AOA_SWEEP_END_DEG,
    AOA_SWEEP_START_DEG,
    AOA_SWEEP_STEP_DEG,
    CHORD,
    COUPLING,
    EXPORT_AOA_SWEEP_PARAVIEW,
    GeometryLimits,
    Chamber,
    Pore,
    PassageState,
    PassageSpec,
    PorousModelSpec,
    MultiPassageResult,
    P_INF,
    REYNOLDS_EXTERNAL,
    RHO_INF,
    MU_INF,
    SPAN_WIDTH,
    USE_XFOIL,
    XFOIL_EXE_PATH,
    XFOIL_MACH,
    XFOIL_OUTPUT_ROOT,
    XFOIL_TIMEOUT,
    circular_laminar_resistance,
    reynolds_to_velocity,
)

from paraview_export import export_aoa_sweep_collections, export_paraview_files

# =============================================================================
# ONE INDEPENDENT CHANNEL
# =============================================================================
class IndependentPassage:
    def __init__(
        self,
        name: str,
        pore1: Pore,
        pore2: Pore,
        chamber: Chamber,
        span_width: float,
    ) -> None:
        self.name = name
        self.pore1 = pore1
        self.pore2 = pore2
        self.chamber = chamber
        self.span_width = float(span_width)

    def pore_surface_point(self, geom: PanelGeometry, pore: Pore) -> SurfacePoint:
        return geom.surface_point_from_fraction(pore.x_frac, pore.side)

    def passage_length(self, geom: PanelGeometry) -> float:
        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        def safe_thickness(x_frac: float) -> float:
            if x_frac <= 1e-8 or x_frac >= 1.0 - 1e-8:
                return 0.0
            return geom.local_thickness_at_fraction(x_frac)

        t1 = safe_thickness(self.pore1.x_frac)
        t2 = safe_thickness(self.pore2.x_frac)

        straight = np.hypot(sp2.x - sp1.x, sp2.y - sp1.y)
        return max(float(straight + 0.5 * t1 + 0.5 * t2), 1e-6)


    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        for label, pore in (("pore1", self.pore1), ("pore2", self.pore2)):
            if not (limits.x_min_frac <= pore.x_frac <= limits.x_max_frac):
                raise ValueError(
                    f"{self.name}:{label} x/c={pore.x_frac:.4f} is outside "
                    f"[{limits.x_min_frac}, {limits.x_max_frac}]"
                )

            is_endpoint_pore = (
                abs(pore.x_frac - 0.0) <= limits.endpoint_tol
                or abs(pore.x_frac - 1.0) <= limits.endpoint_tol
            )

            # Exact LE/TE surface points have zero local thickness in 2D.
            # They are allowed only as endpoint pores.
            if not is_endpoint_pore:
                thickness = geom.local_thickness_at_fraction(pore.x_frac)

                if pore.diameter > limits.thickness_fraction_limit * thickness:
                    raise ValueError(
                        f"{self.name}:{label} diameter={pore.diameter:.4e} is too large "
                        f"for local thickness={thickness:.4e}"
                    )

        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        dist = float(np.hypot(sp2.x - sp1.x, sp2.y - sp1.y))
        required = 0.5 * (self.pore1.diameter + self.pore2.diameter) + limits.min_gap

        if dist < required:
            raise ValueError(
                f"{self.name} pore centers are too close: distance={dist:.4e}, "
                f"required={required:.4e}"
            )

        x_min = min(self.pore1.x_frac, self.pore2.x_frac)
        x_max = max(self.pore1.x_frac, self.pore2.x_frac)

        # Skip exact LE/TE zero-thickness locations for chamber fit checking.
        check_x_min = max(x_min, limits.endpoint_fit_margin)
        check_x_max = min(x_max, 1.0 - limits.endpoint_fit_margin)

        if check_x_min <= check_x_max:
            if abs(check_x_max - check_x_min) <= 1e-12:
                xs = np.array([check_x_min])
            else:
                xs = np.linspace(check_x_min, check_x_max, 40)

            t_min = min(geom.local_thickness_at_fraction(float(x)) for x in xs)

            if self.chamber.hydraulic_diameter > limits.fit_fraction * t_min:
                raise ValueError(
                    f"{self.name} chamber diameter is too large for the minimum local thickness. "
                    f"diameter={self.chamber.hydraulic_diameter:.4e}, t_min={t_min:.4e}"
                )

    def equivalent_reynolds(self, q_abs: float, flow: FlowConfig) -> float:
        u_internal = q_abs / max(self.chamber.area, 1e-16)
        return (
            flow.rho_inf
            * abs(u_internal)
            * self.chamber.hydraulic_diameter
            / max(flow.mu_inf, 1e-16)
        )

    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
    ) -> PassageState:
        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        p1 = float(panel_pressures[sp1.panel_id])
        p2 = float(panel_pressures[sp2.panel_id])
        dp = p1 - p2

        length = self.passage_length(geom)
        rs, _ = circular_laminar_resistance(
            mu=flow.mu_inf,
            length=length,
            diameter=self.chamber.hydraulic_diameter,
        )
        q = dp / rs

        return PassageState(
            Q=float(q),
            p1=p1,
            p2=p2,
            dp_total=float(abs(dp)),
            Rs=float(rs),
            reynolds_equivalent=self.equivalent_reynolds(abs(q), flow),
        )

    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        state: PassageState,
    ) -> np.ndarray:
        """
        Convert internal pore flow rate Q into panel normal transpiration velocity.

        No weighting is used.

        Each pore flow is applied only to the nearest surface panel:
            pore1 panel gets -Q
            pore2 panel gets +Q

        Positive Q means internal flow goes from pore1 to pore2.
        """
        vn = np.zeros(geom.num_pan, dtype=float)

        Q = float(state.Q)
        if abs(Q) < 1e-30:
            return vn

        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        panel1 = int(sp1.panel_id)
        panel2 = int(sp2.panel_id)

        q1 = -Q
        q2 = +Q

        vn[panel1] += q1 / max(self.span_width * geom.S[panel1], 1e-16)
        vn[panel2] += q2 / max(self.span_width * geom.S[panel2], 1e-16)

        return vn

# =============================================================================
# NETWORK OF INDEPENDENT CHANNELS
# =============================================================================
class IndependentPassageNetwork:
    def __init__(self, passages: list[IndependentPassage]) -> None:
        if not passages:
            raise ValueError("At least one passage is required.")
        self.passages = passages

    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        for passage in self.passages:
            passage.validate_geometry(geom, limits)

        pore_refs: list[tuple[str, Pore, SurfacePoint]] = []
        for passage in self.passages:
            pore_refs.append(
                (
                    f"{passage.name}:pore1",
                    passage.pore1,
                    passage.pore_surface_point(geom, passage.pore1),
                )
            )
            pore_refs.append(
                (
                    f"{passage.name}:pore2",
                    passage.pore2,
                    passage.pore_surface_point(geom, passage.pore2),
                )
            )

        for i in range(len(pore_refs)):
            name_i, pore_i, sp_i = pore_refs[i]

            for j in range(i + 1, len(pore_refs)):
                name_j, pore_j, sp_j = pore_refs[j]

                dist = float(np.hypot(sp_i.x - sp_j.x, sp_i.y - sp_j.y))
                required = 0.5 * (pore_i.diameter + pore_j.diameter) + limits.min_gap

                if dist < required:
                    raise ValueError(
                        f"Pore overlap between {name_i} and {name_j}: "
                        f"distance={dist:.4e}, required={required:.4e}"
                    )

    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
    ) -> list[PassageState]:
        return [
            passage.solve(
                geom=geom,
                panel_pressures=panel_pressures,
                flow=flow,
            )
            for passage in self.passages
        ]

    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        states: list[PassageState],
    ) -> np.ndarray:
        vn = np.zeros(geom.num_pan, dtype=float)

        for passage, state in zip(self.passages, states):
            vn += passage.transpiration_from_flow(
                geom=geom,
                state=state,
            )

        return vn



# =============================================================================
# COUPLED PANEL + POROUS SOLVER
# =============================================================================
class MultiPassageCoupledSolver:
    def __init__(
        self,
        aero_solver: SourceVortexPanelMethod,
        network: IndependentPassageNetwork,
        coupling: CouplingConfig,
        limits: GeometryLimits,
    ) -> None:
        self.aero_solver = aero_solver
        self.network = network
        self.coupling = coupling
        self.limits = limits

    def solve(self, verbose: bool = False) -> MultiPassageResult:
        geom = self.aero_solver.geom
        flow = self.aero_solver.flow

        self.network.validate_geometry(geom, self.limits)

        vn = np.zeros(geom.num_pan, dtype=float)
        previous_q: list[float] | None = None

        converged = False
        final_iteration = 0

        for iteration in range(1, self.coupling.max_iter + 1):
            aero_result = self.aero_solver.solve(normal_transpiration=vn)
            panel_pressures = self.aero_solver.panel_pressures(aero_result)

            states = self.network.solve(
                geom=geom,
                panel_pressures=panel_pressures,
                flow=flow,
            )

            vn_target = self.network.transpiration_from_flow(
                geom=geom,
                states=states,
            )

            vn_new = (
                (1.0 - self.coupling.relaxation) * vn
                + self.coupling.relaxation * vn_target
            )

            current_q = [float(s.Q) for s in states]
            if previous_q is None:
                dq = np.inf
            else:
                dq = max(abs(q_new - q_old) for q_new, q_old in zip(current_q, previous_q))

            dvn = float(np.max(np.abs(vn_new - vn)))

            if verbose:
                max_q = max(abs(q) for q in current_q)
                print(
                    f"[Coupling] iter={iteration:03d} | "
                    f"CL={aero_result.CL: .8f} | "
                    f"max|Q|={max_q: .6e} | "
                    f"dvn={dvn: .3e} | dq={dq: .3e}"
                )

            vn = vn_new
            previous_q = current_q
            final_iteration = iteration

            if dvn < self.coupling.tol_vn and dq < self.coupling.tol_q:
                converged = True
                break

        aero_result = self.aero_solver.solve(normal_transpiration=vn)
        panel_pressures = self.aero_solver.panel_pressures(aero_result)

        states = self.network.solve(
            geom=geom,
            panel_pressures=panel_pressures,
            flow=flow,
        )

        max_vn = float(np.max(np.abs(vn)))

        return MultiPassageResult(
            aero_result=aero_result,
            passage_states=states,
            normal_transpiration=vn,
            converged=converged,
            iterations=final_iteration,
            max_vn=max_vn,
            max_vn_over_vinf=max_vn / max(flow.v_inf, 1e-16),
        )



# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
def _arc_length_spaced_points_on_side(
    geom: PanelGeometry,
    side: str,
    x_start_frac: float,
    x_end_frac: float,
    n_points: int,
    n_sample: int = 800,
) -> list[tuple[float, str]]:
    """
    Return n_points equally spaced by surface arc length on one airfoil side.

    Output:
        [(x_frac, side), ...]
    """
    side = side.lower()

    if side == "upper":
        sx, sy = geom.upper_x, geom.upper_y
    elif side == "lower":
        sx, sy = geom.lower_x, geom.lower_y
    else:
        raise ValueError("side must be 'upper' or 'lower'")

    x_frac = np.linspace(x_start_frac, x_end_frac, n_sample)
    x_abs = geom.x_le + x_frac * geom.chord
    y_abs = np.interp(x_abs, sx, sy)

    ds = np.hypot(np.diff(x_abs), np.diff(y_abs))
    s = np.concatenate([[0.0], np.cumsum(ds)])

    targets = np.linspace(0.0, s[-1], n_points)
    x_out = np.interp(targets, s, x_frac)

    return [(float(x), side) for x in x_out]


def _center_plus_four_each_side_surface_band(
    geom: PanelGeometry,
    center_x_frac: float,
    outer_x_frac: float,
) -> list[tuple[float, str]]:
    """
    Build 9 surface pore points:

        4 on lower surface
        1 exact center endpoint
        4 on upper surface

    For LE:
        center_x_frac = 0.0

    For TE:
        center_x_frac = 1.0

    The center point is assigned to the upper surface because the panel method
    requires each pore to belong to either 'upper' or 'lower'.

    At x/c = 0, upper and lower meet at the LE.
    At x/c = 1, upper and lower meet at the TE.
    """
    lower_points = _arc_length_spaced_points_on_side(
        geom=geom,
        side="lower",
        x_start_frac=outer_x_frac,
        x_end_frac=center_x_frac,
        n_points=5,
    )[:-1]

    center_point = [(float(center_x_frac), "upper")]

    upper_points = _arc_length_spaced_points_on_side(
        geom=geom,
        side="upper",
        x_start_frac=center_x_frac,
        x_end_frac=outer_x_frac,
        n_points=5,
    )[1:]

    return lower_points + center_point + upper_points

def model1_le_te_surface_pores(
    geom: PanelGeometry,
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """
    Model 1 pore placement.

    Layout:
        LE side:
            4 lower-side pores
            1 exact LE middle pore at x/c = 0.0
            4 upper-side pores

        TE side:
            4 lower-side pores
            1 exact TE middle pore at x/c = 1.0
            4 upper-side pores

    Each LE point is connected to the corresponding TE point.
    """
    le_center = 0.0
    le_outer = 0.155

    te_center = 1.0
    te_outer = 0.845

    le_points = _center_plus_four_each_side_surface_band(
        geom=geom,
        center_x_frac=le_center,
        outer_x_frac=le_outer,
    )

    te_points = _center_plus_four_each_side_surface_band(
        geom=geom,
        center_x_frac=te_center,
        outer_x_frac=te_outer,
    )

    return le_points, te_points


def build_model1_chordwise(
    geom: PanelGeometry,
    diameter_m: float,
) -> list[PassageSpec]:
    """
    Model 1:
        9 LE surface pores
        9 TE surface pores
        9 independent internal chordwise channels

    Channel order:
        01-04 : lower side
        05    : middle
        06-09 : upper side
    """
    le_points, te_points = model1_le_te_surface_pores(geom)

    passages: list[PassageSpec] = []

    for i, ((x_le, side_le), (x_te, side_te)) in enumerate(
        zip(le_points, te_points),
        start=1,
    ):
        offset_index = i - 5

        if offset_index == 0:
            channel_label = "middle"
        elif offset_index < 0:
            channel_label = f"lower_{abs(offset_index)}"
        else:
            channel_label = f"upper_{offset_index}"

        passages.append(
            PassageSpec(
                name=f"m1_{i:02d}_{channel_label}_LE_to_TE_internal_channel",
                x1_frac=x_le,
                side1=side_le,
                x2_frac=x_te,
                side2=side_te,
                diameter_m=diameter_m,
                layout_kind="internal_chordwise",
            )
        )

    return passages


def build_vertical_passages(
    x_values: np.ndarray,
    diameter_m: float,
    prefix: str,
) -> list[PassageSpec]:
    """
    Build lower-to-upper perpendicular channels.
    """
    passages: list[PassageSpec] = []

    for i, x in enumerate(x_values, start=1):
        passages.append(
            PassageSpec(
                name=f"{prefix}_{i:02d}_perpendicular",
                x1_frac=float(x),
                side1="lower",
                x2_frac=float(x),
                side2="upper",
                diameter_m=diameter_m,
                layout_kind="perpendicular",
            )
        )

    return passages


def _copy_passages_with_prefix(
    passages: list[PassageSpec],
    prefix: str,
    label: str,
) -> list[PassageSpec]:
    """
    Copy passage definitions and rename them.

    Used by Model 3 so it can reuse the exact Model 1 channel geometry.
    """
    copied: list[PassageSpec] = []

    for i, passage in enumerate(passages, start=1):
        copied.append(
            PassageSpec(
                name=f"{prefix}_{i:02d}_{label}",
                x1_frac=passage.x1_frac,
                side1=passage.side1,
                x2_frac=passage.x2_frac,
                side2=passage.side2,
                diameter_m=passage.diameter_m,
                layout_kind=passage.layout_kind,
            )
        )

    return copied


# =============================================================================
# MODEL 4: SAVED CASE 1 FROM OLD run_test.py
# =============================================================================
def build_model4_saved_case1() -> list[PassageSpec]:
    """
    Model 4:
        Saved case_1 from the old run_test.py validation script.

    Original saved case:
        topology = lower-upper and lower-upper

        passage 1:
            x1/c = 0.400094 on lower surface
            x2/c = 0.933001 on upper surface
            diameter = 1.911007e-02 m

        passage 2:
            x1/c = 0.925069 on lower surface
            x2/c = 0.955118 on upper surface
            diameter = 1.324942e-02 m

    This model uses its own fixed diameters.
    It ignores the common diameter_m passed to build_model_specs().
    """
    return [
        PassageSpec(
            name="m4_case1_passage_01_lower_to_upper",
            x1_frac=0.400094,
            side1="lower",
            x2_frac=0.933001,
            side2="upper",
            diameter_m=1.2e-02,  # original diameter was 1.911007e-02 m
            layout_kind="saved_case_1",
        ),
        PassageSpec(
            name="m4_case1_passage_02_lower_to_upper",
            x1_frac=0.925069,
            side1="lower",
            x2_frac=0.955118,
            side2="upper",
            diameter_m=0.9e-02,  # original diameter was 1.324942e-02 m
            layout_kind="saved_case_1",
        ),
    ]


def build_model_specs(
    geom: PanelGeometry,
    diameter_m: float,
) -> list[PorousModelSpec]:
    """
    Build the fixed porous models.

    Models 1-3 use the common diameter_m.
    Model 4 is the saved case_1 design from the old run_test.py and uses
    its own two fixed diameters.
    """
    model_1_chordwise = build_model1_chordwise(
        geom=geom,
        diameter_m=diameter_m,
    )

    model_2_perpendicular = build_vertical_passages(
        x_values=np.linspace(0.100, 0.900, 9),
        diameter_m=diameter_m,
        prefix="m2",
    )

    model_3_chordwise = _copy_passages_with_prefix(
        passages=model_1_chordwise,
        prefix="m3c",
        label="LE_to_TE_internal_channel",
    )

    model_3_perpendicular = build_vertical_passages(
        x_values=np.linspace(0.100, 0.900, 9),
        diameter_m=diameter_m,
        prefix="m3v",
    )

    model_3_combined = model_3_chordwise + model_3_perpendicular

    model_4_saved_case1 = build_model4_saved_case1()

    return [
        PorousModelSpec(
            name="model_1_9_chordwise",
            description=(
                "9 LE-to-TE internal chordwise channels: one middle channel "
                "and four equally spaced channels on each side."
            ),
            passages=tuple(model_1_chordwise),
        ),
        PorousModelSpec(
            name="model_2_9_perpendicular",
            description="9 independent lower-to-upper perpendicular channels.",
            passages=tuple(model_2_perpendicular),
        ),
        PorousModelSpec(
            name="model_3_combined_independent",
            description=(
                "Combined Model 1 + Model 2: 9 LE-to-TE chordwise channels "
                "plus 9 lower-to-upper perpendicular channels. All 18 passages "
                "are independent; internal intersections are not connected."
            ),
            passages=tuple(model_3_combined),
        ),
        PorousModelSpec(
            name="model_4_saved_case_1",
            description=(
                "Saved case_1 from old run_test.py: two independent lower-to-upper "
                "dual porous passages with fixed pore locations and fixed diameters. "
                "This model does not use the common DEFAULT_PORE_DIAMETER."
            ),
            passages=tuple(model_4_saved_case1),
        ),
    ]



# =============================================================================
# BUILDERS
# =============================================================================
def build_base_problem(
    airfoil_name: str,
    chord: float,
    n_panels: int,
    aoa_deg: float,
    reynolds_external: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
) -> tuple[FlowConfig, PanelGeometry, ReferenceConfig, SourceVortexPanelMethod]:
    v_inf = reynolds_to_velocity(
        reynolds=reynolds_external,
        rho=rho_inf,
        mu=mu_inf,
        chord=chord,
    )

    flow = FlowConfig(
        aoa_deg=aoa_deg,
        v_inf=v_inf,
        rho_inf=rho_inf,
        mu_inf=mu_inf,
        p_inf=p_inf,
    )

    xb, yb = generate_naca4(
        airfoil_name,
        n_panels,
        chord=chord,
    )

    geom = PanelGeometry(xb, yb, aoa_deg)
    ref = ReferenceConfig(x_ref=0.25 * chord, y_ref=0.0)
    aero_solver = SourceVortexPanelMethod(geom, flow, ref)

    return flow, geom, ref, aero_solver


def build_network_from_model(
    model: PorousModelSpec,
    span_width: float,
) -> IndependentPassageNetwork:
    passages: list[IndependentPassage] = []

    for spec in model.passages:
        d = float(spec.diameter_m)

        passages.append(
            IndependentPassage(
                name=spec.name,
                pore1=Pore(spec.x1_frac, spec.side1, d),
                pore2=Pore(spec.x2_frac, spec.side2, d),
                chamber=Chamber(
                    hydraulic_diameter=d,
                    area=0.25 * np.pi * d**2,
                ),
                span_width=span_width,
            )
        )

    return IndependentPassageNetwork(passages)



# =============================================================================
# XFOIL HELPERS
# =============================================================================
def run_xfoil_cp_for_case(
    airfoil_name: str,
    aoa_deg: float,
    output_dir: Path,
) -> dict | None:
    """
    Run XFOIL Cp export and return loaded Cp data.
    """
    if not USE_XFOIL or XFOIL_EXE_PATH is None:
        print("[XFOIL] Skipping Cp comparison because xfoil.exe was not found.")
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        cp_path = run_xfoil_cp(
            xfoil_exe_path=XFOIL_EXE_PATH,
            airfoil_name=airfoil_name,
            aoa_deg=aoa_deg,
            output_dir=output_dir,
            mach=XFOIL_MACH,
            timeout=XFOIL_TIMEOUT,
            reynolds=REYNOLDS_EXTERNAL,
        )

        print(f"[XFOIL] Cp file created: {cp_path}")
        return load_xfoil_cp(cp_path)

    except Exception as exc:
        print(f"[XFOIL] Cp comparison failed: {exc}")
        return None


def run_xfoil_aoa_sweep_for_case(
    airfoil_name: str,
    output_dir: Path,
    alpha_start: float = -5.0,
    alpha_end: float = 15.0,
    alpha_step: float = 1.0,
) -> pd.DataFrame | None:
    """
    Run XFOIL AoA sweep.
    """
    if not USE_XFOIL or XFOIL_EXE_PATH is None:
        print("[XFOIL] Skipping AoA sweep because xfoil.exe was not found.")
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        sweep_path = run_xfoil_polar_sweep(
            xfoil_exe_path=XFOIL_EXE_PATH,
            airfoil_name=airfoil_name,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=alpha_step,
            output_dir=output_dir,
            mach=XFOIL_MACH,
            timeout=XFOIL_TIMEOUT,
            reynolds=REYNOLDS_EXTERNAL,
        )

        print(f"[XFOIL] AoA sweep file created: {sweep_path}")
        return load_xfoil_polar_dataframe(sweep_path)

    except Exception as exc:
        print(f"[XFOIL] AoA sweep failed: {exc}")
        return None


def add_xfoil_to_sweep_comparison(
    sweep_df: pd.DataFrame,
    xfoil_sweep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge XFOIL polar data into panel-method sweep data.
    """
    xfoil_df = xfoil_sweep_df.copy()
    xfoil_df.columns = [str(c).strip() for c in xfoil_df.columns]

    rename_map: dict[str, str] = {}

    for col in xfoil_df.columns:
        c = col.strip().lower()

        if c in {"alpha", "alpha_deg", "aoa", "aoa_deg"}:
            rename_map[col] = "alpha_deg"
        elif c == "cl":
            rename_map[col] = "xfoil_CL"
        elif c == "cd":
            rename_map[col] = "xfoil_CD"
        elif c == "cm":
            rename_map[col] = "xfoil_CM"

    xfoil_df = xfoil_df.rename(columns=rename_map)

    required = {"alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM"}
    missing = required - set(xfoil_df.columns)

    if missing:
        raise ValueError(
            f"XFOIL sweep dataframe missing columns after normalization: {sorted(missing)}. "
            f"Available columns: {list(xfoil_sweep_df.columns)}"
        )

    return sweep_df.merge(
        xfoil_df[["alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM"]],
        on="alpha_deg",
        how="left",
    )



# =============================================================================
# CSV EXPORTS
# =============================================================================
def build_passage_dataframe(
    model: PorousModelSpec,
    network: IndependentPassageNetwork,
    geom: PanelGeometry,
    result: MultiPassageResult,
) -> pd.DataFrame:
    rows = []

    for passage_id, (passage, state) in enumerate(
        zip(network.passages, result.passage_states),
        start=1,
    ):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        rows.append(
            {
                "model_name": model.name,
                "passage_id": passage_id,
                "passage_name": passage.name,
                "pore1_side": passage.pore1.side,
                "pore2_side": passage.pore2.side,
                "x1_frac": passage.pore1.x_frac,
                "x2_frac": passage.pore2.x_frac,
                "x1_surface_m": sp1.x,
                "y1_surface_m": sp1.y,
                "x2_surface_m": sp2.x,
                "y2_surface_m": sp2.y,
                "x1_over_c_surface": sp1.x / geom.chord,
                "y1_over_c_surface": sp1.y / geom.chord,
                "x2_over_c_surface": sp2.x / geom.chord,
                "y2_over_c_surface": sp2.y / geom.chord,
                "pore_diameter_m": passage.pore1.diameter,
                "passage_length_m": passage.passage_length(geom),
                "Q_m3_s": state.Q,
                "abs_Q_m3_s": abs(state.Q),
                "Rs_Pa_s_per_m3": state.Rs,
                "dp_total_Pa": state.dp_total,
                "reynolds_internal_equivalent": state.reynolds_equivalent,
                "p1_Pa": state.p1,
                "p2_Pa": state.p2,
            }
        )

    return pd.DataFrame(rows)


def export_surface_data(
    output_dir: Path,
    geom: PanelGeometry,
    solid_result: object,
    porous_result: MultiPassageResult,
) -> None:
    aero = porous_result.aero_result

    df = pd.DataFrame(
        {
            "panel_id": np.arange(geom.num_pan),
            "XB_start_m": geom.XB[:-1],
            "YB_start_m": geom.YB[:-1],
            "XB_end_m": geom.XB[1:],
            "YB_end_m": geom.YB[1:],
            "XC_m": geom.XC,
            "YC_m": geom.YC,
            "XC_over_c": geom.XC / geom.chord,
            "YC_over_c": geom.YC / geom.chord,
            "panel_length_m": geom.S,
            "porous_Vt_m_s": aero.Vt,
            "porous_Cp": aero.Cp,
            "solid_Vt_m_s": solid_result.Vt,
            "solid_Cp": solid_result.Cp,
            "delta_Cp_porous_minus_solid": aero.Cp - solid_result.Cp,
            "normal_transpiration_m_s": porous_result.normal_transpiration,
        }
    )

    path = output_dir / "surface_data.csv"
    df.to_csv(path, index=False)
    print(f"[CSV Saved] {path}")


def build_model_summary_row(
    model: PorousModelSpec,
    flow: FlowConfig,
    solid_result: object,
    porous_result: MultiPassageResult,
    elapsed_seconds: float,
) -> dict:
    aero = porous_result.aero_result

    q_abs = [abs(float(s.Q)) for s in porous_result.passage_states]
    dp_values = [float(s.dp_total) for s in porous_result.passage_states]
    re_values = [float(s.reynolds_equivalent) for s in porous_result.passage_states]

    return {
        "model_name": model.name,
        "description": model.description,
        "num_independent_passages": len(model.passages),
        "reynolds_external": REYNOLDS_EXTERNAL,
        "v_inf_m_s": flow.v_inf,
        "aoa_deg": flow.aoa_deg,
        "solid_CL": solid_result.CL,
        "solid_CD": solid_result.CD,
        "solid_CM": solid_result.CM,
        "porous_CL": aero.CL,
        "porous_CD": aero.CD,
        "porous_CM": aero.CM,
        "delta_CL": aero.CL - solid_result.CL,
        "delta_CD": aero.CD - solid_result.CD,
        "delta_CM": aero.CM - solid_result.CM,
        "gain_percent_vs_solid_panel": 100.0 * (aero.CL / solid_result.CL - 1.0),
        "max_abs_Q_m3_s": max(q_abs) if q_abs else 0.0,
        "sum_abs_Q_m3_s": sum(q_abs),
        "max_dp_total_Pa": max(dp_values) if dp_values else 0.0,
        "max_internal_reynolds_equivalent": max(re_values) if re_values else 0.0,
        "max_vn_m_s": porous_result.max_vn,
        "max_vn_over_vinf": porous_result.max_vn_over_vinf,
        "coupling_converged": porous_result.converged,
        "coupling_iterations": porous_result.iterations,
        "elapsed_seconds": elapsed_seconds,
    }



def _alpha_directory_name(alpha_deg: float) -> str:
    """Return a stable folder name for one AoA sweep case."""
    sign = "p" if alpha_deg >= 0.0 else "m"
    magnitude = f"{abs(alpha_deg):06.2f}".replace(".", "p")
    return f"alpha_{sign}{magnitude}_deg"


# =============================================================================
# AOA SWEEP
# =============================================================================
def run_porous_aoa_sweep_for_model(
    model: PorousModelSpec,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    reynolds_external: float,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    alpha_start: float = -5.0,
    alpha_end: float = 15.0,
    alpha_step: float = 1.0,
    export_paraview: bool = False,
    paraview_output_dir: Path | None = None,
    field_nx: int = 300,
    field_ny: int = 300,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    paraview_cases: list[dict[str, Path]] = []
    paraview_alphas: list[float] = []

    if export_paraview:
        if paraview_output_dir is None:
            raise ValueError("paraview_output_dir is required when export_paraview=True.")

        paraview_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ParaView] AoA sweep export folder: {paraview_output_dir}")

    for alpha in np.arange(alpha_start, alpha_end + 0.1 * alpha_step, alpha_step):
        alpha = float(alpha)

        flow_i, geom_i, _, aero_solver_i = build_base_problem(
            airfoil_name=airfoil_name,
            chord=chord,
            n_panels=n_panels,
            aoa_deg=alpha,
            reynolds_external=reynolds_external,
            rho_inf=rho_inf,
            mu_inf=mu_inf,
            p_inf=p_inf,
        )

        solid_i = aero_solver_i.solve()

        network_i = build_network_from_model(
            model=model,
            span_width=SPAN_WIDTH,
        )

        coupled_i = MultiPassageCoupledSolver(
            aero_solver=aero_solver_i,
            network=network_i,
            coupling=coupling,
            limits=limits,
        ).solve(verbose=False)

        if export_paraview and paraview_output_dir is not None:
            alpha_dir = paraview_output_dir / _alpha_directory_name(alpha)

            exported_files = export_paraview_files(
                output_dir=alpha_dir,
                network=network_i,
                geom=geom_i,
                aero_solver=aero_solver_i,
                solid_result=solid_i,
                porous_result=coupled_i,
                field_nx=field_nx,
                field_ny=field_ny,
                include_solid_exports=True,
                collection_name="paraview_results.pvd",
            )

            paraview_cases.append(exported_files)
            paraview_alphas.append(alpha)

        rows.append(
            {
                "alpha_deg": alpha,
                "porous_CL": float(coupled_i.aero_result.CL),
                "porous_CD": float(coupled_i.aero_result.CD),
                "porous_CM": float(coupled_i.aero_result.CM),
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
                "delta_CL": float(coupled_i.aero_result.CL - solid_i.CL),
                "delta_CD": float(coupled_i.aero_result.CD - solid_i.CD),
                "delta_CM": float(coupled_i.aero_result.CM - solid_i.CM),
                "max_abs_Q_m3_s": max(abs(float(s.Q)) for s in coupled_i.passage_states),
                "sum_abs_Q_m3_s": sum(abs(float(s.Q)) for s in coupled_i.passage_states),
                "max_dp_total_Pa": max(float(s.dp_total) for s in coupled_i.passage_states),
                "max_internal_reynolds_equivalent": max(
                    float(s.reynolds_equivalent) for s in coupled_i.passage_states
                ),
                "max_vn_m_s": float(coupled_i.max_vn),
                "coupling_converged": bool(coupled_i.converged),
                "coupling_iterations": int(coupled_i.iterations),
                "max_vn_over_vinf": float(coupled_i.max_vn_over_vinf),
            }
        )

    if export_paraview and paraview_output_dir is not None:
        export_aoa_sweep_collections(
            output_dir=paraview_output_dir,
            cases=paraview_cases,
            alpha_values=paraview_alphas,
        )

    return pd.DataFrame(rows)


# =============================================================================
# RUN ONE MODEL
# =============================================================================
def run_one_model(
    model: PorousModelSpec,
    geom: PanelGeometry,
    flow: FlowConfig,
    aero_solver: SourceVortexPanelMethod,
    solid_result: object,
    output_root: Path,
    make_contours: bool,
    contour_nx: int,
    contour_ny: int,
) -> dict:
    model_dir = output_root / model.name
    model_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"RUNNING {model.name}")
    print("=" * 80)
    print(model.description)
    print(f"Independent channels: {len(model.passages)}")

    network = build_network_from_model(model, SPAN_WIDTH)

    t0 = time.perf_counter()

    coupled = MultiPassageCoupledSolver(
        aero_solver=aero_solver,
        network=network,
        coupling=COUPLING,
        limits=GeometryLimits(),
    ).solve(verbose=True)

    elapsed = time.perf_counter() - t0

    print()
    print(f"[Result] converged              : {coupled.converged}")
    print(f"[Result] coupling iterations    : {coupled.iterations}")
    print(f"[Result] solid CL               : {solid_result.CL:.8f}")
    print(f"[Result] porous CL              : {coupled.aero_result.CL:.8f}")
    print(
        f"[Result] lift gain vs solid [%] : "
        f"{100.0 * (coupled.aero_result.CL / solid_result.CL - 1.0):.6f}"
    )
    print(f"[Result] max |vn| / Vinf        : {coupled.max_vn_over_vinf:.8f}")
    print(f"[Result] elapsed [s]            : {elapsed:.3f}")

    passage_df = build_passage_dataframe(
        model=model,
        network=network,
        geom=geom,
        result=coupled,
    )
    passage_path = model_dir / "passage_summary.csv"
    passage_df.to_csv(passage_path, index=False)
    print(f"[CSV Saved] {passage_path}")

    export_surface_data(
        output_dir=model_dir,
        geom=geom,
        solid_result=solid_result,
        porous_result=coupled,
    )
    export_paraview_files(
        output_dir=model_dir,
        network=network,
        geom=geom,
        aero_solver=aero_solver,
        solid_result=solid_result,
        porous_result=coupled,
        field_nx=contour_nx,
        field_ny=contour_ny,
    )

    summary_row = build_model_summary_row(
        model=model,
        flow=flow,
        solid_result=solid_result,
        porous_result=coupled,
        elapsed_seconds=elapsed,
    )
    summary_path = model_dir / "model_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"[CSV Saved] {summary_path}")

    xfoil_model_dir = XFOIL_OUTPUT_ROOT / model.name

    xfoil_cp_data = run_xfoil_cp_for_case(
        airfoil_name=AIRFOIL_NAME,
        aoa_deg=AOA_DEG,
        output_dir=xfoil_model_dir,
    )

    plot_airfoil_with_porous_network(
        geom=geom,
        network=network,
        coupled_result=coupled,
        output_dir=model_dir,
        title=f"{model.name}: Airfoil with Porous Network",
    )

    plot_cp_distribution_with_xfoil(
        geom=geom,
        porous_result=coupled.aero_result,
        solid_result=solid_result,
        output_dir=model_dir,
        xfoil_cp_data=xfoil_cp_data,
    )

    if make_contours:
        plot_velocity_contours_comparison(
            porous_solver=aero_solver,
            porous_result=coupled.aero_result,
            solid_solver=aero_solver,
            solid_result=solid_result,
            output_dir=model_dir,
            nx=contour_nx,
            ny=contour_ny,
        )

        plot_pressure_contours_comparison(
            porous_solver=aero_solver,
            porous_result=coupled.aero_result,
            solid_solver=aero_solver,
            solid_result=solid_result,
            output_dir=model_dir,
            nx=contour_nx,
            ny=contour_ny,
        )

        plot_difference_contours(
            porous_solver=aero_solver,
            porous_result=coupled.aero_result,
            solid_solver=aero_solver,
            solid_result=solid_result,
            output_dir=model_dir,
            nx=contour_nx,
            ny=contour_ny,
        )
    else:
        print("[Plot] Skipping contour plots because --no-contours was used.")

    print(
        "[Sweep] Running porous + solid panel AoA sweep "
        f"from {AOA_SWEEP_START_DEG:g} to {AOA_SWEEP_END_DEG:g} deg "
        f"in {AOA_SWEEP_STEP_DEG:g} deg steps..."
    )

    if EXPORT_AOA_SWEEP_PARAVIEW:
        print("[ParaView] Exporting all AoA sweep cases for ParaView...")

    sweep_df = run_porous_aoa_sweep_for_model(
        model=model,
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=geom.num_pan,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
        reynolds_external=REYNOLDS_EXTERNAL,
        coupling=COUPLING,
        limits=GeometryLimits(),
        alpha_start=AOA_SWEEP_START_DEG,
        alpha_end=AOA_SWEEP_END_DEG,
        alpha_step=AOA_SWEEP_STEP_DEG,
        export_paraview=EXPORT_AOA_SWEEP_PARAVIEW,
        paraview_output_dir=model_dir / "aoa_sweep_paraview",
        field_nx=contour_nx,
        field_ny=contour_ny,
    )

    xfoil_sweep_df = run_xfoil_aoa_sweep_for_case(
        airfoil_name=AIRFOIL_NAME,
        output_dir=xfoil_model_dir,
        alpha_start=AOA_SWEEP_START_DEG,
        alpha_end=AOA_SWEEP_END_DEG,
        alpha_step=AOA_SWEEP_STEP_DEG,
    )

    if xfoil_sweep_df is not None:
        sweep_df = add_xfoil_to_sweep_comparison(
            sweep_df=sweep_df,
            xfoil_sweep_df=xfoil_sweep_df,
        )

    sweep_csv_path = model_dir / "aoa_sweep_comparison.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"[CSV Saved] {sweep_csv_path}")

    plot_aoa_sweep_comparison(
        sweep_df=sweep_df,
        output_dir=model_dir,
    )

    return summary_row
