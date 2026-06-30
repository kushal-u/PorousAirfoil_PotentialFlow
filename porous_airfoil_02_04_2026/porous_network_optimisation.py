"""
porous_network_optimisation.py

Porous-airfoil optimisation module with a lumped hydraulic-resistance model.

Model summary
-------------
The two surface openings are coupled through one equivalent internal hydraulic
resistance Rs. The internal passage is not split into separate pore, throat,
or chamber losses. Instead, the full internal connection is represented by

    Q = (p1 - p2) / Rs

where:
- Q  is the volumetric flow rate between the two surface openings
- p1 is the external pressure at pore 1
- p2 is the external pressure at pore 2
- Rs is the equivalent viscous resistance of the internal passage

This preserves the modular structure of the newer code while matching the
physics of the older reduced-order porous model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution

from solver import (
    CouplingConfig,
    FlowConfig,
    PanelGeometry,
    SPVPResult,
    SourceVortexPanelMethod,
    SurfacePoint,
)


# =============================================================================
# OPTIMISATION CONFIGURATION
# =============================================================================
@dataclass
class OptimizationConfig:
    """
    Differential Evolution settings and geometric limits.

    Notes
    -----
    The internal hydraulic model is lumped. The optimiser searches for
    two opening locations and sizes, plus an effective internal hydraulic
    diameter used to build the equivalent resistance and enforce fit
    constraints.
    """

    x_min_frac: float = 0.01
    x_max_frac: float = 0.98

    d_min: float = 0.0020
    d_max: float = 0.0200
    thickness_fraction_limit: float = 0.85
    min_gap: float = 0.0020

    chamber_dh_min: float = 0.002
    chamber_dh_max: float = 0.020

    span_width: float = 0.02
    q_max: float = 5e-3
    vn_max: float = 1.2
    min_panels_per_pore: int = 2

    target_lift_gain: float = 0.10
    target_gain_penalty: float = 0.0

    de_maxiter: int = 120
    de_popsize: int = 28
    de_seed: int = 2
    polish: bool = True

    vn_penalty_weight: float = 1.0
    q_penalty_weight: float = 1.0
    cd_penalty_weight: float = 20.0


# =============================================================================
# NETWORK DATACLASSES
# =============================================================================
@dataclass
class Pore:
    """
    Single opening on the airfoil surface.

    Parameters
    ----------
    x_frac : float
        Chord fraction x/c.
    side : str
        Surface side: 'upper' or 'lower'.
    diameter : float
        Opening diameter in metres.
    """

    x_frac: float
    side: str
    diameter: float

    def __post_init__(self) -> None:
        self.side = self.side.lower()
        if self.side not in ("upper", "lower"):
            raise ValueError("Pore side must be 'upper' or 'lower'.")
        if not (0.0 <= self.x_frac <= 1.0):
            raise ValueError("x_frac must lie between 0 and 1.")
        if self.diameter <= 0.0:
            raise ValueError("Pore diameter must be positive.")

    @property
    def area(self) -> float:
        """Return the geometric opening area."""
        return 0.25 * np.pi * self.diameter**2


@dataclass
class Chamber:
    """
    Effective internal passage properties.

    Parameters
    ----------
    hydraulic_diameter : float
        Effective hydraulic diameter Dh used in the lumped resistance model.
    area : float
        Effective cross-sectional area of the internal passage.
    friction_scale : float, optional
        Scalar multiplier applied to the equivalent resistance.
    geometry_type : str, optional
        'any', 'circular', or 'rectangle'.
    width, height : float | None, optional
        Optional rectangular dimensions if a rectangular passage is desired.
    """

    hydraulic_diameter: float
    area: float
    friction_scale: float = 1.0
    geometry_type: str = "any"
    width: float | None = None
    height: float | None = None

    def __post_init__(self) -> None:
        if self.hydraulic_diameter <= 0.0:
            raise ValueError("hydraulic_diameter must be positive.")
        if self.area <= 0.0:
            raise ValueError("area must be positive.")
        if self.friction_scale <= 0.0:
            raise ValueError("friction_scale must be positive.")
        self.geometry_type = self.geometry_type.lower()
        if self.geometry_type not in ("any", "circular", "rectangle"):
            raise ValueError("geometry_type must be 'any', 'circular', or 'rectangle'.")


@dataclass
class NetworkState:
    """
    Solved state of the lumped porous connection.

    Attributes
    ----------
    Q : float
        Volumetric flow rate. Positive means pore1 -> pore2.
    p1, p2 : float
        External pressures at the two openings.
    p_internal_1, p_internal_2 : float
        Internal pressure states stored for reporting. In the lumped model,
        these are taken equal to p1 and p2.
    dp_total : float
        Total pressure drop across the internal passage.
    Rs : float
        Equivalent hydraulic resistance.
    reynolds_equivalent : float
        Equivalent Reynolds number formed with chamber area and hydraulic
        diameter. Diagnostic only.
    """

    Q: float
    p1: float
    p2: float
    p_internal_1: float
    p_internal_2: float
    dp_total: float
    Rs: float
    reynolds_equivalent: float


@dataclass
class CoupledResult:
    """Output of the coupled porous-airfoil solve."""

    aero_result: SPVPResult
    network_state: NetworkState
    normal_transpiration: np.ndarray
    converged: bool
    iterations: int
    max_vn: float
    max_vn_over_vinf: float
    q_history: list[float] = field(default_factory=list)
    cl_history: list[float] = field(default_factory=list)


@dataclass
class Topology:
    """Fixed topology for one optimisation run."""

    pore1_side: str
    pore2_side: str


@dataclass
class OptimizationResult:
    """Best result returned from one topology-specific DE run."""

    topology: Topology
    x_opt: np.ndarray
    best_fun: float
    coupled_result: CoupledResult | None
    success: bool
    message: str


# =============================================================================
# HYDRAULIC RESISTANCE HELPER
# =============================================================================
def hydraulic_resistance(
    mu: float,
    length: float,
    geometry_type: str = "any",
    geometry: tuple[float, ...] = (1.0,),
) -> tuple[float, float, float]:
    """
    Compute a lumped hydraulic resistance from simple duct geometry.

    Parameters
    ----------
    mu : float
        Dynamic viscosity.
    length : float
        Effective passage length.
    geometry_type : str, optional
        One of:
        - 'circular'  : geometry = (radius,)
        - 'rectangle' : geometry = (width, height)
        - 'any'       : geometry = (hydraulic_diameter,)
    geometry : tuple[float, ...], optional
        Geometry tuple matching the chosen geometry_type.

    Returns
    -------
    tuple[float, float, float]
        Rs, Dh, A = equivalent hydraulic resistance, hydraulic diameter, area.
    """
    geometry_type = geometry_type.lower()

    if geometry_type == "circular" and len(geometry) == 1:
        radius = float(geometry[0])
        if radius <= 0.0:
            raise ValueError("Circular radius must be positive.")
        Dh = 2.0 * radius
        Rs = 128.0 * mu * length / max(np.pi * Dh**4, 1e-30)
        A = np.pi * radius**2
        return float(Rs), float(Dh), float(A)

    if geometry_type == "rectangle" and len(geometry) == 2:
        width = float(geometry[0])
        height = float(geometry[1])
        if width <= 0.0 or height <= 0.0:
            raise ValueError("Rectangle width and height must be positive.")
        Dh = 4.0 * width * height / (2.0 * (width + height))
        h_small = min(width, height)
        h_large = max(width, height)
        correction = max(1.0 - 0.63 * (h_small / h_large), 1e-12)
        Rs = 12.0 * mu * length / max((h_large * h_small**3) * correction, 1e-30)
        A = width * height
        return float(Rs), float(Dh), float(A)

    if geometry_type == "any" and len(geometry) == 1:
        Dh = float(geometry[0])
        if Dh <= 0.0:
            raise ValueError("Hydraulic diameter must be positive.")
        Rs = 128.0 * mu * length / max(np.pi * Dh**4, 1e-30)
        A = 0.25 * np.pi * Dh**2
        return float(Rs), float(Dh), float(A)

    raise ValueError(f"Invalid hydraulic resistance specification: {geometry_type=}, {geometry=}")


# =============================================================================
# LUMPED TWO-OPENING MODEL
# =============================================================================
class TwoPoreOneChamberNetwork:
    """
    Two surface openings connected by one lumped equivalent resistance.

    The internal passage between the two openings is modeled by one hydraulic
    resistance Rs, so that:

        Q = (p1 - p2) / Rs
    """

    def __init__(self, pore1: Pore, pore2: Pore, chamber: Chamber, span_width: float = 0.02):
        self.pore1 = pore1
        self.pore2 = pore2
        self.chamber = chamber
        self.span_width = float(span_width)

        if self.span_width <= 0.0:
            raise ValueError("span_width must be positive.")

    def pore_surface_point(self, geom: PanelGeometry, pore: Pore) -> SurfacePoint:
        """Map a pore definition to a surface point on the airfoil."""
        return geom.surface_point_from_fraction(pore.x_frac, pore.side)

    def derived_passage_length(self, geom: PanelGeometry) -> float:
        """Derive an effective internal passage length between the two openings."""
        p1 = self.pore_surface_point(geom, self.pore1)
        p2 = self.pore_surface_point(geom, self.pore2)

        dx = p2.x - p1.x
        dy = p2.y - p1.y
        straight = np.hypot(dx, dy)

        if self.pore1.side == self.pore2.side:
            return max(1.10 * straight, 1e-6)

        x_mid = 0.5 * (self.pore1.x_frac + self.pore2.x_frac)
        t_mid = geom.local_thickness_at_fraction(x_mid)
        return max(np.hypot(abs(dx), t_mid), 1e-6)

    def validate_geometry(self, geom: PanelGeometry, opt: OptimizationConfig) -> None:
        """Enforce geometric constraints on the opening pair and internal passage."""
        p1 = self.pore_surface_point(geom, self.pore1)
        p2 = self.pore_surface_point(geom, self.pore2)

        if self.pore1.x_frac > self.pore2.x_frac:
            raise ValueError("Ordering rule violated: require x1 <= x2.")

        if not (opt.x_min_frac <= self.pore1.x_frac <= opt.x_max_frac):
            raise ValueError("Pore 1 violates LE/TE exclusion zone.")
        if not (opt.x_min_frac <= self.pore2.x_frac <= opt.x_max_frac):
            raise ValueError("Pore 2 violates LE/TE exclusion zone.")

        t1 = geom.local_thickness_at_fraction(self.pore1.x_frac)
        t2 = geom.local_thickness_at_fraction(self.pore2.x_frac)

        if self.pore1.diameter > opt.thickness_fraction_limit * t1:
            raise ValueError("Pore 1 diameter exceeds local thickness limit.")
        if self.pore2.diameter > opt.thickness_fraction_limit * t2:
            raise ValueError("Pore 2 diameter exceeds local thickness limit.")

        center_dist = np.hypot(p2.x - p1.x, p2.y - p1.y)
        required = 0.5 * (self.pore1.diameter + self.pore2.diameter) + opt.min_gap
        if center_dist < required:
            raise ValueError("Pores overlap or violate minimum gap.")

        xs = np.linspace(self.pore1.x_frac, self.pore2.x_frac, 25)
        tmin = min(geom.local_thickness_at_fraction(x) for x in xs)
        if self.chamber.hydraulic_diameter > 0.7 * tmin:
            raise ValueError("Effective hydraulic diameter violates fit constraint.")

    def pressure_at_pore(self, panel_pressures: np.ndarray, geom: PanelGeometry, pore: Pore) -> float:
        """Get the external pressure at the panel associated with a pore."""
        sp = self.pore_surface_point(geom, pore)
        return float(panel_pressures[sp.panel_id])

    def equivalent_resistance(self, geom: PanelGeometry, flow: FlowConfig) -> float:
        """Return the equivalent internal hydraulic resistance Rs."""
        length = self.derived_passage_length(geom)

        if self.chamber.geometry_type == "rectangle" and self.chamber.width and self.chamber.height:
            Rs, _, _ = hydraulic_resistance(
                mu=flow.mu_inf,
                length=length,
                geometry_type="rectangle",
                geometry=(self.chamber.width, self.chamber.height),
            )
        elif self.chamber.geometry_type == "circular":
            radius = 0.5 * self.chamber.hydraulic_diameter
            Rs, _, _ = hydraulic_resistance(
                mu=flow.mu_inf,
                length=length,
                geometry_type="circular",
                geometry=(radius,),
            )
        else:
            Rs, _, _ = hydraulic_resistance(
                mu=flow.mu_inf,
                length=length,
                geometry_type="any",
                geometry=(self.chamber.hydraulic_diameter,),
            )

        Rs *= self.chamber.friction_scale
        return max(float(Rs), 1e-16)

    def equivalent_reynolds(self, Q_abs: float, flow: FlowConfig) -> float:
        """Equivalent Reynolds number based on chamber area and hydraulic diameter."""
        U = Q_abs / max(self.chamber.area, 1e-16)
        return flow.rho_inf * abs(U) * self.chamber.hydraulic_diameter / max(flow.mu_inf, 1e-16)

    def solve(self, geom: PanelGeometry, panel_pressures: np.ndarray, flow: FlowConfig) -> NetworkState:
        """
        Solve the lumped internal flow rate from the two surface pressures.

        Positive Q means pore1 -> pore2.
        """
        p1 = self.pressure_at_pore(panel_pressures, geom, self.pore1)
        p2 = self.pressure_at_pore(panel_pressures, geom, self.pore2)

        Rs = self.equivalent_resistance(geom, flow)
        Q = (p1 - p2) / Rs

        return NetworkState(
            Q=float(Q),
            p1=float(p1),
            p2=float(p2),
            p_internal_1=float(p1),
            p_internal_2=float(p2),
            dp_total=float(abs(p1 - p2)),
            Rs=float(Rs),
            reynolds_equivalent=float(self.equivalent_reynolds(abs(Q), flow)),
        )

    def _covered_panels(
        self,
        geom: PanelGeometry,
        center_panel_id: int,
        side: str,
        opening_diameter: float,
        min_panels_per_pore: int = 2,
    ) -> np.ndarray:
        """
        Return all panels covered by the pore opening on the requested side.
        """
        if side == "upper":
            side_ids = np.asarray(geom.upper_panel_ids, dtype=int)
        elif side == "lower":
            side_ids = np.asarray(geom.lower_panel_ids, dtype=int)
        else:
            raise ValueError("side must be 'upper' or 'lower'.")

        if side_ids.size == 0:
            raise RuntimeError(f"No panels found on side '{side}'.")

        matches = np.where(side_ids == center_panel_id)[0]
        if matches.size == 0:
            k = int(np.argmin(np.abs(side_ids - center_panel_id)))
        else:
            k = int(matches[0])

        selected = [int(side_ids[k])]
        total_len = geom.S[side_ids[k]]

        left = k - 1
        right = k + 1

        while (total_len < opening_diameter) or (len(selected) < min_panels_per_pore):
            added = False

            if left >= 0:
                selected.append(int(side_ids[left]))
                total_len += geom.S[side_ids[left]]
                left -= 1
                added = True

            if (total_len >= opening_diameter) and (len(selected) >= min_panels_per_pore):
                break

            if right < side_ids.size:
                selected.append(int(side_ids[right]))
                total_len += geom.S[side_ids[right]]
                right += 1
                added = True

            if not added:
                break

        return np.array(sorted(set(selected)), dtype=int)

    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        state: NetworkState,
        min_panels_per_pore: int = 2,
    ) -> np.ndarray:
        """
        Convert pore flow into equivalent panel transpiration.
        """
        vn = np.zeros(geom.num_pan, dtype=float)

        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        ids1 = self._covered_panels(
            geom=geom,
            center_panel_id=sp1.panel_id,
            side=self.pore1.side,
            opening_diameter=self.pore1.diameter,
            min_panels_per_pore=min_panels_per_pore,
        )
        ids2 = self._covered_panels(
            geom=geom,
            center_panel_id=sp2.panel_id,
            side=self.pore2.side,
            opening_diameter=self.pore2.diameter,
            min_panels_per_pore=min_panels_per_pore,
        )

        q = float(state.Q)

        q1 = -q
        q2 = +q

        area1 = self.span_width * np.sum(geom.S[ids1])
        area2 = self.span_width * np.sum(geom.S[ids2])

        vn1 = q1 / max(area1, 1e-16)
        vn2 = q2 / max(area2, 1e-16)

        vn[ids1] += vn1
        vn[ids2] += vn2

        return vn


# =============================================================================
# COUPLED SOLVER
# =============================================================================
class CoupledPorousAirfoilSolver:
    """Coupled porous-airfoil solver using fixed-point iteration."""

    def __init__(
        self,
        aero_solver: SourceVortexPanelMethod,
        network: TwoPoreOneChamberNetwork,
        coupling: CouplingConfig,
        opt: OptimizationConfig | None = None,
    ):
        self.aero_solver = aero_solver
        self.network = network
        self.coupling = coupling
        self.opt = opt

    def solve(self, verbose: bool = False) -> CoupledResult:
        """Solve the coupled porous-airfoil problem."""
        geom = self.aero_solver.geom
        flow = self.aero_solver.flow

        if self.opt is not None:
            self.network.validate_geometry(geom, self.opt)

        vn = np.zeros(geom.num_pan)
        prev_q = None
        q_history: list[float] = []
        cl_history: list[float] = []

        converged = False
        aero_result: SPVPResult | None = None
        network_state: NetworkState | None = None

        for it in range(1, self.coupling.max_iter + 1):
            aero_result = self.aero_solver.solve(normal_transpiration=vn)
            panel_pressures = self.aero_solver.panel_pressures(aero_result)

            network_state = self.network.solve(geom, panel_pressures, flow)
            min_panels = self.opt.min_panels_per_pore if self.opt is not None else 2
            vn_target = self.network.transpiration_from_flow(
                geom,
                network_state,
                min_panels_per_pore=min_panels,
            )

            if self.opt is not None:
                if abs(network_state.Q) > self.opt.q_max:
                    raise ValueError("Maximum flow-rate constraint violated.")
                if np.max(np.abs(vn_target)) > self.opt.vn_max:
                    raise ValueError("Maximum transpiration constraint violated.")

            vn_new = (1.0 - self.coupling.relaxation) * vn + self.coupling.relaxation * vn_target
            dvn = np.max(np.abs(vn_new - vn))
            dq = np.inf if prev_q is None else abs(network_state.Q - prev_q)

            if verbose:
                max_vn_iter = float(np.max(np.abs(vn_target)))
                max_vn_iter_ratio = max_vn_iter / max(flow.v_inf, 1e-16)
                print(
                    f"[Coupling] iter={it:02d} | "
                    f"CL={aero_result.CL: .6f} | "
                    f"Q={network_state.Q: .6e} m^3/s | "
                    f"Rs={network_state.Rs: .6e} Pa·s/m^3 | "
                    f"max|vn|={max_vn_iter: .3e} m/s | "
                    f"max|vn|/Vinf={max_vn_iter_ratio: .3e} | "
                    f"dvn={dvn: .3e} | dq={dq: .3e}"
                )

            vn = vn_new
            prev_q = network_state.Q
            q_history.append(network_state.Q)
            cl_history.append(aero_result.CL)

            if dvn < self.coupling.tol_vn and dq < self.coupling.tol_q:
                converged = True
                break

        aero_result = self.aero_solver.solve(normal_transpiration=vn)
        panel_pressures = self.aero_solver.panel_pressures(aero_result)
        network_state = self.network.solve(geom, panel_pressures, flow)

        max_vn = float(np.max(np.abs(vn)))
        max_vn_over_vinf = max_vn / max(flow.v_inf, 1e-16)

        return CoupledResult(
            aero_result=aero_result,
            network_state=network_state,
            normal_transpiration=vn,
            converged=converged,
            iterations=it,
            max_vn=max_vn,
            max_vn_over_vinf=max_vn_over_vinf,
            q_history=q_history,
            cl_history=cl_history,
        )


# =============================================================================
# OPTIMISER
# =============================================================================
class PorousNetworkOptimizer:
    """Differential Evolution optimiser for the lumped porous network."""

    def __init__(
        self,
        geom: PanelGeometry,
        aero_solver: SourceVortexPanelMethod,
        coupling: CouplingConfig,
        opt: OptimizationConfig,
    ):
        self.geom = geom
        self.aero_solver = aero_solver
        self.coupling = coupling
        self.opt = opt
        self._eval_counter = 0
        self.saved_design_rows: list[dict] = []

        self.solid_result = self.aero_solver.solve()
        self.solid_cl = float(self.solid_result.CL)
        self.target_cl = (1.0 + self.opt.target_lift_gain) * self.solid_cl

    def design_to_network(self, x: np.ndarray, topology: Topology) -> TwoPoreOneChamberNetwork:
        """
        Convert a design vector into a physical network.

        Design vector
        -------------
        [x1, x2, d1, d2, chamber_dh]
        """
        x1, x2, d1, d2, dh = x
        x1, x2 = sorted([float(x1), float(x2)])

        pore1 = Pore(
            x_frac=float(x1),
            side=topology.pore1_side,
            diameter=float(d1),
        )
        pore2 = Pore(
            x_frac=float(x2),
            side=topology.pore2_side,
            diameter=float(d2),
        )

        dh = float(dh)
        area = 0.25 * np.pi * dh**2

        chamber = Chamber(
            hydraulic_diameter=dh,
            area=area,
            friction_scale=1.0,
            geometry_type="any",
        )

        return TwoPoreOneChamberNetwork(
            pore1=pore1,
            pore2=pore2,
            chamber=chamber,
            span_width=self.opt.span_width,
        )

    def objective(self, x: np.ndarray, topology: Topology) -> float:
        """
        Objective for Differential Evolution.

        Goal
        ----
        - maximise lift gain over the solid baseline
        - discourage overly strong porous actuation
        - discourage excessive internal flow
        - discourage excessive drag increase

        Also
        ----
        Save every evaluated design whose lift gain is at least 5%, even if the
        coupled solver did not converge, provided a coupled result was produced.
        """
        self._eval_counter += 1

        coupled = None
        network = None

        try:
            network = self.design_to_network(x, topology)

            coupled = CoupledPorousAirfoilSolver(
                aero_solver=self.aero_solver,
                network=network,
                coupling=self.coupling,
                opt=self.opt,
            ).solve(verbose=False)

            cl = float(coupled.aero_result.CL)
            cd = float(coupled.aero_result.CD)
            cm = float(coupled.aero_result.CM)

            gain = cl / self.solid_cl - 1.0
            gain_percent = 100.0 * gain

            vn_ratio = float(coupled.max_vn_over_vinf)
            q_ratio = abs(coupled.network_state.Q) / max(self.opt.q_max, 1e-12)
            drag_increase = max(0.0, cd - self.solid_result.CD)

            f = (
                -gain
                + self.opt.vn_penalty_weight * vn_ratio**2
                + self.opt.q_penalty_weight * q_ratio**2
                + self.opt.cd_penalty_weight * drag_increase
            )

            if gain_percent >= 5.0:
                self.saved_design_rows.append(
                    {
                        "eval_id": int(self._eval_counter),
                        "topology_pore1_side": topology.pore1_side,
                        "topology_pore2_side": topology.pore2_side,
                        "x1_frac": float(network.pore1.x_frac),
                        "x2_frac": float(network.pore2.x_frac),
                        "d1_m": float(network.pore1.diameter),
                        "d2_m": float(network.pore2.diameter),
                        "effective_hydraulic_diameter_m": float(network.chamber.hydraulic_diameter),
                        "CL": cl,
                        "CD": cd,
                        "CM": cm,
                        "gain_percent": float(gain_percent),
                        "Q_m3_s": float(coupled.network_state.Q),
                        "Rs_Pa_s_per_m3": float(coupled.network_state.Rs),
                        "dp_total_Pa": float(coupled.network_state.dp_total),
                        "Re_equivalent": float(coupled.network_state.reynolds_equivalent),
                        "max_vn_m_s": float(coupled.max_vn),
                        "max_vn_over_vinf": float(coupled.max_vn_over_vinf),
                        "coupling_converged": bool(coupled.converged),
                        "coupling_iterations": int(coupled.iterations),
                        "objective_value": float(f),
                    }
                )

            if self._eval_counter % 20 == 0:
                print(
                    f"[DE Eval {self._eval_counter:05d}] "
                    f"topology=({topology.pore1_side},{topology.pore2_side}) | "
                    f"CL={cl: .6f} | "
                    f"gain={gain_percent: .3f}% | "
                    f"Q={coupled.network_state.Q: .6e} | "
                    f"max_vn/Vinf={vn_ratio: .4f} | "
                    f"CD={cd: .6e} | "
                    f"converged={coupled.converged}"
                )

            if not coupled.converged:
                return 1e6

            return float(f)

        except Exception as exc:
            if self._eval_counter % 20 == 0:
                print(
                    f"[DE Eval {self._eval_counter:05d}] "
                    f"topology=({topology.pore1_side},{topology.pore2_side}) | "
                    f"infeasible | reason={exc}"
                )
            return 1e6

    def bounds(self) -> list[tuple[float, float]]:
        """Return DE bounds for [x1, x2, d1, d2, chamber_dh]."""
        return [
            (self.opt.x_min_frac, self.opt.x_max_frac),
            (self.opt.x_min_frac, self.opt.x_max_frac),
            (self.opt.d_min, self.opt.d_max),
            (self.opt.d_min, self.opt.d_max),
            (self.opt.chamber_dh_min, self.opt.chamber_dh_max),
        ]

    def optimize_topology(self, topology: Topology) -> OptimizationResult:
        """Run Differential Evolution for one fixed topology."""
        self._eval_counter = 0

        print("=" * 72)
        print(f"Starting DE for topology: ({topology.pore1_side}, {topology.pore2_side})")
        print("=" * 72)

        result = differential_evolution(
            func=lambda xx: self.objective(xx, topology),
            bounds=self.bounds(),
            maxiter=self.opt.de_maxiter,
            popsize=self.opt.de_popsize,
            seed=self.opt.de_seed,
            polish=self.opt.polish,
            updating="deferred",
            workers=1,
        )

        coupled_best = None
        try:
            network = self.design_to_network(result.x, topology)
            candidate = CoupledPorousAirfoilSolver(
                aero_solver=self.aero_solver,
                network=network,
                coupling=self.coupling,
                opt=self.opt,
            ).solve(verbose=True)
            if candidate.converged:
                coupled_best = candidate
        except Exception:
            pass

        print(f"[DE Finished] topology=({topology.pore1_side},{topology.pore2_side})")
        print(f"success = {result.success}")
        print(f"message = {result.message}")
        print(f"best f  = {result.fun:.8f}")
        print(f"x_opt   = {result.x}")
        print()

        return OptimizationResult(
            topology=topology,
            x_opt=np.asarray(result.x, dtype=float),
            best_fun=float(result.fun),
            coupled_result=coupled_best,
            success=bool(result.success),
            message=str(result.message),
        )

    def optimize_all_topologies(self) -> list[OptimizationResult]:
        """Run DE separately for all four topologies."""
        topologies = [
            Topology("upper", "upper"),
            Topology("lower", "lower"),
            Topology("upper", "lower"),
            Topology("lower", "upper"),
        ]
        return [self.optimize_topology(topo) for topo in topologies]

    def shortlist_results(
        self,
        results: list[OptimizationResult],
        max_candidates: int = 5,
        max_vn_over_vinf: float = 0.05,
        max_q: float | None = None,
    ) -> list[OptimizationResult]:
        """
        Return a shortlist of the most physically plausible converged designs.
        """
        shortlisted: list[OptimizationResult] = []

        for r in results:
            if r.coupled_result is None:
                continue
            if not r.coupled_result.converged:
                continue
            if r.coupled_result.max_vn_over_vinf > max_vn_over_vinf:
                continue
            if max_q is not None and abs(r.coupled_result.network_state.Q) > max_q:
                continue
            shortlisted.append(r)

        shortlisted.sort(key=lambda rr: rr.coupled_result.aero_result.CL, reverse=True)
        return shortlisted[:max_candidates]