"""
Core data model for Nexus-Energy.

Defines: EnergySystem, Bus, Carrier, Generator, Storage, Load, Link,
and the constraint generation + solve pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import nexus as nx
except ImportError:
    import nexus_opt as nx

from .components.thermal import (
    add_must_run,
    add_regulation_reserve_vars,
    build_pwl_heat_rate,
    build_three_bin_uc,
)
from . import network as _network


# ---------------------------------------------------------------------------
# Economics helpers
# ---------------------------------------------------------------------------


def annuity(overnight: float, rate: float, lifetime: int,
            prepaid: bool = False) -> float:
    """Convert an overnight capital cost to an annualised payment.

    Args:
        overnight: total up-front investment, typically $/MW.
        rate: per-period discount rate (e.g. 0.07 for 7 %).
        lifetime: number of periods over which the investment is
            amortised.
        prepaid: when True use the annuity-due form
            ``r / ((1+r)(1 − (1+r)^{−T}))`` (Tulipa / PyPSA
            convention — first payment at t=0). When False use the
            in-arrears capital recovery factor
            ``r / (1 − (1+r)^{−T})`` (standard engineering economics).
            Default False.

    Returns:
        Equivalent annualised $/MW-year to assign to
        ``Generator.capital_cost`` / ``Storage.capital_cost_*``.

    Edge cases: ``lifetime <= 0`` or ``rate <= 0`` → returns
    ``overnight`` unchanged (flat, no time value of money).
    """
    if lifetime <= 0 or rate <= 0:
        return overnight
    denom = 1.0 - (1.0 + rate) ** (-lifetime)
    if prepaid:
        denom = (1.0 + rate) * denom
    return overnight * rate / denom


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

def _mat_is_one_hot(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff every row of ``matrix`` is one-hot (a degenerate distribution).

    A one-hot mapping_matrix reproduces the integer ``mapping`` exactly, so the
    Phase 16.5 fractional Kotzur path collapses to the original integer
    recursion — used to decide whether the weighted combination is needed.
    """
    m = np.asarray(matrix, dtype=float)
    return bool(np.all(np.abs(m - (m > (1.0 - tol))) < tol))


@dataclass
class Carrier:
    """An energy carrier type (electricity, heat, hydrogen, etc.)."""
    name: str
    unit: str = "MWh"

    def __repr__(self) -> str:
        return f"Carrier({self.name!r}, unit={self.unit!r})"


# Default carriers (users can also create custom ones)
CARRIERS = {
    "electricity": Carrier("electricity", "MWh"),
    "heat": Carrier("heat", "MWh_th"),
    "heat_high": Carrier("heat_high", "MWh_th"),
    "hydrogen": Carrier("hydrogen", "MWh"),
    "natural_gas": Carrier("natural_gas", "MWh_th"),
    "biomass": Carrier("biomass", "MWh_th"),
    "co2": Carrier("co2", "tCO2"),
    "water": Carrier("water", "m3"),
}


@dataclass
class Bus:
    """A connection point where components meet. Typed by carrier."""
    name: str
    carrier: Carrier
    _id: int = -1
    # DC-OPF phase-angle variables; populated by ``optimise()`` when any
    # connected Link is marked ``model_type='dc_opf'``.
    _theta_vars: list = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return f"Bus({self.name!r}, carrier={self.carrier.name!r})"


@dataclass
class Generator:
    """A component that produces energy on a bus."""
    name: str
    bus: Bus
    capacity: float  # MW (or carrier-appropriate unit)
    marginal_cost: float = 0.0  # $/MWh
    capital_cost: float = 0.0  # $/MW/year (annualised)
    p_min: float = 0.0  # MW — minimum stable output
    carrier_factor: Optional[np.ndarray] = None  # time-varying availability [0,1]
    emission_factor: float = 0.0  # tCO2/MWh
    co2_output_bus: Optional[Bus] = None  # if set, dispatch emits CO₂ onto this bus as a physical flow
    co2_output_factor: float = 0.0  # carrier-unit per MWh dispatched (e.g., tCO₂/MWh)
    ramp_up: Optional[float] = None  # MW/timestep
    ramp_down: Optional[float] = None  # MW/timestep
    # Phase 14 — fractional ramp (fraction of installed capacity per timestep).
    # When set AND the gen is extendable, the constraint is
    # ``p[t]-p[t-1] <= ramp_up_frac × cap_var`` (linear in cap_var). When the
    # gen is fixed-cap, this is multiplied by `capacity` at build time.
    ramp_up_frac: Optional[float] = None
    ramp_down_frac: Optional[float] = None
    committable: bool = False  # requires unit commitment (binary on/off)
    initial_status: int = 0  # 0=off, 1=on at t=-1 (PyPSA up_time_before > 0 → 1)
    up_time_before: int = 0  # hours already on before t=0 (PyPSA up_time_before)
    down_time_before: int = 0  # hours already off before t=0 (PyPSA down_time_before)
    min_up_time: int = 0  # timesteps
    min_down_time: int = 0  # timesteps
    startup_cost: float = 0.0  # $/startup
    shutdown_cost: float = 0.0  # $/shutdown (Morales-España 3-bin)
    # Phase 2.x — energy-per-start fuel cost. GenX ``Start_Fuel_MMBTU_per_MW``,
    # SpineOpt ``start_up_fuel``, Sienna ``startup_fuel``. Accounting-separate
    # from ``startup_cost`` so users can report fuel vs fixed startup overhead
    # independently; both are summed onto the same v[t] coefficient at build
    # time. Users map GenX-style data via
    # ``startup_fuel_cost = Start_Fuel_MMBTU_per_MW × cap_size × fuel_price``.
    startup_fuel_cost: float = 0.0
    # Phase 16.x — no-load / idling cost per on-unit per timestep.
    # Applied as ``no_load_cost × u[t] × dt`` for committable gens; lets the
    # user model the fuel / maintenance overhead a unit accrues just by being
    # synced (Tulipa ``units_on_cost``, PLEXOS NoLoadCost).
    no_load_cost: float = 0.0
    # Phase 2.2 — multi-state (hot / warm / cold) startup cost. List of
    # ``(min_off_timesteps, startup_cost)`` tuples, sorted ascending by
    # ``min_off_timesteps``, describing the temperature-dependent start cost
    # (GenX multi-stage start, PLEXOS/PowerSimulations hot/warm/cold).
    # ``min_off_timesteps`` is the minimum number of consecutive off steps
    # *before* a start for that segment's cost to apply; the first entry should
    # be ``0`` (hottest, immediate restart) and costs must be non-decreasing.
    # When set, overrides the flat ``startup_cost`` and uses the tight
    # Morales-España (2013) start-type formulation. Binary UC only (clustered
    # falls back to flat ``startup_cost``).
    start_up_segments: Optional[list] = None
    must_run: bool = False  # always on — u[t]=1 if committable, else p[t]>=capacity
    # Clustered linearized UC (GenX UCommit: 2 analogue): n_units identical units
    # lumped together; u, v, w continuous in [0, n_units].
    clustered: bool = False
    n_units: int = 1
    # PWL heat rate / part-load efficiency: list of (p_MW_breakpoint, marginal_cost_$/MWh)
    # points defining convex increasing piecewise-linear fuel cost. Overrides flat
    # `marginal_cost` when set. Breakpoints must be sorted ascending in p.
    heat_rate_segments: Optional[list] = None
    # Regulation reserves (fraction of capacity each unit can contribute)
    reg_up_max: float = 0.0
    reg_down_max: float = 0.0
    extendable: bool = False  # capacity is a decision variable
    max_capacity: float = float("inf")  # upper bound for extendable
    min_capacity: float = 0.0  # lower bound for extendable (PyPSA p_nom_min)
    tech: Optional[str] = None  # technology tag for bucket carveouts (e.g. "solar", "wind")
    # Phase 5 — investment planning depth
    fixed_om: float = 0.0  # $/MW/year — paid on cap_var even when not dispatched
    integer_investment: bool = False  # cap_var = unit_size × integer count
    unit_size: float = 1.0  # MW per discrete unit (only used when integer_investment=True)
    # PWL CapEx (economies of scale): list of (cap_breakpoint_MW, $/MW) sorted
    # ascending in MW with strictly decreasing slopes. Overrides flat
    # ``capital_cost`` when set.
    capex_segments: Optional[list] = None
    retire_at_year: Optional[int] = None  # scheduled retirement year (multi-stage only)
    build_year: Optional[int] = None  # commissioning year for vintage tracking (multi-stage)
    lifetime_years: int = 25  # plant life; vintage at build_year expires at build_year + lifetime_years
    build_lead_years: int = 0  # construction lead time — new-build from stage S active at year ≥ year(S) + lead
    # Retrofit / fuel-switching: when set, new-build capacity of this gen at
    # stage S is upper-bounded by the amount of ``retrofit_of`` capacity that
    # is retiring between S-1 and S. Lets users model coal → biomass, oil
    # boiler → heat pump etc. at reduced ``capital_cost`` (the discount vs
    # greenfield is captured by the user-set capital_cost itself).
    retrofit_of: Optional[str] = None
    # Internal — set during model build
    _p_vars: list = field(default_factory=list, repr=False)
    _cap_var: object = field(default=None, repr=False)
    _status_vars: list = field(default_factory=list, repr=False)
    _startup_vars: list = field(default_factory=list, repr=False)
    _shutdown_vars: list = field(default_factory=list, repr=False)
    _start_type_vars: list = field(default_factory=list, repr=False)  # [T][n_start_types] — Phase 2.2 multi-state start
    _seg_vars: list = field(default_factory=list, repr=False)  # [T][n_segments] — heat-rate PWL
    _reg_up_vars: list = field(default_factory=list, repr=False)
    _reg_down_vars: list = field(default_factory=list, repr=False)
    _capex_seg_vars: list = field(default_factory=list, repr=False)  # Phase 5 — PWL CapEx
    _capex_seg_slopes: list = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return f"Generator({self.name!r}, bus={self.bus.name!r}, capacity={self.capacity})"


@dataclass
class Storage:
    """A component that stores and releases energy on a bus."""
    name: str
    bus: Bus
    power_capacity: float  # MW — max charge/discharge rate
    energy_capacity: float  # MWh — total storage capacity
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.95
    self_discharge: float = 0.0  # fraction per timestep
    soc_min: float = 0.0  # fraction [0,1]
    soc_max: float = 1.0  # fraction [0,1]
    soc_initial: float = 0.5  # fraction [0,1]
    # 18.P2 (temporal decomposition LB blocks): when True, the initial SOC is
    # a free decision variable in [soc_min, soc_max]·capacity instead of the
    # pinned soc_initial level. This RELAXES the model (any pinned solution
    # stays feasible) — used to build valid per-block lower bounds. Only
    # supported for non-cyclic, non-extendable, non-LDS storages.
    soc_initial_free: bool = False
    # 18.P2 boundary machinery (all default-off, optimum-preserving when
    # unset). ``soc_terminal_min`` (absolute MWh) floors the final-step SOC —
    # a restriction used to de-myopify rolling/stitched solves (also useful
    # as an MPC terminal condition). ``soc_start_cost`` prices the free
    # initial-SOC variable (requires soc_initial_free) and
    # ``soc_terminal_cost`` prices the final-step SOC — the ±λ boundary
    # terms of a temporal Lagrangian dual decomposition.
    soc_terminal_min: Optional[float] = None
    soc_terminal_max: Optional[float] = None
    # Optional absolute-MWh bounds on the FREE initial-SOC variable
    # (requires soc_initial_free; intersected with [soc_min, soc_max]·cap).
    # Used by 18.P2 reachability envelopes: a decomposition block's start
    # SOC cannot exceed what the full problem could physically have stored
    # by that boundary.
    soc_initial_free_min: Optional[float] = None
    soc_initial_free_max: Optional[float] = None
    soc_start_cost: float = 0.0
    soc_terminal_cost: float = 0.0
    cyclic: bool = True  # enforce soc(0) == soc(T)
    # How the cyclic boundary anchors the *level*. "fixed" (default) pins
    # soc(0) == soc_initial·capacity (so the cycle closes at a known level —
    # historical nexus behaviour). "free" only enforces continuity
    # soc(0) == soc(T) and lets the optimiser choose the cheapest cyclic level
    # — this is PyPSA's ``cyclic_state_of_charge`` convention. ``from_pypsa``
    # sets "free" for parity; pinning would over-constrain (and for extendable
    # storage, where energy_capacity starts at 0, would force start/end empty).
    cyclic_level: str = "fixed"  # "fixed" | "free"
    marginal_cost: float = 0.0  # $/MWh on discharge leg (PyPSA convention; var-OM on p_dispatch)
    marginal_cost_charge: float = 0.0  # $/MWh on charge leg (GenX applies a second Var_OM on charging)
    capital_cost_power: float = 0.0  # $/MW/year
    capital_cost_energy: float = 0.0  # $/MWh/year
    extendable: bool = False
    max_power_capacity: float = float("inf")
    max_energy_capacity: float = float("inf")
    min_power_capacity: float = 0.0  # lower bound on power cap (PyPSA p_nom_min)
    min_energy_capacity: float = 0.0  # lower bound on energy cap (PyPSA e_nom_min)
    max_hours: float | None = None  # if set alone, binds energy_cap == power_cap * max_hours (PyPSA StorageUnit)
    min_hours: float | None = None  # if set with max_hours, binds energy_cap into [min_hours, max_hours] window (GenX)
    tech: Optional[str] = None  # technology tag for bucket carveouts
    # Phase 4 — sector coupling depth
    inflow: Optional[np.ndarray] = None  # natural inflow per timestep (MWh / dt). Hydro reservoirs.
    spill_to: Optional["Storage"] = None  # downstream cascade reservoir (receives this one's spill)
    pump_capacity: Optional[float] = None  # MW — overrides power_capacity for the charge leg (PSH)
    turbine_capacity: Optional[float] = None  # MW — overrides power_capacity for the discharge leg (PSH)
    availability: Optional[np.ndarray] = None  # fraction in [0,1] multiplying (dis)charge bounds (V2G)
    # Tiny default ($/MWh) so the LP prefers keeping water over wasting it
    # when truly indifferent. Set 0 to allow free spill; set higher to model
    # economic / environmental opportunity cost (PyPSA spill_cost convention).
    spill_cost: float = 1e-3
    # Phase 14 — timestep SOC pin (PyPSA state_of_charge_set parity).
    # Dict of {timestep_index: soc_value_in_MWh} — emits `soc[t] == value`
    # equalities for the listed t's. MWh units match PyPSA's
    # state_of_charge_set convention. Use None / {} to skip.
    soc_fixed: Optional[dict] = None
    # Phase 7 — long-duration storage (Kotzur 2018 inter-period superposition).
    # When True AND the system has representative periods + chronological
    # mapping, SOC = soc_intra[t in rep] + soc_inter[d_orig], with the
    # inter-period series tracking carry-over across the original day index.
    long_duration: bool = False
    # Phase 4.x — physical exclusivity of the charge/discharge legs. When True,
    # the optimiser adds a binary z[t] with ``ch[t] <= p_cap·z[t]`` and
    # ``dis[t] <= p_cap·(1−z[t])`` (or the extendable capacity variable's upper
    # bound), preventing simultaneous charge+discharge at round-trip η=1. Only
    # needed when efficiency_charge × efficiency_discharge == 1 or when the
    # user explicitly requires the physical constraint — costs one binary/
    # timestep/storage, so leave off for pure LP dispatch.
    no_simultaneous: bool = False
    ramp_cost: float = 0.0  # $/MW per ramp event (applied to |net_power[t] - net_power[t-1]|)
    # 18.P2 v3 — ramp continuity handoff (net discharge-charge reference for
    # the t=0 ramp cost; None = historical ramp-from-zero).
    ramp_t0_reference: Optional[float] = None
    # 18.P2 v4 — linear boundary net-flow prices on (discharge - charge) at
    # the first/last step (default 0.0 = no-op); see Link.flow_t0_cost.
    net_t0_cost: float = 0.0
    net_terminal_cost: float = 0.0
    # 18.P2 v4b — proximal boundary terms (all None = no-op):
    # soc_start_v_cost (ref, rate): +rate·|soc_start - ref| (convex; needs
    #   soc_initial_free) — the receiver-side anti-teleport charge.
    # soc_terminal_v_rebate (ref, rate): -rate·|soc[T-1] - ref| (one
    #   binary) — the donor-side mirror. On any true trajectory the two
    #   sides cancel exactly (s_in = s_out), so the bound stays valid for
    #   ANY rate ≥ 0; good refs/rates only tighten it.
    soc_start_v_cost: Optional[tuple] = None
    soc_terminal_v_rebate: Optional[tuple] = None
    net_terminal_v_rebate: Optional[tuple] = None
    # Storage formulation selector (PyPSA-style Store vs nexus default 3-var).
    # 'auto' (default): use 'store' (1 var e[t]) for lossless symmetric
    # storages without extras; otherwise use 'full' (3 vars charge/discharge/
    # soc). 'store' / 'full' force the model. The store model shrinks the
    # LP polyhedron — necessary for HiGHS to find integer-feasible vertices
    # at root in models with committable links (PyPSA's secret sauce).
    storage_model: str = "auto"
    # Phase 5 — investment planning depth (multi-stage vintage tracking).
    fixed_om_power: float = 0.0  # $/MW/year on active power cap
    fixed_om_energy: float = 0.0  # $/MWh/year on active energy cap
    retire_at_year: Optional[int] = None
    build_year: Optional[int] = None
    lifetime_years: int = 25
    build_lead_years: int = 0  # construction lead time — new-build from stage S active at year ≥ year(S) + lead
    # Phase 5.1 — retrofit / repower. New-build POWER capacity at stage S is
    # bounded by the host storage's retiring power capacity between S-1 and S
    # (pumped-hydro revamp, battery re-cell). Energy track is bounded
    # analogously by the host's retiring energy capacity.
    retrofit_of: Optional[str] = None
    # Internal
    _charge_vars: list = field(default_factory=list, repr=False)
    _discharge_vars: list = field(default_factory=list, repr=False)
    _soc_vars: list = field(default_factory=list, repr=False)
    _soc_start_var: object = field(default=None, repr=False)  # soc_initial_free
    # Phase 18.x — 1-var Store mode. Single energy state per timestep.
    # Populated only when ``effective_storage_model() == 'store'``; in
    # that mode ``_charge_vars`` / ``_discharge_vars`` stay empty and
    # ``_soc_vars`` aliases ``_e_vars`` so existing result-extraction
    # code (storage_soc) and downstream tests still work.
    _e_vars: list = field(default_factory=list, repr=False)
    _spill_vars: list = field(default_factory=list, repr=False)  # only when inflow is set
    _cap_power_var: object = field(default=None, repr=False)
    _cap_energy_var: object = field(default=None, repr=False)
    _soc_inter_vars: list = field(default_factory=list, repr=False)  # LDS Kotzur
    _nosim_vars: list = field(default_factory=list, repr=False)  # binary z[t] when no_simultaneous=True
    _ramp_up_vars: list = field(default_factory=list, repr=False)
    _ramp_down_vars: list = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return (f"Storage({self.name!r}, bus={self.bus.name!r}, "
                f"power={self.power_capacity}MW, energy={self.energy_capacity}MWh)")

    def effective_storage_model(self) -> str:
        """Resolve 'auto' -> 'store' or 'full' based on the storage's
        properties. A storage qualifies for 'store' (1-var e[t]) only when
        every aspect that requires separate charge/discharge variables is
        absent: round-trip eff must be ~1, charge and discharge capacities
        must be symmetric, and none of the 3-var-only features (inflow,
        spill, ramp_cost, no_simultaneous, asymmetric availability) may
        be in use."""
        if self.storage_model in ("store", "full"):
            return self.storage_model
        # 'auto' — decide.
        round_trip = float(self.efficiency_charge) * float(self.efficiency_discharge)
        if round_trip < 0.999:
            return "full"
        if self.no_simultaneous or self.ramp_cost > 0.0:
            return "full"
        if self.inflow is not None or self.spill_to is not None:
            return "full"
        if self.availability is not None:
            return "full"
        pump = self.pump_capacity if self.pump_capacity is not None else self.power_capacity
        turb = self.turbine_capacity if self.turbine_capacity is not None else self.power_capacity
        if abs(pump - turb) > 1e-9:
            return "full"
        if self.long_duration:
            return "full"
        if self.marginal_cost_charge != 0.0 or self.marginal_cost != 0.0:
            # Cost on charge vs discharge legs differs from a net-Δe cost.
            return "full"
        return "store"


@dataclass
class Load:
    """A demand for energy on a bus."""
    name: str
    bus: Bus
    amount: float | np.ndarray = 0.0  # MW (scalar or time-series)

    def __repr__(self) -> str:
        if isinstance(self.amount, np.ndarray):
            return f"Load({self.name!r}, bus={self.bus.name!r}, T={len(self.amount)})"
        return f"Load({self.name!r}, bus={self.bus.name!r}, amount={self.amount})"


@dataclass
class Link:
    """A connection that transports/converts energy between two buses."""
    name: str
    bus_from: Bus
    bus_to: Bus
    capacity: float  # MW
    efficiency: float = 1.0  # conversion efficiency
    marginal_cost: float = 0.0
    capital_cost: float = 0.0  # $/MW/year
    bidirectional: bool = False
    extendable: bool = False
    max_capacity: float = float("inf")
    min_capacity: float = 0.0  # lower bound for extendable (PyPSA p_nom_min / s_nom_min)
    loss: float = 0.0  # linear fractional loss on arrival (0 = lossless)
    # Phase 10 — quadratic transmission loss coefficient (per-MW² fraction).
    # Total loss = loss * |f| + loss_quadratic * f². Approximated by K
    # outer-tangent cuts on f² (PWL). Only honoured on ``model_type='transport'``
    # links; DC-OPF / PTDF lines stay lossless (PyPSA Line semantics).
    loss_quadratic: float = 0.0
    loss_pwl_breakpoints: int = 4
    _loss_vars: list = field(default_factory=list, repr=False)
    # Phase 4.4 — part-load efficiency curve. List of ``(load_fraction, efficiency)``
    # points (load_fraction ∈ (0, 1] of ``capacity``) describing a *concave*
    # input→output conversion curve (efficiency typically falls toward full
    # load for electrolysers / fuel cells). When set, the delivered output at
    # ``bus_to`` becomes a dedicated variable bounded by the concave upper
    # envelope of the segment supporting-lines instead of the flat
    # ``efficiency × flow``. LP-exact when the output is valued downstream
    # (the envelope binds from below). Points must be sorted ascending in
    # load_fraction; the implied output curve must be concave.
    efficiency_segments: Optional[list] = None
    _eta_out_vars: list = field(default_factory=list, repr=False)  # part-load delivered output
    bus_to_2: Optional[Bus] = None  # second output bus (multi-output link, e.g. CHP elec+heat)
    efficiency2: float = 0.0  # conversion efficiency to bus_to_2
    bus_to_3: Optional[Bus] = None  # third output bus
    efficiency3: float = 0.0
    # Unit commitment on links (mirrors Generator UC fields)
    committable: bool = False
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0
    min_up_time: int = 0
    min_down_time: int = 0
    ramp_up_limit: Optional[float] = None  # MW/timestep
    ramp_down_limit: Optional[float] = None
    ramp_cost: float = 0.0  # $/MW per ramp event (applied to |flow[t] - flow[t-1]|)
    # 18.P2 v3 — ramp continuity handoff: when set, the t=0 ramp cost is
    # priced against this flow level (the previous window's final flow in a
    # rolling/stitched solve) instead of against zero. None = historical
    # ramp-from-zero behaviour.
    ramp_t0_reference: Optional[float] = None
    # 18.P2 v4 — linear boundary-flow prices (default 0.0 = no-op): objective
    # gains flow_t0_cost·flow[0] + flow_terminal_cost·flow[T-1]. Used by the
    # temporal decomposition's LB blocks to under-approximate the relaxed
    # cross-boundary ramp cost (rc·|Δf| ≥ rc·s·Δf for any |s| ≤ 1).
    flow_t0_cost: float = 0.0
    flow_terminal_cost: float = 0.0
    # 18.P2 v4b — donor-side V-rebate (ref, rate): objective gains
    # -rate·|flow[T-1] - ref| (one binary; valid Lagrangian counterpart of a
    # downstream window's t=0 ramp charge against the same reference —
    # charge - rebate ≤ rate·|Δflow| by the triangle inequality).
    flow_terminal_v_rebate: Optional[tuple] = None
    _status_vars: list = field(default_factory=list, repr=False)
    _startup_vars: list = field(default_factory=list, repr=False)
    _shutdown_vars: list = field(default_factory=list, repr=False)
    co2_output_bus: Optional[Bus] = None  # if set, flow emits CO₂ onto this bus as physical carrier
    co2_output_factor: float = 0.0  # carrier-unit per MWh of input flow (e.g., tCO₂/MWh)
    # Network physics (Phase 3)
    reactance: float = 0.0  # per-unit reactance on system MVA base; >0 enables DC-OPF
    model_type: str = "transport"  # 'transport' | 'dc_opf' | 'ptdf' | 'switched'
    switchable: bool = False  # Link can be turned on/off per-timestep (MILP)
    # Linepack (Phase 4) — pipe inventory for gas / H₂ networks. When set,
    # the link grows a per-timestep inventory variable that decouples
    # injection at bus_from from withdrawal at bus_to. Inflow vs outflow can
    # differ within the inventory window, modeling pipe storage.
    linepack_capacity: float = 0.0  # MWh of storable carrier in the pipe (0 = no linepack)
    linepack_initial: float = 0.5  # fraction in [0,1] of linepack_capacity at t=0
    linepack_cyclic: bool = False  # if True, inv[T-1] = inv[-1] (start)
    # Phase 5 — investment planning depth (multi-stage vintage tracking).
    fixed_om: float = 0.0  # $/MW/year on active link cap
    retire_at_year: Optional[int] = None
    build_year: Optional[int] = None
    lifetime_years: int = 40  # transmission typical 40y
    build_lead_years: int = 0  # construction lead time (years)
    # Phase 5.1 — retrofit / repower. New-build capacity at stage S is bounded
    # by the host link's retiring capacity between S-1 and S (e.g. natural-gas
    # pipeline → hydrogen, AC line reconductoring).
    retrofit_of: Optional[str] = None
    integer_investment: bool = False  # cap built = unit_size × integer count (discrete tx expansion)
    unit_size: float = 1.0  # MW per discrete unit
    # Internal
    _formulation_override: "str | None" = field(default=None, repr=False)  # A/B reformulation pin
    _flow_vars: list = field(default_factory=list, repr=False)
    _flow_rev_vars: list = field(default_factory=list, repr=False)
    _flow_signed_vars: list = field(default_factory=list, repr=False)  # DC-OPF signed flow
    _flow_out_vars: list = field(default_factory=list, repr=False)  # linepack: withdrawal at bus_to
    _inv_vars: list = field(default_factory=list, repr=False)  # linepack: pipe inventory
    _switch_vars: list = field(default_factory=list, repr=False)  # transmission switching binaries
    _cap_var: object = field(default=None, repr=False)
    _ramp_up_vars: list = field(default_factory=list, repr=False)
    _ramp_down_vars: list = field(default_factory=list, repr=False)

    def effective_link_model(self) -> str:
        """Resolve which LP/MIP variable shape to emit for this link.

        Returns one of:
          * 'dc_opf' / 'ptdf'  — network-physics modes (always signed
            flow, handled by ``network.build_*``).
          * 'signed'           — PyPSA-style symmetric bidirectional
            transport: single ``f[t] ∈ [-cap, cap]`` per timestep,
            no fwd/rev mutex. Halves variable + constraint count for
            transmission lines.
          * 'fwd_rev'          — default: one non-negative flow var
            (and a non-negative reverse var when ``bidirectional``).

        Eligibility for 'signed' is conservative — anything that
        wants to apply a coefficient to the absolute flow magnitude
        (loss, marginal_cost, ramp_cost, multi-output efficiency2/3,
        CO2 output, UC, linepack) keeps the fwd/rev formulation so
        the LP stays linear.

        ``_formulation_override`` (set by ``optimise(link_formulation=...)``)
        forces a shape for A/B reformulation measurement: ``"fwd_rev"`` pins the
        naive baseline (disables the signed collapse); ``None``/``"auto"`` keeps
        the eligibility logic below. Physics modes (dc_opf/ptdf) ignore it.
        """
        if self.model_type in ("dc_opf", "ptdf"):
            return self.model_type
        if getattr(self, "_formulation_override", None) == "fwd_rev":
            return "fwd_rev"
        if not self.bidirectional:
            return "fwd_rev"
        # Cost-aware reformulation choice. For a FIXED-capacity bidi link the
        # signed form is a strict win: 1 var + 0 extra rows (bounds ±cap) vs
        # fwd+rev's 2 vars + 1 mutex row. But for an EXTENDABLE link the cap is
        # a variable, so signed needs a *two-sided* coupling (f ≤ cap_var AND
        # −f ≤ cap_var = 2 rows) where fwd+rev needs only one (fwd+rev ≤ cap_var):
        # signed then trades −1 var for +1 row and measures SLOWER. So only
        # collapse to signed when capacity is fixed. (Pure accounting — the
        # feasible region and optimum are identical either way.) An explicit
        # ``link_formulation="signed"`` override can still force it.
        if self.extendable and getattr(self, "_formulation_override", None) != "signed":
            return "fwd_rev"
        if self.efficiency != 1.0 or self.loss != 0.0 or self.loss_quadratic != 0.0:
            return "fwd_rev"
        if self.bus_to_2 is not None or self.bus_to_3 is not None:
            return "fwd_rev"
        if self.co2_output_bus is not None:
            return "fwd_rev"
        if self.committable or self.switchable:
            return "fwd_rev"
        if self.linepack_capacity > 0:
            return "fwd_rev"
        if self.marginal_cost != 0.0:
            return "fwd_rev"
        if self.ramp_cost != 0.0:
            return "fwd_rev"
        if self.ramp_up_limit is not None or self.ramp_down_limit is not None:
            return "fwd_rev"
        return "signed"

    def __repr__(self) -> str:
        return (f"Link({self.name!r}, {self.bus_from.name!r} → {self.bus_to.name!r}, "
                f"capacity={self.capacity}, η={self.efficiency})")


# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------

@dataclass
class OptimisationResult:
    """Container for optimisation results."""
    status: str
    total_cost: float
    solve_time: float
    # Dispatch results (populated after solve)
    generator_dispatch: dict[str, np.ndarray] = field(default_factory=dict)
    storage_charge: dict[str, np.ndarray] = field(default_factory=dict)
    storage_discharge: dict[str, np.ndarray] = field(default_factory=dict)
    storage_soc: dict[str, np.ndarray] = field(default_factory=dict)
    link_flow: dict[str, np.ndarray] = field(default_factory=dict)
    bus_shadow_prices: dict[str, np.ndarray] = field(default_factory=dict)
    # 18.P2 — duals of the SOC-recursion equalities (full-mode storages,
    # LP solves only): the marginal value of stored energy ($/MWh).
    storage_soc_duals: dict[str, np.ndarray] = field(default_factory=dict)
    # 18.P2 v3 — duals of soc_fixed pin rows, keyed (name, t): the marginal
    # cost of delivering the pinned SOC level (LP solves only).
    soc_fixed_duals: dict = field(default_factory=dict)
    # Unit commitment schedules — per-generator u[t] in [0, n_units] for
    # clustered UC, {0,1} for single-unit. Populated only when
    # committable=True. Used by Phase 11 ML warm-start: the GNN /
    # historical-neighbour predictor learns to map per-timestep features
    # → u[t], then passes the vector back through ``optimise(warm_start=)``.
    unit_status: dict[str, np.ndarray] = field(default_factory=dict)
    # Investment results
    capacity_additions: dict[str, float] = field(default_factory=dict)
    # Phase 8 — Benders subproblem support. Populated only when
    # ``optimise(benders_fix_caps=...)`` is used: dual of the
    # ``cap_var == fixed`` equality, i.e. the marginal operational cost
    # of one more MW of that capacity. The Benders coefficient β_j.
    cap_dual: dict[str, float] = field(default_factory=dict)
    # Raw nexus-opt result
    _raw: object = field(default=None, repr=False)
    # Phase 18.a — (bus_name, t) → balance-constraint ROW index, recorded
    # at build time. Lets mpc.PersistentDispatchSession update demand RHS
    # in-place on a PersistentHighs instance without rebuilding the model.
    _balance_row_idx: object = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (f"OptimisationResult(status={self.status!r}, "
                f"total_cost={self.total_cost:.2f}, solve_time={self.solve_time:.3f}s)")


# ---------------------------------------------------------------------------
# EnergySystem — the main class
# ---------------------------------------------------------------------------

class EnergySystem:
    """
    An energy system model that can be optimised.

    Usage:
        >>> sys = EnergySystem("test")
        >>> elec = sys.add_bus("elec", carrier="electricity")
        >>> sys.add_generator("pv", bus=elec, capacity=100, marginal_cost=0)
        >>> sys.add_load("demand", bus=elec, amount=80)
        >>> result = sys.optimise()
    """

    def __init__(self, name: str = "energy_system",
                 capex_mode: str = "total"):
        if capex_mode not in ("total", "incremental"):
            raise ValueError(
                f"capex_mode must be 'total' or 'incremental', got {capex_mode!r}")
        self.name = name
        self._capex_mode = capex_mode  # Phase 14: "total" (greenfield) vs "incremental" (PyPSA-style)
        self._buses: list[Bus] = []
        self._generators: list[Generator] = []
        self._storages: list[Storage] = []
        self._loads: list[Load] = []
        self._links: list[Link] = []
        self._carriers: dict[str, Carrier] = dict(CARRIERS)  # copy defaults
        self._timesteps: int = 1
        self._dt: float = 1.0  # hours per timestep
        self._emission_limit: float | None = None
        self._co2_price: float | None = None  # $/tCO2, applied to marginal cost of every emitting gen
        self._capacity_buckets: list[dict] = []  # [{"tech": str, "bus": Bus|None, "min": float, "max": float}]
        self._spinning_reserve: float | None = None  # fraction of load, single-product reserve headroom
        # Phase 2.3 — single-largest-unit contingency (N-1 generation) reserve.
        # When set, system spinnable headroom excluding any one contingency unit
        # must cover that unit's output at every timestep.
        self._contingency_reserve: dict | None = None  # {"generators": set[str] | None}
        self._reg_up_fraction: float | None = None  # fraction of load for regulation up
        self._reg_down_fraction: float | None = None  # fraction of load for regulation down
        self._n_minus_1_lines: list[str] = []  # contingency line names for preventive N-1 (DC-OPF only)
        # Phase 4.4 — shared-capacity converter locks. Each entry
        # {"links": [name,...], "mutex": bool} ties the listed links to one
        # shared converter rating (one electrolyser/fuel-cell stack used both
        # ways) and, when mutex, forbids simultaneous full-power operation.
        self._shared_caps: list[dict] = []
        # Phase 6 — policy library
        self._co2_rate_cap: float | None = None  # tCO2 / MWh delivered
        # Phase 14 — RHS storage-losses flag on the system rate cap (GenX CO2Cap=2).
        # When True: RHS = rate × (delivered_energy + storage_losses).
        self._co2_rate_cap_slosses: bool = True
        self._co2_zone_caps: list[dict] = []  # [{"bus": Bus, "limit": float, "is_rate": bool, "storage_losses_on_rhs": bool}]
        # Phase 23 — pooled multi-zone CO2 cap-group (GenX Cap_Zone). One
        # constraint over a SET of buses is *tighter* than independent per-bus
        # caps (it forbids inter-zone emission averaging). Default empty → no
        # behaviour change. ``loss_accounting`` ∈ {"net","dissipation"}: "net"
        # (default, bit-stable) keeps the legacy Σ(charge−discharge) RHS term;
        # "dissipation" uses the true round-trip loss (1/η_c−1)·charge+(1−η_d)·discharge.
        self._co2_cap_groups: list[dict] = []
        self._rps: dict | None = None  # {"fraction": float, "techs": set[str]}
        self._ces: dict | None = None  # {"fraction": float, "scores": dict[str, float]}
        self._itc: dict[str, float] = {}  # {tech: credit_fraction in [0,1]}
        self._ptc: dict[str, float] = {}  # {tech: $/MWh credit}
        self._hourly_matching: dict | None = None  # {"load": Load, "techs": set[str]}
        self._reserve_margin: dict | None = None  # {"margin": float, "firm_credit": {tech: [0,1]}}
        # Phase 2.x — fuel-supply limits. List of entries
        # {"name": str, "cap": float (MMBtu/annum), "coeffs": {gen_name: heat_rate}}.
        # Constraint: sum_{g, t} heat_rate[g] × p[g,t] × w[t] × dt ≤ cap.
        self._fuel_limits: list[dict] = []
        # Phase 7 — temporal aggregation
        self._snapshot_weights: np.ndarray | None = None  # weight per timestep; scales cost/emission/policy aggregations only
        self._chrono_mapping: np.ndarray | None = None  # original_day_idx -> representative_period_idx (Kotzur LDS)
        self._period_length: int | None = None  # timesteps per representative period
        self._snapshot_durations: np.ndarray | None = None  # Phase 7.3/7.4 variable-resolution per-snapshot hours

    # ---- Carrier management ----

    def add_carrier(self, name: str, unit: str = "MWh") -> Carrier:
        """Register a custom carrier."""
        c = Carrier(name, unit)
        self._carriers[name] = c
        return c

    def _resolve_carrier(self, carrier: str | Carrier) -> Carrier:
        if isinstance(carrier, Carrier):
            return carrier
        if carrier in self._carriers:
            return self._carriers[carrier]
        raise ValueError(
            f"Unknown carrier {carrier!r}. "
            f"Available: {list(self._carriers.keys())}. "
            f"Use add_carrier() to register custom carriers."
        )

    # ---- Bus ----

    def add_bus(self, name: str, carrier: str | Carrier = "electricity") -> Bus:
        """Add a bus (connection point) to the system."""
        c = self._resolve_carrier(carrier)
        bus = Bus(name=name, carrier=c, _id=len(self._buses))
        self._buses.append(bus)
        return bus

    # ---- Generator ----

    def add_generator(self, name: str, bus: Bus, capacity: float,
                      marginal_cost: float = 0.0, **kwargs) -> Generator:
        """Add a generator (energy source) to the system."""
        gen = Generator(name=name, bus=bus, capacity=capacity,
                        marginal_cost=marginal_cost, **kwargs)
        self._generators.append(gen)
        return gen

    # ---- Storage ----

    def add_storage(self, name: str, bus: Bus,
                    power_capacity: float, energy_capacity: float,
                    **kwargs) -> Storage:
        """Add a storage unit to the system."""
        sto = Storage(name=name, bus=bus,
                      power_capacity=power_capacity,
                      energy_capacity=energy_capacity, **kwargs)
        self._storages.append(sto)
        return sto

    # ---- Load ----

    def add_load(self, name: str, bus: Bus,
                 amount: float | np.ndarray = 0.0) -> Load:
        """Add a demand/load to the system."""
        load = Load(name=name, bus=bus, amount=amount)
        self._loads.append(load)
        return load

    # ---- Link ----

    def add_link(self, name: str, bus_from: Bus, bus_to: Bus,
                 capacity: float, efficiency: float = 1.0,
                 **kwargs) -> Link:
        """Add a link (transport/conversion) between two buses."""
        link = Link(name=name, bus_from=bus_from, bus_to=bus_to,
                    capacity=capacity, efficiency=efficiency, **kwargs)
        self._links.append(link)
        return link

    # ---- Time-series setup ----

    def set_timesteps(self, n: int, dt: float = 1.0):
        """Set the number of timesteps and duration per step (hours)."""
        self._timesteps = n
        self._dt = dt

    def set_snapshot_durations(self, durations):
        """Set per-snapshot wall-clock durations (hours) — variable resolution.

        Phase 7.3 / 7.4. When set, snapshot ``t`` spans ``durations[t]`` hours
        instead of the uniform ``dt``: storage SOC, power-rate caps, linepack,
        cost and emission energy integrals all use the per-snapshot length, so
        a coarse (merged) block correctly moves ``power × duration`` of energy.
        Length must equal the number of timesteps. Pass ``None`` to clear and
        return to a uniform clock. Typically configured by
        :func:`temporal.apply_adaptive_resolution`.
        """
        if durations is None:
            self._snapshot_durations = None
            return
        self._snapshot_durations = np.asarray(durations, dtype=float)

    def set_chronological_mapping(self, mapping, period_length: int):
        """Map each original period (e.g., day-of-year) to its representative.

        Required for long-duration storage (LDS) inter-period superposition
        (Kotzur 2018): the chronological order of original periods is what
        lets seasonal storages carry SOC across rep-day boundaries even
        though the LP only solves the rep periods themselves. Set
        automatically by ``apply_representative_days``.

        Args:
            mapping: 1-D array of length ``n_original_periods``; entry ``d``
                gives the representative period index that day ``d`` maps to.
            period_length: timesteps per representative period (24 for days).
        """
        self._chrono_mapping = np.asarray(mapping, dtype=np.int64)
        self._period_length = int(period_length)

    def set_snapshot_weights(self, weights):
        """Per-timestep weight applied to cost / emission / policy aggregations.

        Used by temporal aggregation (representative periods, TDR): a snapshot
        with weight w counts w times in the objective, emission caps, RPS / CES
        accounting, etc. Per-timestep physics (SOC evolution, bus balance,
        ramps, capacity) are NOT scaled — those bind on each snapshot
        independently. Pass ``None`` to clear.
        """
        if weights is None:
            self._snapshot_weights = None
            return
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError(f"snapshot weights must be 1-D, got shape {w.shape}")
        if (w < 0).any():
            raise ValueError("snapshot weights must be non-negative")
        self._snapshot_weights = w

    def set_emission_limit(self, limit: float):
        """Set a global CO2 emission cap (tCO2 over the entire horizon)."""
        self._emission_limit = limit

    def set_co2_price(self, price: float):
        """Set a global CO2 price ($/tCO2). Added to marginal cost of every emitting generator."""
        self._co2_price = price

    def set_capacity_bucket(self, tech: str, min_mw: float = 0.0,
                            max_mw: float = float("inf"),
                            bus: Bus | None = None):
        """Constrain total capacity (existing + extendable) of a technology bucket.

        If bus is None, the bucket aggregates across the system; otherwise it's
        per-bus. Mirrors PyPSA carrier-level and GenX MinCapReq/MaxCapReq.
        """
        self._capacity_buckets.append({"tech": tech, "bus": bus,
                                       "min": float(min_mw), "max": float(max_mw)})

    def set_spinning_reserve(self, fraction: float):
        """Require total generator headroom (capacity - dispatch) ≥ fraction * load at every timestep."""
        self._spinning_reserve = float(fraction)

    def set_contingency_reserve(self, generators: "list[str] | None" = None):
        """Single-largest-unit (N-1 generation) contingency reserve.

        Requires that, at every timestep, the system's *spinnable* headroom
        excluding any one contingency unit covers that unit's output — so the
        loss of any single unit can be replaced without shedding load. This is
        the standard exact linearisation used by GenX / SpineOpt / Sienna:
        for each contingency unit ``g`` and timestep ``t``,

            Σ_{h ≠ g} headroom_h[t]  ≥  p[g, t]

        where ``headroom_h`` is committed-but-unused capacity
        (``avail_cap_h · u_h − p_h`` for committable units, ``avail_cap_h − p_h``
        for always-on dispatchable units; VRE with ``carrier_factor`` provides
        none). Layers on top of :meth:`set_spinning_reserve`.

        Args:
            generators: names of units treated as contingencies (the set whose
                single failure must be survivable). ``None`` (default) means all
                dispatchable, non-VRE generators.
        """
        self._contingency_reserve = {
            "generators": set(generators) if generators is not None else None
        }

    def set_outage(self, generator: "str | Generator",
                   windows: "list[tuple[int, int]]",
                   availability: float = 0.0):
        """Planned-outage / maintenance window (time-varying availability).

        Convenience wrapper over ``Generator.carrier_factor``: derates the unit
        to ``availability`` (default 0.0 = full outage) over each
        ``(start, end)`` half-open timestep window, composing multiplicatively
        with any existing ``carrier_factor`` (so a VRE profile and a maintenance
        window stack). GenX ``Maintenance`` / SpineOpt / Sienna planned-outage
        analogue. Requires the horizon length to be set first
        (:meth:`set_timesteps` or a load profile).

        Args:
            generator: generator name or instance.
            windows: list of ``(start, end)`` half-open timestep ranges that the
                unit is on outage.
            availability: residual availability factor during the window
                (0.0 = full outage, 0.5 = 50 % derate, …).
        """
        if isinstance(generator, Generator):
            gen = generator
        else:
            gen = next((g for g in self._generators if g.name == generator), None)
            if gen is None:
                raise KeyError(f"set_outage: no generator named {generator!r}")
        T = self._timesteps
        if not T or T < 1:
            raise ValueError(
                "set_outage requires the horizon length to be set first "
                "(call set_timesteps(...) or add a load profile)."
            )
        cf = (np.ones(T, dtype=float) if gen.carrier_factor is None
              else np.asarray(gen.carrier_factor, dtype=float).copy())
        if cf.shape[0] != T:
            raise ValueError(
                f"set_outage({gen.name!r}): existing carrier_factor length "
                f"{cf.shape[0]} != horizon {T}."
            )
        for (start, end) in windows:
            s = max(0, int(start))
            e = min(T, int(end))
            if e > s:
                cf[s:e] *= float(availability)
        gen.carrier_factor = cf

    def set_regulation_reserve(self, up_fraction: float = 0.0,
                               down_fraction: float = 0.0):
        """System-wide regulation reserve requirement.

        Each contributing generator must have ``reg_up_max`` / ``reg_down_max``
        > 0. Reserve vars are capped by `reg_*_max * capacity`, co-bound with
        dispatch (``p + reg_up <= cap*u``; ``p - reg_down >= p_min*u``), and
        summed to ≥ fraction * load at every timestep.
        """
        if up_fraction > 0:
            self._reg_up_fraction = float(up_fraction)
        if down_fraction > 0:
            self._reg_down_fraction = float(down_fraction)

    # ---- Phase 6 policy setters ----

    def set_co2_rate_cap(self, rate: float, *,
                         storage_losses_on_rhs: bool = True):
        """Bound emissions per MWh of energy delivered (GenX CO2Cap=2/3).

        sum(emission_factor × p[g,t]) ≤ rate × (delivered_energy + storage_losses)

        storage_losses_on_rhs (default True) matches GenX CO2Cap=2 by adding
        sum(charge − discharge) to the RHS so round-trip losses don't eat into
        the emissions budget. Set False for the stricter GenX CO2Cap=3.
        """
        self._co2_rate_cap = float(rate)
        self._co2_rate_cap_slosses = bool(storage_losses_on_rhs)

    def set_co2_zone_cap(self, bus: "Bus", limit: float, *,
                         is_rate: bool = False,
                         storage_losses_on_rhs: bool = True):
        """Bound emissions from generators connected to ``bus``.

        Default (is_rate=False) is an absolute tCO2 budget. With is_rate=True
        the ``limit`` is tCO2/MWh and the zone constraint becomes
        sum(ef·p[z,t]·ω·dt) ≤ limit × (zone_load + storage_losses[z])
        (GenX CO2Cap=2 per-zone). storage_losses_on_rhs toggles the storage-loss
        RHS term.
        """
        self._co2_zone_caps.append({"bus": bus,
                                    "limit": float(limit),
                                    "is_rate": bool(is_rate),
                                    "storage_losses_on_rhs": bool(storage_losses_on_rhs)})

    def set_co2_cap_group(self, buses: "list[Bus]", limit: float, *,
                          is_rate: bool = True,
                          storage_losses_on_rhs: bool = True,
                          loss_accounting: str = "net"):
        """Pooled CO2 cap over a *group* of buses (GenX ``Cap_Zone``) — Phase 23.

        A single constraint summing emissions (and, for ``is_rate``, load +
        storage losses) across all ``buses`` is **tighter** than independent
        per-bus :meth:`set_co2_zone_cap` calls: it forbids inter-zone emission
        averaging, which is the dominant driver of the GenX ``rate_co2``
        parity gap (per-bus caps let a clean importer subsidise a dirty
        exporter). Use this when the GenX case defines one shared cap zone.

        Args:
            buses: buses pooled under one cap.
            limit: tCO2 budget (``is_rate=False``) or tCO2/MWh rate (default).
            storage_losses_on_rhs: add storage losses to the rate RHS (GenX CO2Cap=2).
            loss_accounting: ``"net"`` (legacy Σ(charge−discharge), bit-stable) or
                ``"dissipation"`` (true round-trip loss
                ``(1/η_c−1)·charge + (1−η_d)·discharge``, the physically-correct,
                strictly-positive dissipated energy).
        """
        if loss_accounting not in ("net", "dissipation"):
            raise ValueError("loss_accounting must be 'net' or 'dissipation'")
        self._co2_cap_groups.append({
            "buses": list(buses),
            "limit": float(limit),
            "is_rate": bool(is_rate),
            "storage_losses_on_rhs": bool(storage_losses_on_rhs),
            "loss_accounting": str(loss_accounting),
        })

    def set_rps(self, fraction: float, qualifying_techs: list[str],
                slack_penalty: "float | None" = None):
        """Require a minimum share of energy from qualifying tech tags.

        Phase 6.2 — when ``slack_penalty`` is given (\\$/MWh of shortfall), the
        target becomes a *soft* constraint: a non-negative slack absorbs any
        miss at that penalty price (GenX ``policies_slack``), so an infeasible
        portfolio reports a priced shortfall instead of a hard infeasibility.
        ``None`` (default) keeps the hard constraint — no behaviour change.
        """
        self._rps = {"fraction": float(fraction),
                     "techs": set(qualifying_techs),
                     "slack_penalty": (None if slack_penalty is None
                                       else float(slack_penalty))}

    def set_ces(self, fraction: float, scores: dict[str, float],
                slack_penalty: "float | None" = None):
        """Clean Energy Standard: weighted sum(score[tech] × dispatch) ≥ fraction × load.

        Phase 6.2 — ``slack_penalty`` (\\$/MWh of shortfall) makes the standard a
        soft constraint priced at that penalty (GenX ``policies_slack``); ``None``
        keeps it hard.
        """
        self._ces = {"fraction": float(fraction),
                     "scores": dict(scores),
                     "slack_penalty": (None if slack_penalty is None
                                       else float(slack_penalty))}

    def set_itc(self, credits: dict[str, float]):
        """Investment Tax Credit: per-tech capital_cost is multiplied by (1 − credit)."""
        self._itc = dict(credits)

    def set_ptc(self, credits: dict[str, float]):
        """Production Tax Credit: $/MWh subtracted from marginal cost on qualifying dispatch."""
        self._ptc = dict(credits)

    def set_fuel_supply_limit(self, name: str, max_fuel: float, *,
                              generators: dict[str, float]):
        """Cap the total fuel consumed by a set of generators.

        GenX ``Fuel_Supply`` / Calliope / oemof / SpineOpt all expose this
        as an annual bucket on generators sharing a fuel. Formulation:

            sum_{g in generators, t} heat_rate[g] × p[g,t] × w[t] × dt
                                   ≤ max_fuel

        ``max_fuel`` is in the same energy unit as the heat rates
        (commonly MMBtu/yr when ``heat_rate`` is MMBtu/MWh; use MWh_fuel/yr
        if you pass dimensionless efficiencies). nexus does not store a
        standalone ``heat_rate`` field on ``Generator`` (``marginal_cost``
        already folds heat_rate × fuel_price + var_OM) so the caller
        supplies the coefficient explicitly via ``generators``.
        """
        if not generators:
            raise ValueError(
                f"set_fuel_supply_limit({name!r}): at least one generator required"
            )
        self._fuel_limits.append({
            "name": str(name),
            "cap": float(max_fuel),
            "coeffs": {str(k): float(v) for k, v in generators.items()},
        })

    def set_hourly_matching(self, load_name: str, qualifying_techs: list[str]):
        """24/7 clean matching: qualifying dispatch ≥ this load demand at every timestep."""
        # Resolve the load at solve time (may be added after this call)
        self._hourly_matching = {"load_name": load_name,
                                  "techs": set(qualifying_techs)}

    def set_reserve_margin(self, margin: float,
                            firm_credit: dict[str, float],
                            peak_override: Optional[float] = None):
        """Capacity reserve margin: derated firm capacity ≥ (1 + margin) × peak load.

        ``peak_override`` substitutes a caller-supplied peak-load value
        (MW) for the auto-derived system peak. Used by the SAA chance-
        constrained solver to pin the constraint at the (1-α)-quantile
        peak across a scenario ensemble, rather than each subsystem's
        own max-load.
        """
        self._reserve_margin = {"margin": float(margin),
                                 "firm_credit": dict(firm_credit),
                                 "peak_override": (
                                     float(peak_override)
                                     if peak_override is not None else None)}

    def set_shared_capacity(self, link_names: "list[str]", *, mutex: bool = True):
        """Lock a set of links to one shared converter rating (Phase 4.4).

        Models an electrolyser / fuel-cell co-location where a single
        power-electronics stack is used in both directions: the listed links
        share one capacity, and (with ``mutex=True``) cannot run at full power
        simultaneously. For extendable links the shared rating is a single
        decision variable (their ``_cap_var``s are tied equal); for
        fixed-capacity links it is their common nameplate. The mutex adds, per
        timestep, ``Σ_i flow_i[t] ≤ shared_cap`` so combined throughput never
        exceeds the converter rating.

        Args:
            link_names: ≥2 link names sharing the converter.
            mutex: forbid simultaneous full-power operation (default True).
        """
        if len(link_names) < 2:
            raise ValueError("set_shared_capacity needs at least two links.")
        self._shared_caps.append({"links": list(link_names), "mutex": bool(mutex)})

    def set_n_minus_1(self, line_names: list[str]):
        """Enable preventive N-1 security against the listed contingency lines.

        Each named line must be a Link with ``model_type='dc_opf'`` (the
        network module needs phase-angle physics to redistribute flows
        when a line trips). Generation dispatch is identical in base and
        contingency states; for ramp-aware corrective N-1, layer reserves
        on top via ``set_spinning_reserve`` / ``set_regulation_reserve``.
        """
        self._n_minus_1_lines = list(line_names)

    # ---- Info ----

    @property
    def n_buses(self) -> int:
        return len(self._buses)

    @property
    def n_components(self) -> int:
        return len(self._generators) + len(self._storages) + len(self._links)

    @property
    def n_timesteps(self) -> int:
        return self._timesteps

    def summary(self) -> str:
        """Human-readable model summary."""
        lines = [
            f"EnergySystem: {self.name}",
            f"  Buses: {len(self._buses)}",
            f"  Generators: {len(self._generators)}",
            f"  Storages: {len(self._storages)}",
            f"  Loads: {len(self._loads)}",
            f"  Links: {len(self._links)}",
            f"  Timesteps: {self._timesteps} (dt={self._dt}h)",
        ]
        return "\n".join(lines)

    # ---- Detect timesteps from data ----

    def _infer_timesteps(self):
        """Infer number of timesteps from time-series data if not explicitly set."""
        if self._timesteps > 1:
            return  # already set
        for load in self._loads:
            if isinstance(load.amount, np.ndarray):
                self._timesteps = len(load.amount)
                return
        for gen in self._generators:
            if gen.carrier_factor is not None:
                self._timesteps = len(gen.carrier_factor)
                return

    # ---- Build & Solve ----

    class _RelaxedModel:
        """Proxy used by ``mip_strategy='auto'``/``'lp_first'`` to solve the
        LP relaxation: forwards ``binary()`` and ``integer()`` to plain
        continuous variables, leaves every other method untouched. Records
        which names were relaxed so the integrality check afterwards tests
        exactly the variables that are discrete in the true MIP (a name
        prefix cannot know that, e.g., a selectively-demoted ``lv_``/``lw_``
        var is already continuous).
        """
        __slots__ = ("_inner", "_bin_names", "_int_names")

        def __init__(self, inner, bin_names=None, int_names=None):
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_bin_names",
                               bin_names if bin_names is not None else [])
            object.__setattr__(self, "_int_names",
                               int_names if int_names is not None else [])

        def binary(self, name):
            self._bin_names.append(name)
            return self._inner.variable(name, lower=0.0, upper=1.0)

        def integer(self, name, **kw):
            self._int_names.append(name)
            return self._inner.variable(name, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def optimise(self, objective: str = "min_cost",
                 solver: str | None = None,
                 time_limit: float | None = None,
                 gap: float | None = None,
                 threads: int | None = None,
                 solver_method: str | None = None,
                 run_crossover: str | None = None,
                 parallel: str | None = None,
                 lp_backend: str | None = None,
                 scale_cleanup: bool = True,
                 simplex_scale_strategy: int | None = None,
                 eliminate_redundant: bool = True,
                 link_formulation: str = "auto",
                 ramp_cost_formulation: str = "split",
                 verbose: bool = False,
                 presolve: bool = True,
                 benders_fix_caps: dict[str, float] | None = None,
                 benders_skip_capex: bool = False,
                 uc_fix_schedule: "dict[str, np.ndarray] | None" = None,
                 warm_start=None,
                 model_hook=None,
                 basis=None,
                 mip_strategy: str = "auto",
                 _relax_integers: bool = False,
                 _ramp_cost_skip_t0: bool = False) -> OptimisationResult:
        """
        Build the optimisation model and solve.

        Args:
            objective: "min_cost" (default) or "min_emissions"
            solver: Force a specific solver ("highs", "osqp", etc.)
            time_limit: Maximum solve time in seconds
            gap: MIP relative gap tolerance
            verbose: Print solver output
            presolve: Run nexus-opt presolve
            benders_fix_caps: Phase 8 Benders hook. Name → fixed MW. Adds
                ``cap_var == value`` equalities so the subproblem treats the
                listed extendable capacities as parameters. Names follow
                ``capacity_additions`` conventions: ``gen.name``, ``link.name``,
                ``f"{sto.name}_power"`` / ``f"{sto.name}_energy"``. Duals on
                these equalities are returned in ``result.cap_dual`` and
                become Benders-cut coefficients β_j.
            benders_skip_capex: Skip all capex / fixed-O&M terms in the
                objective so the subproblem's ``total_cost`` is pure
                operational cost. Use in conjunction with ``benders_fix_caps``.
            mip_strategy: "auto" (default) — LP-first shortcut for small
                MIPs (≤ 5000 est. binaries), falling through to the MIP
                solver when the relaxation is fractional. "mip_only" —
                always go straight to the MIP solver. "lp_first" (opt-in,
                N_En_Phase 18 loophole-1) — force the LP-first path at any
                size and solve the relaxation on a vertex-producing backend
                (simplex unless the caller picked a vertex-safe one); exact
                by the LP-relaxation theorem: the LP optimum is returned
                only when every relaxed binary/integer is integral at the
                vertex, otherwise the standard MIP solve runs (the LP cost
                is then paid twice — opt-in for that reason).
                "fix_and_certify" (opt-in, loophole-2) — lp_first plus: on a
                fractional vertex, fix every u[t] that IS integral there,
                solve the residual MIP, and return it when its cost is
                within ``gap`` of the relaxation bound (a valid optimality
                certificate); falls back to the full MIP otherwise.

        Returns:
            OptimisationResult with dispatch, costs, and diagnostics.
        """
        # Reformulation control (default "auto" = behaviour-preserving). "fwd_rev"
        # pins the naive link shape (disables the signed-flow collapse) so the
        # signed reformulation's value can be A/B-measured; the optimum is
        # identical either way (signed and fwd+rev describe the same feasible
        # region). dc_opf/ptdf physics links ignore this.
        if link_formulation not in ("auto", "fwd_rev", "signed"):
            raise ValueError("link_formulation must be 'auto', 'fwd_rev', or 'signed'")
        # 18.t.2 (loophole-3 switch, opt-in): "signed" replaces each ramp-cost
        # up/down aux pair with ONE aux var r bounded by r >= ±Δ. Both forms
        # price exactly |Δ| at optimum (cost > 0), so the optimum is identical;
        # signed just has one column fewer per component per step.
        if ramp_cost_formulation not in ("split", "signed"):
            raise ValueError("ramp_cost_formulation must be 'split' or 'signed'")
        _lf_ov = None if link_formulation == "auto" else link_formulation
        for _lk in self._links:
            _lk._formulation_override = _lf_ov

        # Dynamic switcher for MIP solving. The "auto" strategy is
        # accuracy-preserving: it only short-circuits to an LP-only path
        # when that path's result is provably as good as what the
        # standard MIP solver would return.
        #
        # Pipeline (when ``mip_strategy=='auto'``):
        #   * Estimate whether the LP relaxation is cheap relative to the
        #     full MIP (heuristic: small problems where the LP costs ≪
        #     MIP). Skip the LP-first path for large problems to avoid
        #     paying the LP cost twice.
        #   * Solve the LP relaxation.
        #   * If every binary var in the LP solution is within ``INT_TOL``
        #     of {0,1}, the LP optimum IS the MIP optimum (LP-relaxation
        #     theorem). Return immediately — single LP solve, zero gap.
        #   * Otherwise fall through to the standard MIP solver below.
        #
        # NOTE: a round-then-fix-then-verify path was tried earlier but
        # rejected — for non-tight LPs (e.g. CINDER) the rounded primal
        # is far from the MIP optimum (691 % gap) and never passes the
        # ``rel ≤ gap`` check, so the work is wasted. Better to surrender
        # to HiGHS-MIP fast than burn cycles on a failed shortcut.
        _LP_FIRST_BINARY_THRESHOLD = 5000
        _has_integers = (
            any(g.committable for g in self._generators)
            or any(g.integer_investment for g in self._generators)
            or any(g.capex_segments is not None for g in self._generators)
            or any(l.committable for l in self._links)
            or any(s.no_simultaneous for s in self._storages)
        )
        # Cheap upper-bound estimate of binary count from data (avoids
        # paying a model build to decide whether to do the LP-first path).
        T_est = self._timesteps if self._timesteps is not None else 1
        _est_binaries = sum(
            T_est * (1 if g.committable else 0)
            + (1 if g.integer_investment else 0)
            + (len(g.capex_segments) - 1 if g.capex_segments else 0)
            for g in self._generators
        ) + sum(T_est * (1 if l.committable else 0) for l in self._links) \
          + sum(T_est * (1 if s.no_simultaneous else 0) for s in self._storages)

        # "lp_first" (N_En_Phase 18 loophole-1 switch, opt-in): force the
        # LP-first path regardless of problem size. Unlike "auto" it also
        # pins the relaxation onto a vertex-producing backend — an interior
        # point is fractional even when a vertex optimum is integral, so a
        # non-vertex relaxation would spuriously fail the check below.
        # "fix_and_certify" (loophole-2 switch, opt-in): like lp_first, but
        # when the vertex relaxation is fractional, fix every committable
        # unit's u[t] that IS integral at the vertex (NaN = leave free) and
        # solve the much smaller residual MIP. The result is certified
        # against the relaxation bound: restriction cost ≥ optimum ≥ LP
        # bound, so (cost - lp_bound)/cost ≤ gap proves gap-optimality. On
        # certificate failure it falls back to the standard full MIP.
        _fix_certify = (mip_strategy == "fix_and_certify")
        _lp_first_forced = (mip_strategy == "lp_first") or _fix_certify
        if ((_lp_first_forced
             or (mip_strategy == "auto"
                 and _est_binaries <= _LP_FIRST_BINARY_THRESHOLD))
                and _has_integers
                and not _relax_integers and uc_fix_schedule is None):
            INT_TOL = 1.0e-4
            if lp_backend in ("simplex", "ipm", "pdlp"):
                _relax_backend = lp_backend  # caller chose a vertex-safe one
            elif _lp_first_forced:
                _relax_backend = "simplex"
            else:
                _relax_backend = None  # "auto" path: unchanged behaviour
            lp_result = self.optimise(
                objective=objective, solver=solver,
                time_limit=time_limit, gap=None,
                threads=threads, solver_method=solver_method,
                run_crossover=run_crossover, parallel=parallel,
                lp_backend=_relax_backend,
                scale_cleanup=scale_cleanup,
                simplex_scale_strategy=simplex_scale_strategy,
                eliminate_redundant=eliminate_redundant,
                ramp_cost_formulation=ramp_cost_formulation,
                verbose=verbose, presolve=presolve,
                benders_fix_caps=benders_fix_caps,
                benders_skip_capex=benders_skip_capex,
                model_hook=model_hook, basis=basis,
                mip_strategy="mip_only",
                _relax_integers=True,
                _ramp_cost_skip_t0=_ramp_cost_skip_t0,
            )
            if lp_result.status == "optimal":
                names = lp_result._raw.var_names_list
                primals = list(lp_result._raw.primals)
                idx = {nm: i for i, nm in enumerate(names)}
                lp_is_integer_tight = True
                for nm in getattr(self, "_relax_bin_names", ()):
                    i = idx.get(nm)
                    v = primals[i] if i is not None else None
                    if v is None or min(abs(v), abs(v - 1.0)) > INT_TOL:
                        lp_is_integer_tight = False
                        break
                if lp_is_integer_tight:
                    for nm in getattr(self, "_relax_int_names", ()):
                        i = idx.get(nm)
                        v = primals[i] if i is not None else None
                        if v is None or abs(v - round(v)) > INT_TOL:
                            lp_is_integer_tight = False
                            break
                if lp_is_integer_tight:
                    return lp_result
                if _fix_certify:
                    INT_TOL_FIX = 1.0e-6
                    sched: dict[str, np.ndarray] = {}
                    n_fixed = 0
                    n_free = 0
                    raw = lp_result._raw
                    for comp in list(self._generators) + list(self._links):
                        if not (comp.committable and comp._status_vars):
                            continue
                        vals = np.array(
                            [raw.value(u) for u in comp._status_vars])
                        arr = np.full(vals.shape[0], np.nan)
                        is_int = np.minimum(
                            np.abs(vals), np.abs(vals - 1.0)) <= INT_TOL_FIX
                        arr[is_int] = np.round(vals[is_int])
                        n_fixed += int(is_int.sum())
                        n_free += int((~is_int).sum())
                        sched[comp.name] = arr
                    if verbose:
                        print(f"[nexus] fix_and_certify: {n_fixed} u[t] "
                              f"fixed at the vertex, {n_free} left to B&B")
                    fixed_res = self.optimise(
                        objective=objective, solver=solver,
                        time_limit=time_limit, gap=gap,
                        threads=threads, solver_method=solver_method,
                        run_crossover=run_crossover, parallel=parallel,
                        lp_backend=lp_backend,
                        scale_cleanup=scale_cleanup,
                        simplex_scale_strategy=simplex_scale_strategy,
                        eliminate_redundant=eliminate_redundant,
                        ramp_cost_formulation=ramp_cost_formulation,
                        verbose=verbose, presolve=presolve,
                        benders_fix_caps=benders_fix_caps,
                        benders_skip_capex=benders_skip_capex,
                        model_hook=model_hook,
                        uc_fix_schedule=sched or None,
                        mip_strategy="mip_only",
                    )
                    _tol = gap if gap is not None else 1e-4
                    if fixed_res.status == "optimal":
                        rel = ((fixed_res.total_cost - lp_result.total_cost)
                               / max(1.0, abs(fixed_res.total_cost)))
                        if rel <= _tol + 1e-12:
                            return fixed_res  # certified within gap
                        if verbose:
                            print(f"[nexus] fix_and_certify: certificate "
                                  f"failed ({rel:.4%} > {_tol:.4%}) — "
                                  "falling back to the full MIP")
            # Fall through to standard MIP solve below.

        self._infer_timesteps()
        T = self._timesteps
        dt = self._dt

        # Phase 7: snapshot weights scale aggregations (cost / emission /
        # policy / energy totals) but NOT per-step physics. w[t] defaults to
        # 1.0 when no weights are set.
        if self._snapshot_weights is None:
            w = np.ones(T, dtype=float)
        else:
            if len(self._snapshot_weights) != T:
                raise ValueError(
                    f"snapshot_weights length {len(self._snapshot_weights)} "
                    f"does not match timesteps {T}"
                )
            w = self._snapshot_weights

        # Phase 7.3/7.4 — variable-resolution clock. When per-snapshot
        # durations are set, ``dt`` becomes time-varying: ``dts[t]`` is the
        # wall-clock length (hours) of snapshot ``t``. Storage SOC, power
        # caps, linepack, cost and emission energy integrals all use the
        # per-snapshot value so a merged low-resolution block moves the right
        # amount of energy. Defaults to the scalar ``dt`` everywhere (zero
        # behaviour change when unset).
        if self._snapshot_durations is not None:
            if len(self._snapshot_durations) != T:
                raise ValueError(
                    f"snapshot_durations length {len(self._snapshot_durations)} "
                    f"does not match timesteps {T}")
            dts = np.asarray(self._snapshot_durations, dtype=float)
        else:
            dts = np.full(T, float(dt))

        model = nx.Model(self.name)
        if _relax_integers:
            # Fresh registries per solve; the outer LP-first caller reads
            # them back to test integrality of exactly the relaxed vars.
            self._relax_bin_names = []
            self._relax_int_names = []
            model = self._RelaxedModel(
                model, self._relax_bin_names, self._relax_int_names)

        # ---- Create variables ----

        # Generator dispatch: p[g, t] >= 0
        for gen in self._generators:
            # Reset per-solve state so repeated optimise() calls don't stack
            # stale segment vars from a prior model.
            gen._capex_seg_vars = []
            gen._capex_seg_slopes = []
            # Compute upper bound on dispatch
            # If extendable, the binding cap constraint comes later via ext_cap
            if gen.extendable:
                upper_base = gen.max_capacity if gen.max_capacity != float("inf") else 1e12
            else:
                upper_base = gen.capacity
                # Clustered UC lumps N identical units together; dispatch
                # can reach ``capacity * n_units`` when all units are on.
                if gen.clustered and gen.n_units > 1:
                    upper_base = gen.capacity * gen.n_units

            # Lower bound: 0 when committable (p_min is enforced by the
            # ``p >= p_min * u`` constraint so the unit can be off). Non-
            # committable gens see p_min as a hard lower bound.
            p_lower = 0.0 if gen.committable else gen.p_min
            if T == 1:
                cf = 1.0
                if gen.carrier_factor is not None:
                    cf = float(gen.carrier_factor[0])
                v = model.variable(f"p_{gen.name}",
                                   lower=p_lower, upper=upper_base * cf)
                gen._p_vars = [v]
            else:
                gen._p_vars = []
                for t in range(T):
                    cf = 1.0
                    if gen.carrier_factor is not None:
                        cf = float(gen.carrier_factor[t])
                    v = model.variable(f"p_{gen.name}_{t}",
                                       lower=p_lower, upper=upper_base * cf)
                    gen._p_vars.append(v)

            # Extendable capacity
            if gen.extendable:
                cap_lower = gen.min_capacity
                cap_upper = gen.max_capacity if gen.max_capacity != float("inf") else 1e12
                gen._cap_var = model.variable(
                    f"cap_{gen.name}", lower=cap_lower, upper=cap_upper)

                # Phase 5 — integer unit investment. Force cap_var onto the
                # integer lattice `unit_size × k` (k integer).
                if gen.integer_investment:
                    if gen.unit_size <= 0:
                        raise ValueError(
                            f"Generator {gen.name!r}: unit_size must be > 0 "
                            f"when integer_investment=True.")
                    u_lower = int(np.ceil(cap_lower / gen.unit_size))
                    u_upper = int(np.floor(cap_upper / gen.unit_size))
                    units_var = model.integer(
                        f"units_{gen.name}", lower=u_lower, upper=u_upper)
                    model.add(
                        gen._cap_var == gen.unit_size * units_var,
                        name=f"intunits_{gen.name}")

                # Phase 5 — PWL CapEx (economies of scale). K segments with
                # strictly decreasing $/MW slopes; binaries enforce that
                # segment i must be fully used before segment i+1 can open.
                if gen.capex_segments is not None:
                    segs = gen.capex_segments
                    if len(segs) < 2:
                        raise ValueError(
                            f"Generator {gen.name!r}: capex_segments must "
                            f"have ≥ 2 breakpoints.")
                    mw_bps = [float(s[0]) for s in segs]
                    slopes = [float(s[1]) for s in segs]
                    for i in range(1, len(mw_bps)):
                        if mw_bps[i] <= mw_bps[i - 1]:
                            raise ValueError(
                                f"Generator {gen.name!r}: capex_segments "
                                f"breakpoints must be strictly ascending.")
                    for i in range(1, len(slopes)):
                        if slopes[i] >= slopes[i - 1]:
                            raise ValueError(
                                f"Generator {gen.name!r}: capex_segments "
                                f"slopes must be strictly decreasing.")
                    widths = [mw_bps[0]] + [
                        mw_bps[i] - mw_bps[i - 1] for i in range(1, len(mw_bps))
                    ]
                    seg_vars = [
                        model.variable(
                            f"capseg_{gen.name}_{i}", lower=0, upper=widths[i])
                        for i in range(len(widths))
                    ]
                    # y[i] = 1 means segment i+1 (and later) can open. Filling
                    # order: seg[0] must be full before seg[1] opens, etc.
                    y_vars = [
                        model.binary(f"capsegy_{gen.name}_{i}")
                        for i in range(len(widths) - 1)
                    ]
                    for i in range(len(widths) - 1):
                        # seg[i] >= width[i] * y[i]  (must fill i if i+1 opens)
                        model.add(
                            seg_vars[i] >= widths[i] * y_vars[i],
                            name=f"capseg_lo_{gen.name}_{i}")
                        # seg[i+1] <= width[i+1] * y[i]  (can't open until i full)
                        model.add(
                            seg_vars[i + 1] <= widths[i + 1] * y_vars[i],
                            name=f"capseg_hi_{gen.name}_{i}")
                    sum_expr = seg_vars[0]
                    for i in range(1, len(seg_vars)):
                        sum_expr = sum_expr + seg_vars[i]
                    model.add(
                        gen._cap_var == sum_expr,
                        name=f"capseg_sum_{gen.name}")
                    # Stash for objective assembly
                    gen._capex_seg_vars = seg_vars
                    gen._capex_seg_slopes = slopes

            # Unit commitment variables (u, v, w) and constraints are built
            # in a dedicated pass below via `build_three_bin_uc` — it needs
            # the p-dispatch vars (just created above) to wire min/max
            # constraints against u[t].

        # Storage variables: 3-var (charge, discharge, soc) by default, or
        # 1-var (e[t] only) when the storage qualifies for the PyPSA-style
        # Store model — see ``Storage.effective_storage_model()``.
        for sto in self._storages:
            sto._charge_vars = []
            sto._discharge_vars = []
            sto._soc_vars = []
            sto._e_vars = []
            sto._spill_vars = []
            sto._ramp_up_vars = []
            sto._ramp_down_vars = []
            sto._soc_start_var = None
            if sto.soc_initial_free:
                if sto.cyclic or sto.extendable or sto.long_duration:
                    raise ValueError(
                        f"soc_initial_free on storage {sto.name!r} requires "
                        "non-cyclic, non-extendable, non-LDS storage")
                _start_lo = sto.soc_min * sto.energy_capacity
                _start_hi = sto.soc_max * sto.energy_capacity
                if sto.soc_initial_free_min is not None:
                    _start_lo = max(_start_lo, float(sto.soc_initial_free_min))
                if sto.soc_initial_free_max is not None:
                    _start_hi = min(_start_hi, float(sto.soc_initial_free_max))
                sto._soc_start_var = model.variable(
                    f"soc_start_{sto.name}",
                    lower=_start_lo, upper=max(_start_hi, _start_lo))

            if sto.extendable:
                p_upper = sto.max_power_capacity if sto.max_power_capacity != float("inf") else 1e12
                e_upper_abs = sto.max_energy_capacity if sto.max_energy_capacity != float("inf") else 1e12
                soc_upper = sto.soc_max * e_upper_abs
            else:
                p_upper = sto.power_capacity
                soc_upper = sto.soc_max * sto.energy_capacity

            # Phase 4 — asymmetric pump/turbine caps (PSH) and availability (V2G).
            # Both default to power_capacity / 1.0 so legacy storages are unchanged.
            ch_cap = sto.pump_capacity if sto.pump_capacity is not None else p_upper
            dis_cap = sto.turbine_capacity if sto.turbine_capacity is not None else p_upper

            sto_mode = sto.effective_storage_model()

            if sto_mode == "store":
                if sto.extendable:
                    e_upper_abs = sto.max_energy_capacity if sto.max_energy_capacity != float("inf") else 1e12
                    for t in range(T):
                        e_var = model.variable(f"e_{sto.name}_{t}",
                                                lower=0,
                                                upper=sto.soc_max * e_upper_abs)
                        sto._e_vars.append(e_var)
                else:
                    soc_lower = sto.soc_min * sto.energy_capacity
                    for t in range(T):
                        e_var = model.variable(f"e_{sto.name}_{t}",
                                                lower=soc_lower,
                                                upper=soc_upper)
                        sto._e_vars.append(e_var)
                sto._soc_vars = sto._e_vars
                if not sto.extendable:
                    continue
                # Extendable store-mode: create cap vars and add SOC/power
                # constraints that reference them (can't use static bounds).
                pc_lower = sto.min_power_capacity
                pc_upper = sto.max_power_capacity if sto.max_power_capacity != float("inf") else 1e12
                ec_lower = sto.min_energy_capacity
                ec_upper = sto.max_energy_capacity if sto.max_energy_capacity != float("inf") else 1e12
                sto._cap_power_var = model.variable(
                    f"cap_power_{sto.name}", lower=pc_lower, upper=pc_upper)
                sto._cap_energy_var = model.variable(
                    f"cap_energy_{sto.name}", lower=ec_lower, upper=ec_upper)
                if sto.min_hours is not None and sto.max_hours is not None:
                    model.add(
                        sto._cap_energy_var >= sto._cap_power_var * sto.min_hours,
                        name=f"min_hours_{sto.name}")
                    model.add(
                        sto._cap_energy_var <= sto._cap_power_var * sto.max_hours,
                        name=f"max_hours_{sto.name}")
                elif sto.max_hours is not None:
                    model.add(
                        sto._cap_energy_var == sto._cap_power_var * sto.max_hours,
                        name=f"max_hours_{sto.name}")
                elif sto.min_hours is not None:
                    model.add(
                        sto._cap_energy_var >= sto._cap_power_var * sto.min_hours,
                        name=f"min_hours_{sto.name}")
                for t in range(T):
                    model.add(sto._e_vars[t] >= sto.soc_min * sto._cap_energy_var,
                              name=f"e_soc_lo_{sto.name}_{t}")
                    model.add(sto._e_vars[t] <= sto.soc_max * sto._cap_energy_var,
                              name=f"e_soc_hi_{sto.name}_{t}")
                continue

            for t in range(T):
                a_t = float(sto.availability[t]) if sto.availability is not None else 1.0
                if sto.extendable:
                    # Extendable storages keep the loose `p_upper` bound; the
                    # binding cap comes from the cap_var constraint below.
                    # Asymmetric caps + availability with extendable are not
                    # supported in this phase.
                    ch_up_t = p_upper
                    dis_up_t = p_upper
                else:
                    ch_up_t = ch_cap * a_t
                    dis_up_t = dis_cap * a_t
                ch = model.variable(f"ch_{sto.name}_{t}",
                                    lower=0, upper=ch_up_t)
                dis = model.variable(f"dis_{sto.name}_{t}",
                                     lower=0, upper=dis_up_t)
                soc_lower = sto.soc_min * (sto.max_energy_capacity if sto.extendable and sto.max_energy_capacity != float("inf") else sto.energy_capacity)
                if sto.extendable:
                    soc_lower = 0  # will be constrained by cap var
                # Phase 7 LDS: intra-period SOC is a *delta* off the
                # inter-period baseline, so it must be allowed to swing
                # negative. Realised SOC = soc_inter[d] + soc_intra[t]
                # is bounded into [soc_min·E, soc_max·E] separately
                # at every original timestep below.
                if sto.long_duration and self._chrono_mapping is not None:
                    e_ref = sto.max_energy_capacity if sto.extendable and sto.max_energy_capacity != float("inf") else sto.energy_capacity
                    soc_lower = -float(e_ref)
                    soc_upper = float(e_ref)
                soc = model.variable(f"soc_{sto.name}_{t}",
                                     lower=soc_lower,
                                     upper=soc_upper)
                sto._charge_vars.append(ch)
                sto._discharge_vars.append(dis)
                sto._soc_vars.append(soc)

            if sto.extendable:
                pc_lower = sto.min_power_capacity
                pc_upper = sto.max_power_capacity if sto.max_power_capacity != float("inf") else 1e12
                ec_lower = sto.min_energy_capacity
                ec_upper = sto.max_energy_capacity if sto.max_energy_capacity != float("inf") else 1e12
                sto._cap_power_var = model.variable(
                    f"cap_power_{sto.name}", lower=pc_lower, upper=pc_upper)
                sto._cap_energy_var = model.variable(
                    f"cap_energy_{sto.name}", lower=ec_lower, upper=ec_upper)
                # Duration window. If only max_hours set → PyPSA-style equality.
                # If both min_hours and max_hours set → GenX-style window.
                if sto.min_hours is not None and sto.max_hours is not None:
                    model.add(
                        sto._cap_energy_var >= sto._cap_power_var * sto.min_hours,
                        name=f"min_hours_{sto.name}"
                    )
                    model.add(
                        sto._cap_energy_var <= sto._cap_power_var * sto.max_hours,
                        name=f"max_hours_{sto.name}"
                    )
                elif sto.max_hours is not None:
                    model.add(
                        sto._cap_energy_var == sto._cap_power_var * sto.max_hours,
                        name=f"max_hours_{sto.name}"
                    )
                elif sto.min_hours is not None:
                    model.add(
                        sto._cap_energy_var >= sto._cap_power_var * sto.min_hours,
                        name=f"min_hours_{sto.name}"
                    )

        # Phase 4 — hydro spill variables. Create one spill var per timestep
        # for any storage with a natural inflow series. Spill is non-negative
        # and unbounded above; the SOC equation removes it from the upstream
        # reservoir, and (if ``spill_to`` is set) adds it to the downstream
        # reservoir's SOC. Without spill, the LP would be infeasible whenever
        # inflow + soc[t-1] > energy_capacity.
        for sto in self._storages:
            if sto.inflow is not None:
                # Tight upper bound: max spill per step is inflow + capacity
                # drained from the reservoir. Avoids the 1e12 big-M that ruins
                # HiGHS numerical conditioning (Bound range warning).
                inflow_max = float(np.max(np.abs(sto.inflow))) if sto.inflow is not None else 0.0
                e_cap = sto.energy_capacity if sto.energy_capacity != float("inf") else 1e6
                spill_upper = max(inflow_max + e_cap, 1.0) * 10.0
                for t in range(T):
                    sv = model.variable(
                        f"spill_{sto.name}_{t}", lower=0, upper=spill_upper)
                    sto._spill_vars.append(sv)

        # Link flow variables. Symmetric bidirectional transport links
        # (lossless, no UC, no multi-output, no per-step ramp/cost
        # decoration) use a single signed variable ``f[t] ∈ [-cap, cap]``
        # — PyPSA Line semantics. All other links keep the fwd/rev pair.
        # See ``Link.effective_link_model()`` for the eligibility logic.
        for link in self._links:
            link._flow_vars = []
            link._flow_rev_vars = []
            link._flow_signed_vars = []
            link._loss_vars = []
            link._flow_out_vars = []
            link._inv_vars = []
            link._eta_out_vars = []
            link._cap_var = None
            link._ramp_up_vars = []
            link._ramp_down_vars = []
            if link.extendable:
                flow_upper = link.max_capacity if link.max_capacity != float("inf") else 1e12
            else:
                flow_upper = link.capacity

            link_mode = link.effective_link_model()
            if link_mode == "signed":
                # 1-var signed flow. Halves Line var count vs fwd+rev.
                # When extendable, the cap is bound via ``|f| <= cap_var``
                # in the extendable-caps block below.
                for t in range(T):
                    fs = model.variable(f"flow_{link.name}_{t}",
                                        lower=-flow_upper, upper=flow_upper)
                    link._flow_signed_vars.append(fs)
            else:
                for t in range(T):
                    f = model.variable(f"flow_{link.name}_{t}",
                                       lower=0, upper=flow_upper)
                    link._flow_vars.append(f)
                if link.bidirectional:
                    link._flow_rev_vars = []
                    for t in range(T):
                        f = model.variable(f"flow_rev_{link.name}_{t}",
                                           lower=0, upper=flow_upper)
                        link._flow_rev_vars.append(f)
                    # Mutex: fwd + rev <= capacity (PyPSA Line semantics: |p| <= s_nom).
                    # Without this, the LP can "use" both directions simultaneously,
                    # effectively doubling the line capacity for free.
                    if not link.extendable:
                        for t in range(T):
                            model.add(
                                link._flow_vars[t] + link._flow_rev_vars[t] <= link.capacity,
                                name=f"bidi_cap_{link.name}_{t}"
                            )
                    # When extendable, the mutex is added in the extendable-caps block below
                    # against _cap_var (which gets created once).

            # Phase 4.4 — part-load efficiency curve (concave conversion).
            # Delivered output at bus_to becomes a dedicated variable bounded
            # by the concave upper envelope of the segment supporting-lines,
            # rather than the flat ``efficiency × flow``. Fixed-capacity links
            # only (breakpoints scale with nameplate; extendable would be
            # bilinear). Output is valued downstream so the envelope binds.
            if link.efficiency_segments:
                if link.extendable:
                    raise ValueError(
                        f"Link {link.name!r}: efficiency_segments requires a "
                        f"fixed capacity (extendable would be bilinear).")
                if link._flow_signed_vars or link.bidirectional:
                    raise ValueError(
                        f"Link {link.name!r}: efficiency_segments requires a "
                        f"unidirectional fwd flow link (not signed/bidirectional).")
                cap = float(link.capacity)
                pts = sorted(((float(lf), float(eff)) for lf, eff in
                              link.efficiency_segments), key=lambda x: x[0])
                # (input_MW, output_MW) breakpoints; prepend origin for the
                # first segment so the envelope passes through (0, 0).
                bps = [(0.0, 0.0)]
                for lf, eff in pts:
                    bps.append((lf * cap, lf * cap * eff))
                # Supporting lines through each adjacent breakpoint pair.
                lines = []  # (slope, intercept)
                for (x0, y0), (x1, y1) in zip(bps[:-1], bps[1:]):
                    if x1 <= x0:
                        continue
                    a = (y1 - y0) / (x1 - x0)
                    b = y0 - a * x0
                    lines.append((a, b))
                out_upper = bps[-1][1]
                for t in range(T):
                    ov = model.variable(
                        f"etaout_{link.name}_{t}", lower=0.0, upper=out_upper)
                    link._eta_out_vars.append(ov)
                    for j, (a, b) in enumerate(lines):
                        model.add(ov <= a * link._flow_vars[t] + b,
                                  name=f"etaout_env_{link.name}_{t}_{j}")

            # Phase 10 — PWL quadratic transmission loss.
            # Allocate one loss[t] var per snapshot, lower-bounded by K
            # tangent cuts on loss_quadratic * f². The variable enters bus
            # balance below as a subtraction from the arrival side; the LP
            # therefore minimises it implicitly because every unit of loss
            # has to be made up by paid-for generation upstream.
            link._loss_vars = []
            if link.loss_quadratic > 0.0:
                if link.model_type in ("dc_opf", "ptdf"):
                    raise ValueError(
                        f"Link {link.name!r}: loss_quadratic is not supported "
                        f"on model_type={link.model_type!r}; PWL losses apply "
                        f"only to transport links.")
                if link.bidirectional:
                    raise ValueError(
                        f"Link {link.name!r}: loss_quadratic on a "
                        f"bidirectional link is not supported in Phase 10 "
                        f"first-pass — split into two unidirectional links.")
                lq = float(link.loss_quadratic)
                K = max(2, int(link.loss_pwl_breakpoints))
                # Breakpoints at evenly spaced f values across [0, capacity].
                cap_for_pwl = float(link.max_capacity if link.extendable else link.capacity)
                if not (cap_for_pwl > 0 and cap_for_pwl < float("inf")):
                    raise ValueError(
                        f"Link {link.name!r}: PWL losses require finite "
                        f"capacity (got {cap_for_pwl}).")
                breakpoints = [cap_for_pwl * k / (K - 1) for k in range(K)]
                loss_upper = lq * cap_for_pwl * cap_for_pwl
                for t in range(T):
                    lv = model.variable(
                        f"loss_pwl_{link.name}_{t}",
                        lower=0.0, upper=loss_upper)
                    link._loss_vars.append(lv)
                    # tangent at f_k: loss ≥ lq * (2 f_k f − f_k²)
                    for k, fk in enumerate(breakpoints):
                        model.add(
                            lv - 2.0 * lq * fk * link._flow_vars[t]
                            >= -lq * fk * fk,
                            name=f"loss_pwl_cut_{link.name}_{t}_{k}",
                        )

            # Linepack (Phase 4): pipe inventory + separate withdrawal var.
            # Decouples injection at bus_from (_flow_vars) from delivery at
            # bus_to (_flow_out_vars). inv[t] tracks pipe storage and is
            # bounded by [0, linepack_capacity]. Bidirectional + linepack
            # is not supported in this phase.
            if link.linepack_capacity > 0:
                if link.bidirectional:
                    raise ValueError(
                        f"Link {link.name!r}: linepack_capacity is not "
                        f"supported on bidirectional links yet."
                    )
                for t in range(T):
                    fo = model.variable(f"flow_out_{link.name}_{t}",
                                        lower=0, upper=flow_upper)
                    iv = model.variable(f"inv_{link.name}_{t}",
                                        lower=0, upper=link.linepack_capacity)
                    link._flow_out_vars.append(fo)
                    link._inv_vars.append(iv)

        # Network physics (Phase 3): phase angles + DC-OPF / PTDF flows.
        # These helpers create signed-flow vars on Link._flow_signed_vars
        # and clear _flow_vars / _flow_rev_vars on the lines they own, so
        # the bus-balance loop below picks up the signed flow exactly once.
        _network.build_theta_vars(model, self, T)
        _network.build_dc_opf_constraints(model, self, T)
        _network.build_ptdf_constraints(model, self, T)

        # ---- Constraints ----

        # Bus balance: generation + imports = demand + exports + storage_charge.
        # Track balance constraint indices as we go so we can look up duals by
        # constraint position (no need for nexus-opt's full sensitivity() pass,
        # which is O(C*V) due to per-constraint hashmap rebuilds and catastrophic
        # on large models).
        balance_idx: dict[tuple[str, int], int] = {}
        # 18.P2 — SOC-recursion constraint indices (full-mode storages) so
        # the marginal value of stored energy ("water value") is recoverable
        # from the LP duals without a sensitivity pass.
        soc_idx: dict[tuple[str, int], int] = {}
        # 18.P2 v3 — soc_fixed pin-row indices: the dual of a pin equality is
        # the marginal cost of delivering that SOC level — the principled
        # Lagrangian boundary price when the pin sits on a guide trajectory.
        soc_fixed_idx: dict[tuple[str, int], int] = {}
        n_before_balance = model.num_constraints
        for t in range(T):
            for bus in self._buses:
                expr = None

                # Generators producing on this bus
                for gen in self._generators:
                    if gen.bus is bus:
                        term = gen._p_vars[t]
                        expr = term if expr is None else expr + term

                # CO₂ byproduct from generators onto this bus
                for gen in self._generators:
                    if (gen.co2_output_bus is bus
                            and gen.co2_output_factor != 0):
                        term = gen._p_vars[t] * gen.co2_output_factor
                        expr = term if expr is None else expr + term

                # CO₂ byproduct from links onto this bus
                for link in self._links:
                    if (link.co2_output_bus is bus
                            and link.co2_output_factor != 0
                            and link._flow_vars):
                        term = link._flow_vars[t] * link.co2_output_factor
                        expr = term if expr is None else expr + term

                # Storage net injection (production) on this bus.
                # Store-mode storages contribute a single signed term
                # ``(e[t-1]*(1-sd*dt) - e[t]) / dt`` (positive when
                # discharging) — the charge half is handled by SKIPPING
                # the subtraction below. Full-mode storages keep the
                # existing two-term pattern (discharge here, charge
                # subtracted later).
                inv_dt = 1.0 / dt
                for sto in self._storages:
                    if sto.bus is bus:
                        if sto._e_vars:
                            if t == 0:
                                if sto.cyclic and T > 1:
                                    e_prev = sto._e_vars[T - 1]
                                elif sto.extendable and sto._cap_energy_var is not None:
                                    e_prev = sto.soc_initial * sto._cap_energy_var
                                elif sto._soc_start_var is not None:
                                    e_prev = sto._soc_start_var
                                else:
                                    e_prev = sto.soc_initial * sto.energy_capacity
                            else:
                                e_prev = sto._e_vars[t - 1]
                            sd_factor = 1.0 - sto.self_discharge * dt
                            # ``(e_prev*sd_factor - e[t]) / dt`` — linear
                            # in e variables (and constant-bias for the
                            # non-cyclic t=0 case where ``e_prev`` is a
                            # float).
                            term = (e_prev * sd_factor - sto._e_vars[t]) * inv_dt
                        else:
                            term = sto._discharge_vars[t]
                        expr = term if expr is None else expr + term

                # Links arriving at this bus (imports).
                # Loss applied linearly on arrival: delivered = flow * efficiency * (1 - loss).
                # DC-OPF / PTDF lines use the signed flow var; their _flow_vars
                # have been cleared by the network builders, so the transport
                # branches below are skipped automatically.
                for link in self._links:
                    link_arrive = link.efficiency * (1.0 - link.loss)
                    if link._flow_signed_vars:
                        # Signed-flow convention: positive f leaves bus_from,
                        # arrives at bus_to. Covers DC-OPF / PTDF physics
                        # AND symmetric bidirectional transport links
                        # (lossless ⇒ no efficiency / loss applied here).
                        if link.bus_to is bus:
                            term = link._flow_signed_vars[t]
                            expr = term if expr is None else expr + term
                        if link.bus_from is bus:
                            term = -link._flow_signed_vars[t]
                            expr = term if expr is None else expr + term
                        continue
                    if link.bus_to is bus and link._flow_vars:
                        # Linepack: arrival is the withdrawal var, not the
                        # injection var. Without this swap the pipe could
                        # "magic" gas across the link without depleting
                        # inventory.
                        if link._eta_out_vars:
                            # Phase 4.4 — part-load curve: delivered output is
                            # the concave-envelope variable; ``efficiency`` is
                            # already folded into it, so only the linear loss
                            # multiplies here.
                            term = link._eta_out_vars[t] * (1.0 - link.loss)
                        elif link._inv_vars:
                            term = link._flow_out_vars[t] * link_arrive
                        else:
                            term = link._flow_vars[t] * link_arrive
                        # Phase 10 — subtract PWL quadratic loss on arrival.
                        if link._loss_vars:
                            term = term - link._loss_vars[t]
                        expr = term if expr is None else expr + term
                    if link.bidirectional and link.bus_from is bus and link._flow_rev_vars:
                        term = link._flow_rev_vars[t] * link_arrive
                        expr = term if expr is None else expr + term
                    # Multi-output: second and third output buses
                    if link.bus_to_2 is bus and link.efficiency2 != 0 and link._flow_vars:
                        term = link._flow_vars[t] * link.efficiency2 * (1.0 - link.loss)
                        expr = term if expr is None else expr + term
                    if link.bus_to_3 is bus and link.efficiency3 != 0 and link._flow_vars:
                        term = link._flow_vars[t] * link.efficiency3 * (1.0 - link.loss)
                        expr = term if expr is None else expr + term

                # Subtract: demand
                for load in self._loads:
                    if load.bus is bus:
                        d = load.amount
                        if isinstance(d, (np.ndarray, list, tuple)):
                            d = float(d[t])
                        if expr is None:
                            expr = -d
                        else:
                            expr = expr - d

                # Subtract: storage charging. Store-mode storages already
                # have their signed net contribution above, so skip them
                # here to avoid double-counting.
                for sto in self._storages:
                    if sto.bus is bus and not sto._e_vars:
                        expr = expr - sto._charge_vars[t]

                # Subtract: link exports from this bus. Signed-flow
                # links (DC-OPF / PTDF / symmetric bidi transport) were
                # already added with sign in the imports block above —
                # skip them here to avoid double-counting.
                for link in self._links:
                    if link._flow_signed_vars:
                        continue
                    if link.bus_from is bus and link._flow_vars:
                        expr = expr - link._flow_vars[t]
                    if link.bidirectional and link.bus_to is bus and link._flow_rev_vars:
                        expr = expr - link._flow_rev_vars[t]

                if expr is not None:
                    if isinstance(expr, (int, float)):
                        # Pure-constant balance: no variables on this bus
                        # at this timestep. Either trivially satisfied
                        # (expr ≈ 0) or unsatisfiable. In the latter case
                        # add an obviously infeasible constraint so the
                        # solver reports infeasibility cleanly instead of
                        # crashing on `model.add(False)`.
                        if abs(float(expr)) > 1e-9:
                            unbal = model.variable(
                                f"unbal_{bus.name}_{t}", lower=0, upper=0)
                            model.add(unbal >= 1.0,
                                      name=f"unbal_{bus.name}_{t}")
                        continue
                    balance_idx[(bus.name, t)] = model.num_constraints
                    model.add(expr == 0, name=f"balance_{bus.name}_{t}")

        # Transmission switching (per-line per-t binary z; KVL relaxed when z=0).
        _network.build_transmission_switching(model, self, T)

        # Preventive N-1: replica DC-OPF state per contingency line.
        _network.build_n_minus_1_constraints(model, self, T)

        # Linepack inventory evolution (Phase 4): inv[t] = inv[t-1] +
        # (flow_in[t] - flow_out[t]) * dt. Initial state from
        # ``linepack_initial`` * capacity unless cyclic, in which case
        # inv[0] closes the loop with inv[T-1].
        for link in self._links:
            if not link._inv_vars:
                continue
            for t in range(T):
                if t == 0:
                    if link.linepack_cyclic and T > 1:
                        inv_prev = link._inv_vars[T - 1]
                    else:
                        inv_prev = link.linepack_initial * link.linepack_capacity
                else:
                    inv_prev = link._inv_vars[t - 1]
                model.add(
                    link._inv_vars[t] == inv_prev
                    + link._flow_vars[t] * dts[t]
                    - link._flow_out_vars[t] * dts[t],
                    name=f"linepack_{link.name}_{t}"
                )

        # Storage SOC evolution. Phase 4 adds two new terms:
        #   + inflow[t]·dt − spill[t]·dt    (natural hydro inflow / overflow)
        #   + Σ upstream.spill[t]·dt        (cascade — water arriving from above)
        # Phase 7 LDS: long-duration storages with chronological mapping
        # use Kotzur 2018 inter-period superposition — intra-period SOC
        # acts as a delta off a per-original-day baseline ``soc_inter[d]``.
        chrono_active = (self._chrono_mapping is not None
                         and self._period_length is not None
                         and self._period_length > 0
                         and (T % self._period_length) == 0)
        n_orig_days = (len(self._chrono_mapping) if chrono_active else 0)

        for sto in self._storages:
            # Store-mode storages have no separate charge/discharge/soc
            # variables — the per-timestep power cap is the only extra
            # constraint they need beyond the variable bounds, since the
            # energy balance is implicit in the bus contribution (built
            # above) and the SOC bounds are direct variable bounds.
            if sto._e_vars:
                _ext_store = sto.extendable and sto._cap_power_var is not None
                for t in range(T):
                    if t == 0:
                        if sto.cyclic and T > 1:
                            e_prev = sto._e_vars[T - 1]
                        elif _ext_store:
                            e_prev = sto.soc_initial * sto._cap_energy_var
                        elif sto._soc_start_var is not None:
                            e_prev = sto._soc_start_var
                        else:
                            e_prev = sto.soc_initial * sto.energy_capacity
                    else:
                        e_prev = sto._e_vars[t - 1]
                    sd_factor = 1.0 - sto.self_discharge * dts[t]
                    if _ext_store:
                        ch_cap_expr = sto._cap_power_var * dts[t]
                        dis_cap_expr = sto._cap_power_var * dts[t]
                    else:
                        ch_cap_expr = sto.power_capacity * dts[t]
                        dis_cap_expr = sto.power_capacity * dts[t]
                    model.add(
                        sto._e_vars[t] - e_prev * sd_factor <= ch_cap_expr,
                        name=f"e_ch_{sto.name}_{t}",
                    )
                    model.add(
                        e_prev * sd_factor - sto._e_vars[t] <= dis_cap_expr,
                        name=f"e_dis_{sto.name}_{t}",
                    )
                # Optional SOC pin (timestep-level), keyed by t-index.
                if sto.soc_fixed:
                    for t_pin, soc_val in sto.soc_fixed.items():
                        if 0 <= int(t_pin) < T:
                            soc_fixed_idx[(sto.name, int(t_pin))] = \
                                model.num_constraints
                            model.add(
                                sto._e_vars[int(t_pin)] == float(soc_val),
                                name=f"soc_fixed_{sto.name}_{int(t_pin)}",
                            )
                if sto.soc_terminal_min is not None:
                    model.add(
                        sto._e_vars[T - 1] >= float(sto.soc_terminal_min),
                        name=f"soc_terminal_min_{sto.name}")
                if sto.soc_terminal_max is not None:
                    model.add(
                        sto._e_vars[T - 1] <= float(sto.soc_terminal_max),
                        name=f"soc_terminal_max_{sto.name}")
                is_store_lds = sto.long_duration and (self._chrono is not None and len(self._chrono) > 0)
                # "free" cyclic level: continuity (e[0] uses e[T-1] above) is
                # enough; skip the absolute pin so the optimiser picks the level
                # (PyPSA convention). "fixed" keeps the historical anchor.
                if sto.cyclic and T > 1 and not is_store_lds and sto.cyclic_level != "free":
                    if _ext_store:
                        model.add(
                            sto._e_vars[0] == sto.soc_initial * sto._cap_energy_var,
                            name=f"soc_cyclic_init_{sto.name}"
                        )
                    else:
                        model.add(
                            sto._e_vars[0] == sto.soc_initial * sto.energy_capacity,
                            name=f"soc_cyclic_init_{sto.name}"
                        )
                continue

            # Pre-compute upstream spill sources once per storage.
            upstream_spillers = [u for u in self._storages
                                 if u is not sto and u.spill_to is sto and u._spill_vars]

            is_lds = sto.long_duration and chrono_active
            # Phase 14 — short-duration storage + TDR: cyclic PER rep period
            # (GenX Model 1 convention). Without this, soc flows freely across
            # non-contiguous rep-period boundaries, giving batteries free
            # seasonal arbitrage and driving massive overbuild.
            per_period_cyclic = (chrono_active
                                 and not is_lds
                                 and self._period_length
                                 and self._period_length > 0
                                 and sto.cyclic
                                 and sto.inflow is None)
            pl_local = self._period_length if per_period_cyclic else None

            for t in range(T):
                if per_period_cyclic and (t % pl_local) == 0:
                    # Start of a rep period: soc_prev = end of same period
                    # (closes cycle within-period).
                    soc_prev = sto._soc_vars[t + pl_local - 1]
                elif t == 0:
                    if is_lds:
                        # LDS intra block always starts each rep period at 0
                        # (delta semantics). The inter-period block carries
                        # the absolute level.
                        soc_prev = 0.0
                    elif sto.cyclic and T > 1:
                        soc_prev = sto._soc_vars[T - 1]
                    elif sto.extendable:
                        soc_prev = 0.0
                    elif sto._soc_start_var is not None:
                        soc_prev = sto._soc_start_var
                    else:
                        soc_prev = sto.soc_initial * sto.energy_capacity
                else:
                    if is_lds and self._period_length and (t % self._period_length) == 0:
                        # First step of a new representative period — reset
                        # intra delta to zero.
                        soc_prev = 0.0
                    else:
                        soc_prev = sto._soc_vars[t - 1]

                discharge_coeff = dts[t] * (1.0 / sto.efficiency_discharge)
                rhs = (soc_prev
                       + sto.efficiency_charge * sto._charge_vars[t] * dts[t]
                       - sto._discharge_vars[t] * discharge_coeff
                       - sto.self_discharge * soc_prev * dts[t])
                if sto.inflow is not None:
                    rhs = rhs + float(sto.inflow[t]) * dts[t] - sto._spill_vars[t] * dts[t]
                for upstream in upstream_spillers:
                    rhs = rhs + upstream._spill_vars[t] * dts[t]
                soc_idx[(sto.name, t)] = model.num_constraints
                model.add(sto._soc_vars[t] == rhs, name=f"soc_{sto.name}_{t}")

            # Phase 14 — timestep SOC pin (PyPSA state_of_charge_set).
            if sto.soc_fixed:
                for t_pin, soc_val in sto.soc_fixed.items():
                    if 0 <= int(t_pin) < T:
                        soc_fixed_idx[(sto.name, int(t_pin))] = \
                            model.num_constraints
                        model.add(
                            sto._soc_vars[int(t_pin)] == float(soc_val),
                            name=f"soc_fixed_{sto.name}_{int(t_pin)}",
                        )
            if sto.soc_terminal_min is not None:
                model.add(
                    sto._soc_vars[T - 1] >= float(sto.soc_terminal_min),
                    name=f"soc_terminal_min_{sto.name}")
            if sto.soc_terminal_max is not None:
                model.add(
                    sto._soc_vars[T - 1] <= float(sto.soc_terminal_max),
                    name=f"soc_terminal_max_{sto.name}")
            if sto.cyclic and T > 1 and not is_lds and sto.cyclic_level != "free":
                model.add(
                    sto._soc_vars[0] == sto.soc_initial * sto.energy_capacity,
                    name=f"soc_cyclic_init_{sto.name}"
                )

            # ---- LDS Kotzur inter-period block ----
            if is_lds:
                pl = self._period_length
                e_ref = (sto.max_energy_capacity
                         if sto.extendable and sto.max_energy_capacity != float("inf")
                         else sto.energy_capacity)
                soc_lo_abs = sto.soc_min * e_ref
                soc_hi_abs = sto.soc_max * e_ref
                # soc_inter[d] = absolute SOC at the start of original day d.
                # Bounds use the loose abs envelope; the per-hour realised
                # bounds below are tighter and binding.
                inter_vars = []
                for d in range(n_orig_days + 1):
                    v = model.variable(
                        f"soc_inter_{sto.name}_{d}",
                        lower=-float(e_ref), upper=float(e_ref))
                    inter_vars.append(v)
                sto._soc_inter_vars = inter_vars

                # Initial baseline.
                if sto.extendable:
                    model.add(inter_vars[0] == 0.0,
                              name=f"sociinit_{sto.name}")
                else:
                    model.add(inter_vars[0] == sto.soc_initial * sto.energy_capacity,
                              name=f"sociinit_{sto.name}")

                # Phase 16.5 — generalised Kotzur recursion under fractional
                # representative-period mapping. With a one-hot mapping the
                # inter-period delta for original day d is exactly the
                # end-of-period intra-SOC of its single rep period. Under a
                # fractional mapping_matrix M (row d = distribution over reps),
                # the expected daily storage swing is the weighted combination
                # ``Σ_p M[d,p]·soc_intra[end_of_p]`` — the natural stochastic
                # generalisation of the integer recursion (degenerate one-hot M
                # recovers it bit-for-bit).
                rep_obj = getattr(self, "_rep_periods", None)
                frac_M = None
                if rep_obj is not None and getattr(rep_obj, "mapping_matrix", None) is not None:
                    M = np.asarray(rep_obj.mapping_matrix, dtype=float)
                    if M.shape[0] == n_orig_days and not _mat_is_one_hot(M):
                        frac_M = M

                def _delta_for_day(d):
                    if frac_M is not None:
                        acc = None
                        for p in range(frac_M.shape[1]):
                            wgt = float(frac_M[d, p])
                            if wgt == 0.0:
                                continue
                            term = wgt * sto._soc_vars[(p + 1) * pl - 1]
                            acc = term if acc is None else acc + term
                        return acc
                    p = int(self._chrono_mapping[d])
                    return sto._soc_vars[(p + 1) * pl - 1]

                def _intra_for_day(d, h):
                    if frac_M is not None:
                        acc = None
                        for p in range(frac_M.shape[1]):
                            wgt = float(frac_M[d, p])
                            if wgt == 0.0:
                                continue
                            term = wgt * sto._soc_vars[p * pl + h]
                            acc = term if acc is None else acc + term
                        return acc
                    p = int(self._chrono_mapping[d])
                    return sto._soc_vars[p * pl + h]

                # Chronological recursion: soc_inter[d+1] = soc_inter[d] + delta_d.
                for d in range(n_orig_days):
                    delta = _delta_for_day(d)
                    model.add(inter_vars[d + 1] == inter_vars[d] + delta,
                              name=f"sociter_{sto.name}_{d}")

                # Cyclic year (default for LDS): soc_inter[N] == soc_inter[0].
                if sto.cyclic:
                    model.add(inter_vars[n_orig_days] == inter_vars[0],
                              name=f"sociccl_{sto.name}")

                # Realised SOC bounds at every original hour.
                # soc_real(d, h) = soc_inter[d] + soc_intra[p*pl + h] ∈ [lo, hi].
                if sto.extendable and sto._cap_energy_var is not None:
                    # Fixed-cap bounds replaced by cap_var-tied bounds below.
                    pass
                for d in range(n_orig_days):
                    for h in range(pl):
                        intra = _intra_for_day(d, h)
                        if sto.extendable and sto._cap_energy_var is not None:
                            model.add(inter_vars[d] + intra >= 0.0,
                                      name=f"socreal_lo_{sto.name}_{d}_{h}")
                            model.add(inter_vars[d] + intra <= sto._cap_energy_var * sto.soc_max,
                                      name=f"socreal_hi_{sto.name}_{d}_{h}")
                        else:
                            model.add(inter_vars[d] + intra >= soc_lo_abs,
                                      name=f"socreal_lo_{sto.name}_{d}_{h}")
                            model.add(inter_vars[d] + intra <= soc_hi_abs,
                                      name=f"socreal_hi_{sto.name}_{d}_{h}")

        # Ramp constraints for generators
        if T > 1:
            for gen in self._generators:
                # Absolute MW/timestep ramp
                if gen.ramp_up is not None:
                    for t in range(1, T):
                        model.add(
                            gen._p_vars[t] - gen._p_vars[t - 1] <= gen.ramp_up,
                            name=f"ramp_up_{gen.name}_{t}"
                        )
                if gen.ramp_down is not None:
                    for t in range(1, T):
                        model.add(
                            gen._p_vars[t - 1] - gen._p_vars[t] <= gen.ramp_down,
                            name=f"ramp_down_{gen.name}_{t}"
                        )
                # Phase 14 — fractional ramp (frac × cap). For extendable
                # gens the bound is linear in the cap_var; for fixed-cap
                # gens it collapses to a constant.
                if gen.ramp_up_frac is not None:
                    rhs_ref = (gen._cap_var if (gen.extendable and gen._cap_var is not None)
                               else gen.capacity)
                    for t in range(1, T):
                        model.add(
                            gen._p_vars[t] - gen._p_vars[t - 1]
                            <= gen.ramp_up_frac * rhs_ref,
                            name=f"ramp_up_frac_{gen.name}_{t}"
                        )
                if gen.ramp_down_frac is not None:
                    rhs_ref = (gen._cap_var if (gen.extendable and gen._cap_var is not None)
                               else gen.capacity)
                    for t in range(1, T):
                        model.add(
                            gen._p_vars[t - 1] - gen._p_vars[t]
                            <= gen.ramp_down_frac * rhs_ref,
                            name=f"ramp_down_frac_{gen.name}_{t}"
                        )

        # Unit commitment: Morales-España 3-bin + Rajan-Takriti min-up/down,
        # via the thermal-components helper. Clustered UC (continuous u/v/w
        # in [0, n_units]) is selected per-gen by ``clustered=True``.
        if T > 1:
            for gen in self._generators:
                if not gen.committable:
                    continue

                # Phase 14.x — clustered + extendable uses per-unit cap
                # (unit_size × cf[t]) and the commit-built constraint
                # `unit_size × u[t] ≤ cap_var` inside build_three_bin_uc
                # to linearise what used to be a bilinear product.
                ext_clustered_linearised = (
                    gen.extendable
                    and gen._cap_var is not None
                    and gen.clustered
                    and gen.unit_size > 0
                )

                def _cap_per_t(gen=gen, ext_lin=ext_clustered_linearised):
                    def fn(t):
                        if ext_lin:
                            base = gen.unit_size
                        else:
                            base = gen.capacity
                        if gen.carrier_factor is not None:
                            return base * float(gen.carrier_factor[t])
                        return base
                    return fn

                build_three_bin_uc(
                    model, gen, T, _cap_per_t(),
                    cap_var=gen._cap_var if ext_clustered_linearised else None,
                )

                if gen.must_run:
                    add_must_run(model, gen, T)

        # Must-run for non-committable generators (pin dispatch to
        # available capacity). Committable must-run was handled above.
        for gen in self._generators:
            if gen.must_run and not gen.committable:
                add_must_run(model, gen, T)

        # Unit commitment on links: u (binary), with v/w created only
        # when needed by startup/shutdown costs or Rajan-Takriti min-up/down.
        # This eliminates ~50% of UC variables and constraints for links
        # that have only one of startup/shutdown cost (the common case).
        if T > 1:
            for link in self._links:
                if not link.committable:
                    continue
                link._status_vars = []
                link._startup_vars = []
                link._shutdown_vars = []
                for t in range(T):
                    u = model.binary(f"lu_{link.name}_{t}")
                    link._status_vars.append(u)
                TU = int(link.min_up_time) if link.min_up_time else 0
                TD = int(link.min_down_time) if link.min_down_time else 0
                need_v = link.startup_cost > 0.0 or TU > 1
                need_w = link.shutdown_cost > 0.0 or TD > 1
                if need_v:
                    for t in range(T):
                        v = model.variable(f"lv_{link.name}_{t}", lower=0.0, upper=1.0)
                        link._startup_vars.append(v)
                    for t in range(1, T):
                        model.add(
                            link._startup_vars[t]
                            >= link._status_vars[t] - link._status_vars[t - 1],
                            name=f"luc_trans_up_{link.name}_{t}")
                if need_w:
                    for t in range(T):
                        ww = model.variable(f"lw_{link.name}_{t}", lower=0.0, upper=1.0)
                        link._shutdown_vars.append(ww)
                    for t in range(1, T):
                        model.add(
                            link._shutdown_vars[t]
                            >= link._status_vars[t - 1] - link._status_vars[t],
                            name=f"luc_trans_dn_{link.name}_{t}")

                cap_ref = link.capacity
                for t in range(T):
                    model.add(link._flow_vars[t] <= cap_ref * link._status_vars[t],
                              name=f"luc_max_{link.name}_{t}")
                if TU > 1:
                    for t in range(TU - 1, T):
                        acc = None
                        for s in range(t - TU + 1, t + 1):
                            term = link._startup_vars[s]
                            acc = term if acc is None else acc + term
                        model.add(acc <= link._status_vars[t],
                                  name=f"lmin_up_{link.name}_{t}")
                if TD > 1:
                    for t in range(TD - 1, T):
                        acc = None
                        for s in range(t - TD + 1, t + 1):
                            term = link._shutdown_vars[s]
                            acc = term if acc is None else acc + term
                        model.add(acc <= 1.0 - link._status_vars[t],
                                  name=f"lmin_down_{link.name}_{t}")
                if link.ramp_up_limit is not None:
                    for t in range(1, T):
                        model.add(
                            link._flow_vars[t] - link._flow_vars[t - 1] <= link.ramp_up_limit,
                            name=f"lramp_up_{link.name}_{t}")
                if link.ramp_down_limit is not None:
                    for t in range(1, T):
                        model.add(
                            link._flow_vars[t - 1] - link._flow_vars[t] <= link.ramp_down_limit,
                            name=f"lramp_down_{link.name}_{t}")

        # Ramp cost variables and constraints for links. In "signed" mode
        # (18.t.2) the up/down pair collapses into one r >= |Δ| variable —
        # objective handling below detects the empty _ramp_down_vars list.
        # ``_ramp_cost_skip_t0`` (18.P2 LB blocks): drop the t=0 rows so the
        # first step's ramp-from-zero charge vanishes (aux vars stay, priced
        # at their 0 floor) — a pure relaxation used for valid block bounds.
        _rc_signed = (ramp_cost_formulation == "signed")
        for link in self._links:
            if link.ramp_cost > 0.0 and link._flow_vars:
                for t in range(T):
                    if _rc_signed:
                        r = model.variable(f"lrc_{link.name}_{t}", lower=0.0)
                        link._ramp_up_vars.append(r)
                        if t == 0:
                            if not _ramp_cost_skip_t0:
                                _ref = link.ramp_t0_reference or 0.0
                                model.add(r - link._flow_vars[t] >= -_ref,
                                          name=f"lrc_up_{link.name}_0")
                                model.add(r + link._flow_vars[t] >= _ref,
                                          name=f"lrc_down_{link.name}_0")
                        else:
                            model.add(r >= link._flow_vars[t] - link._flow_vars[t - 1],
                                      name=f"lrc_up_{link.name}_{t}")
                            model.add(r >= link._flow_vars[t - 1] - link._flow_vars[t],
                                      name=f"lrc_down_{link.name}_{t}")
                        continue
                    ru = model.variable(f"lrc_up_{link.name}_{t}", lower=0.0)
                    rd = model.variable(f"lrc_down_{link.name}_{t}", lower=0.0)
                    link._ramp_up_vars.append(ru)
                    link._ramp_down_vars.append(rd)
                    if t == 0:
                        if not _ramp_cost_skip_t0:
                            _ref = link.ramp_t0_reference or 0.0
                            model.add(ru - link._flow_vars[t] >= -_ref,
                                      name=f"lrc_up_{link.name}_0")
                            model.add(rd + link._flow_vars[t] >= _ref,
                                      name=f"lrc_down_{link.name}_0")
                    else:
                        model.add(ru >= link._flow_vars[t] - link._flow_vars[t - 1],
                                  name=f"lrc_up_{link.name}_{t}")
                        model.add(rd >= link._flow_vars[t - 1] - link._flow_vars[t],
                                  name=f"lrc_down_{link.name}_{t}")

        # Ramp cost variables and constraints for storages.
        for sto in self._storages:
            if sto.ramp_cost > 0.0 and sto._discharge_vars and sto._charge_vars:
                for t in range(T):
                    net_t = sto._discharge_vars[t] - sto._charge_vars[t]
                    if _rc_signed:
                        r = model.variable(f"src_{sto.name}_{t}", lower=0.0)
                        sto._ramp_up_vars.append(r)
                        if t == 0:
                            if not _ramp_cost_skip_t0:
                                _ref = sto.ramp_t0_reference or 0.0
                                model.add(r - net_t >= -_ref,
                                          name=f"src_up_{sto.name}_0")
                                model.add(r + net_t >= _ref,
                                          name=f"src_down_{sto.name}_0")
                        else:
                            net_prev = (sto._discharge_vars[t - 1]
                                        - sto._charge_vars[t - 1])
                            model.add(r >= net_t - net_prev,
                                      name=f"src_up_{sto.name}_{t}")
                            model.add(r >= net_prev - net_t,
                                      name=f"src_down_{sto.name}_{t}")
                        continue
                    ru = model.variable(f"src_up_{sto.name}_{t}", lower=0.0)
                    rd = model.variable(f"src_down_{sto.name}_{t}", lower=0.0)
                    sto._ramp_up_vars.append(ru)
                    sto._ramp_down_vars.append(rd)
                    if t == 0:
                        if not _ramp_cost_skip_t0:
                            _ref = sto.ramp_t0_reference or 0.0
                            model.add(ru - net_t >= -_ref,
                                      name=f"src_up_{sto.name}_0")
                            model.add(rd + net_t >= _ref,
                                      name=f"src_down_{sto.name}_0")
                    else:
                        net_prev = sto._discharge_vars[t - 1] - sto._charge_vars[t - 1]
                        model.add(ru >= net_t - net_prev,
                                  name=f"src_up_{sto.name}_{t}")
                        model.add(rd >= net_prev - net_t,
                                  name=f"src_down_{sto.name}_{t}")

        # 18.P2 v4b — proximal boundary V-terms (temporal decomposition LB
        # blocks). Receiver-side |x - ref| charges are convex (aux var + two
        # rows); donor-side rebates -rate·|x - ref| are concave and need one
        # binary each (big-M side selection). All fields default to None.
        v_obj_terms: list = []
        for link in self._links:
            if link.flow_terminal_v_rebate is not None and link._flow_vars:
                _ref, _rate = link.flow_terminal_v_rebate
                _cap = link.capacity if link.capacity != float("inf") else 1e6
                _M = float(_cap + abs(_ref) + 1.0)
                m = model.variable(f"vreb_{link.name}", lower=0.0)
                b = model.binary(f"vrebb_{link.name}")
                f_last = link._flow_vars[-1]
                model.add(m - f_last + _M * b <= _M - _ref,
                          name=f"vreb_up_{link.name}")
                model.add(m + f_last - _M * b <= _ref,
                          name=f"vreb_dn_{link.name}")
                v_obj_terms.append(-float(_rate) * m)
        for sto in self._storages:
            if sto.soc_start_v_cost is not None and sto._soc_start_var is not None:
                _ref, _rate = sto.soc_start_v_cost
                a = model.variable(f"vstart_{sto.name}", lower=0.0)
                model.add(a - sto._soc_start_var >= -float(_ref),
                          name=f"vstart_up_{sto.name}")
                model.add(a + sto._soc_start_var >= float(_ref),
                          name=f"vstart_dn_{sto.name}")
                v_obj_terms.append(float(_rate) * a)
            if sto.soc_terminal_v_rebate is not None:
                _ref, _rate = sto.soc_terminal_v_rebate
                _term_var = sto._e_vars[-1] if sto._e_vars else sto._soc_vars[-1]
                _M = float(sto.soc_max * sto.energy_capacity + abs(_ref) + 1.0)
                m = model.variable(f"vsreb_{sto.name}", lower=0.0)
                b = model.binary(f"vsrebb_{sto.name}")
                model.add(m - _term_var + _M * b <= _M - float(_ref),
                          name=f"vsreb_up_{sto.name}")
                model.add(m + _term_var - _M * b <= float(_ref),
                          name=f"vsreb_dn_{sto.name}")
                v_obj_terms.append(-float(_rate) * m)
            if (sto.net_terminal_v_rebate is not None
                    and sto._discharge_vars and sto._charge_vars):
                _ref, _rate = sto.net_terminal_v_rebate
                _pmax = float(sto.power_capacity)
                _M = float(2.0 * _pmax + abs(_ref) + 1.0)
                m = model.variable(f"vnreb_{sto.name}", lower=0.0)
                b = model.binary(f"vnrebb_{sto.name}")
                net_last = sto._discharge_vars[-1] - sto._charge_vars[-1]
                model.add(m - net_last + _M * b <= _M - float(_ref),
                          name=f"vnreb_up_{sto.name}")
                model.add(m + net_last - _M * b <= float(_ref),
                          name=f"vnreb_dn_{sto.name}")
                v_obj_terms.append(-float(_rate) * m)

        # PWL heat-rate fuel cost: build segment vars once per generator.
        # The resulting cost term is added to the objective block below.
        pwl_cost_terms = []
        for gen in self._generators:
            if gen.heat_rate_segments is not None:
                term = build_pwl_heat_rate(model, gen, T)
                pwl_cost_terms.append(term)

        # Regulation reserve variables (per-generator reg_up / reg_down).
        for gen in self._generators:
            if gen.reg_up_max > 0.0 or gen.reg_down_max > 0.0:
                def _cap_only(gen=gen):
                    def fn(t):
                        if gen.extendable and gen._cap_var is not None:
                            return gen._cap_var  # use expansion var directly
                        base = gen.capacity
                        if gen.carrier_factor is not None:
                            return base * float(gen.carrier_factor[t])
                        return base
                    return fn
                add_regulation_reserve_vars(model, gen, T, _cap_only())

        # System-wide regulation reserve requirements.
        if self._reg_up_fraction is not None and T >= 1:
            for t in range(T):
                total_load_t = 0.0
                for load in self._loads:
                    d = load.amount
                    if isinstance(d, (np.ndarray, list, tuple)):
                        d = float(d[t])
                    total_load_t += d
                if total_load_t <= 0:
                    continue
                acc = None
                for gen in self._generators:
                    if not gen._reg_up_vars:
                        continue
                    term = gen._reg_up_vars[t]
                    acc = term if acc is None else acc + term
                if acc is not None:
                    model.add(
                        acc >= self._reg_up_fraction * total_load_t,
                        name=f"reg_up_req_{t}",
                    )
        if self._reg_down_fraction is not None and T >= 1:
            for t in range(T):
                total_load_t = 0.0
                for load in self._loads:
                    d = load.amount
                    if isinstance(d, (np.ndarray, list, tuple)):
                        d = float(d[t])
                    total_load_t += d
                if total_load_t <= 0:
                    continue
                acc = None
                for gen in self._generators:
                    if not gen._reg_down_vars:
                        continue
                    term = gen._reg_down_vars[t]
                    acc = term if acc is None else acc + term
                if acc is not None:
                    model.add(
                        acc >= self._reg_down_fraction * total_load_t,
                        name=f"reg_down_req_{t}",
                    )

        # Extendable capacity constraints — generators
        for gen in self._generators:
            if gen.extendable and gen._cap_var is not None:
                for t in range(T):
                    cf = 1.0
                    if gen.carrier_factor is not None:
                        cf = float(gen.carrier_factor[t])
                    if cf < 1.0:
                        model.add(
                            gen._p_vars[t] <= gen._cap_var * cf,
                            name=f"ext_cap_{gen.name}_{t}"
                        )
                    else:
                        model.add(
                            gen._p_vars[t] <= gen._cap_var,
                            name=f"ext_cap_{gen.name}_{t}"
                        )

        # Extendable capacity constraints — storage (full-mode only;
        # store-mode extendable storages handle SOC/power bounds via
        # constraints created in the store-mode variable section above).
        for sto in self._storages:
            if sto.extendable and not sto._e_vars:
                for t in range(T):
                    if sto._cap_power_var is not None:
                        model.add(
                            sto._charge_vars[t] <= sto._cap_power_var,
                            name=f"ext_ch_cap_{sto.name}_{t}"
                        )
                        model.add(
                            sto._discharge_vars[t] <= sto._cap_power_var,
                            name=f"ext_dis_cap_{sto.name}_{t}"
                        )
                    if sto._cap_energy_var is not None:
                        model.add(
                            sto._soc_vars[t] <= sto._cap_energy_var,
                            name=f"ext_soc_cap_{sto.name}_{t}"
                        )

        # Phase 4.x — no_simultaneous charge/discharge. Binary z[t] with a
        # big-M upper bound on both legs keeps the constraint linear even
        # for extendable storages (M = max_power_capacity bounds cap_var
        # from above, so ch ≤ M·z AND ch ≤ cap_var is a valid pair).
        for sto in self._storages:
            if not sto.no_simultaneous:
                continue
            if sto.extendable:
                m_ch = sto.max_power_capacity if sto.max_power_capacity != float("inf") else 1e12
                m_dis = m_ch
            else:
                m_ch = sto.pump_capacity if sto.pump_capacity is not None else sto.power_capacity
                m_dis = sto.turbine_capacity if sto.turbine_capacity is not None else sto.power_capacity
            sto._nosim_vars = []
            for t in range(T):
                z = model.binary(f"nosim_{sto.name}_{t}")
                sto._nosim_vars.append(z)
                model.add(
                    sto._charge_vars[t] <= m_ch * z,
                    name=f"nosim_ch_{sto.name}_{t}"
                )
                model.add(
                    sto._discharge_vars[t] <= m_dis * (1 - z),
                    name=f"nosim_dis_{sto.name}_{t}"
                )

        # Extendable capacity constraints — links
        for link in self._links:
            if link.extendable:
                if link._cap_var is None:
                    cap_lower = link.min_capacity
                    cap_upper = link.max_capacity if link.max_capacity != float("inf") else 1e12
                    link._cap_var = model.variable(
                        f"cap_{link.name}", lower=cap_lower, upper=cap_upper)
                for t in range(T):
                    if link._flow_signed_vars:
                        # |f_signed| <= cap. Two-sided to keep it linear.
                        # Covers DC-OPF / PTDF and symmetric bidi.
                        model.add(
                            link._flow_signed_vars[t] <= link._cap_var,
                            name=f"ext_flow_cap_{link.name}_{t}_pos"
                        )
                        model.add(
                            -link._flow_signed_vars[t] <= link._cap_var,
                            name=f"ext_flow_cap_{link.name}_{t}_neg"
                        )
                    elif link.bidirectional and link._flow_rev_vars:
                        # PyPSA Line: |p| <= s_nom_opt ≡ flow_fwd + flow_rev <= cap
                        model.add(
                            link._flow_vars[t] + link._flow_rev_vars[t] <= link._cap_var,
                            name=f"ext_bidi_cap_{link.name}_{t}"
                        )
                    else:
                        model.add(
                            link._flow_vars[t] <= link._cap_var,
                            name=f"ext_flow_cap_{link.name}_{t}"
                        )

        # Global emission limit constraint
        if hasattr(self, '_emission_limit') and self._emission_limit is not None:
            total_emissions = None
            for gen in self._generators:
                if gen.emission_factor > 0:
                    for t in range(T):
                        term = gen.emission_factor * gen._p_vars[t] * (w[t] * dts[t])
                        total_emissions = term if total_emissions is None else total_emissions + term
            if total_emissions is not None:
                model.add(total_emissions <= self._emission_limit, name="emission_cap")

        # Technology-bucket capacity carveouts (PyPSA carrier cap / GenX Min/MaxCapReq).
        # Sum (existing capacity + extendable cap_var) across matching generators
        # AND storage (power capacity — Phase 14).
        for b_idx, bucket in enumerate(self._capacity_buckets):
            total_cap = None
            n_matched = 0
            for gen in self._generators:
                if gen.tech != bucket["tech"]:
                    continue
                if bucket["bus"] is not None and gen.bus is not bucket["bus"]:
                    continue
                n_matched += 1
                if gen.extendable and gen._cap_var is not None:
                    term = gen._cap_var
                else:
                    term = gen.capacity  # constant
                total_cap = term if total_cap is None else total_cap + term
            for sto in self._storages:
                if sto.tech != bucket["tech"]:
                    continue
                if bucket["bus"] is not None and sto.bus is not bucket["bus"]:
                    continue
                n_matched += 1
                if sto.extendable and sto._cap_power_var is not None:
                    term = sto._cap_power_var
                else:
                    term = sto.power_capacity  # constant
                total_cap = term if total_cap is None else total_cap + term
            if n_matched == 0:
                continue
            b_tag = bucket["tech"] if bucket["bus"] is None else f"{bucket['tech']}_{bucket['bus'].name}"
            if bucket["min"] > 0.0:
                model.add(total_cap >= bucket["min"], name=f"cap_min_{b_tag}_{b_idx}")
            if bucket["max"] != float("inf"):
                model.add(total_cap <= bucket["max"], name=f"cap_max_{b_tag}_{b_idx}")

        # ------------------------------------------------------------------
        # Phase 6 — Policy library
        # ------------------------------------------------------------------

        # Precompute total load energy across the horizon (constant — used
        # by RPS, CES, and the CO2 rate cap). Snapshot weights scale here
        # too: a representative day with weight 365 contributes its load
        # 365× to the annual energy total.
        total_load_energy = 0.0
        for load in self._loads:
            d = load.amount
            if isinstance(d, (np.ndarray, list, tuple)):
                for t in range(T):
                    total_load_energy += float(d[t]) * w[t] * dts[t]
            else:
                total_load_energy += float(d) * float(w.sum()) * dt

        # System storage losses expr: sum_t ω·dt·(charge[t] − discharge[t])
        # (Phase 14 — reused on RHS of rate-based CO2 caps to match GenX
        # CO2Cap=2). Zero if no storages or all storages sit idle.
        # Store-mode storages are lossless by construction (round-trip ≈ 1
        # is the eligibility criterion), so they contribute nothing here.
        def _system_storage_losses():
            expr = None
            for sto in self._storages:
                if sto._e_vars:
                    continue
                for t in range(T):
                    term = (sto._charge_vars[t] - sto._discharge_vars[t]) * (w[t] * dts[t])
                    expr = term if expr is None else expr + term
            return expr

        def _zone_storage_losses(zbus, mode="net"):
            # mode="net": legacy Σ(charge−discharge) (≈0 for cyclic storage).
            # mode="dissipation": true round-trip loss (strictly ≥ 0).
            expr = None
            buses = list(zbus) if isinstance(zbus, (list, tuple, set)) else [zbus]
            for sto in self._storages:
                if not any(sto.bus is b for b in buses) or sto._e_vars:
                    continue
                if mode == "dissipation":
                    ec = (1.0 / sto.efficiency_charge - 1.0) if sto.efficiency_charge else 0.0
                    ed = (1.0 - sto.efficiency_discharge)
                    for t in range(T):
                        term = (ec * sto._charge_vars[t] + ed * sto._discharge_vars[t]) * (w[t] * dts[t])
                        expr = term if expr is None else expr + term
                else:
                    for t in range(T):
                        term = (sto._charge_vars[t] - sto._discharge_vars[t]) * (w[t] * dts[t])
                        expr = term if expr is None else expr + term
            return expr

        # CO2 rate cap: emissions ≤ rate × (delivered_energy [+ storage_losses]).
        if self._co2_rate_cap is not None:
            total_em = None
            for gen in self._generators:
                if gen.emission_factor > 0:
                    for t in range(T):
                        term = gen.emission_factor * gen._p_vars[t] * (w[t] * dts[t])
                        total_em = term if total_em is None else total_em + term
            if total_em is not None:
                rhs = self._co2_rate_cap * total_load_energy
                if self._co2_rate_cap_slosses:
                    sl = _system_storage_losses()
                    if sl is not None:
                        rhs = rhs + self._co2_rate_cap * sl
                model.add(total_em <= rhs, name="co2_rate_cap")

        # CO2 zone caps: absolute tCO2 by default; rate-based if is_rate=True.
        for z_idx, zone in enumerate(self._co2_zone_caps):
            zbus = zone["bus"]
            zem = None
            for gen in self._generators:
                if gen.bus is not zbus or gen.emission_factor <= 0:
                    continue
                for t in range(T):
                    term = gen.emission_factor * gen._p_vars[t] * (w[t] * dts[t])
                    zem = term if zem is None else zem + term
            if zem is None:
                continue
            if zone.get("is_rate", False):
                # Rate RHS: limit × (zone_load + [storage_losses]).
                zone_load_energy = 0.0
                for load in self._loads:
                    if load.bus is not zbus:
                        continue
                    d = load.amount
                    if isinstance(d, (np.ndarray, list, tuple)):
                        for t in range(T):
                            zone_load_energy += float(d[t]) * w[t] * dts[t]
                    else:
                        zone_load_energy += float(d) * float(w.sum()) * dt
                rhs = zone["limit"] * zone_load_energy
                if zone.get("storage_losses_on_rhs", True):
                    sl = _zone_storage_losses(zbus)
                    if sl is not None:
                        rhs = rhs + zone["limit"] * sl
                model.add(zem <= rhs, name=f"co2_zone_rate_cap_{zbus.name}_{z_idx}")
            else:
                model.add(zem <= zone["limit"],
                          name=f"co2_zone_cap_{zbus.name}_{z_idx}")

        # Phase 23 — pooled CO2 cap-group (GenX Cap_Zone): one constraint over
        # all group buses (tighter than independent per-bus caps).
        for g_idx, grp in enumerate(self._co2_cap_groups):
            gbuses = list(grp["buses"])
            gem = None
            for gen in self._generators:
                if not any(gen.bus is b for b in gbuses) or gen.emission_factor <= 0:
                    continue
                for t in range(T):
                    term = gen.emission_factor * gen._p_vars[t] * (w[t] * dts[t])
                    gem = term if gem is None else gem + term
            if gem is None:
                continue
            if grp.get("is_rate", True):
                grp_load = 0.0
                for load in self._loads:
                    if not any(load.bus is b for b in gbuses):
                        continue
                    d = load.amount
                    if isinstance(d, (np.ndarray, list, tuple)):
                        for t in range(T):
                            grp_load += float(d[t]) * w[t] * dts[t]
                    else:
                        grp_load += float(d) * float(w.sum()) * dt
                rhs = grp["limit"] * grp_load
                if grp.get("storage_losses_on_rhs", True):
                    sl = _zone_storage_losses(grp["buses"], grp.get("loss_accounting", "net"))
                    if sl is not None:
                        rhs = rhs + grp["limit"] * sl
                model.add(gem <= rhs, name=f"co2_cap_group_rate_{g_idx}")
            else:
                model.add(gem <= grp["limit"], name=f"co2_cap_group_{g_idx}")

        # Phase 6.2 — soft-policy slack penalty terms accumulated here and
        # folded into the objective just before model.minimize.
        policy_slack_terms = []

        # RPS: sum(p[qual,t]) ≥ fraction × total_load_energy.
        if self._rps is not None:
            qual_sum = None
            for gen in self._generators:
                if gen.tech in self._rps["techs"]:
                    for t in range(T):
                        term = gen._p_vars[t] * (w[t] * dts[t])
                        qual_sum = term if qual_sum is None else qual_sum + term
            if qual_sum is not None:
                rps_pen = self._rps.get("slack_penalty")
                if rps_pen is not None:
                    rps_slack = model.variable("rps_slack", lower=0.0,
                                               upper=self._rps["fraction"] * max(total_load_energy, 0.0))
                    model.add(
                        qual_sum + rps_slack >= self._rps["fraction"] * total_load_energy,
                        name="rps")
                    policy_slack_terms.append(rps_pen * rps_slack)
                else:
                    model.add(
                        qual_sum >= self._rps["fraction"] * total_load_energy,
                        name="rps")

        # Fuel supply limits: annual energy budget per fuel bucket
        # (GenX Fuel_Supply, Calliope / oemof / SpineOpt equivalents).
        gen_by_name = {gen.name: gen for gen in self._generators}
        for f_idx, fuel in enumerate(self._fuel_limits):
            total_fuel = None
            for g_name, heat_rate in fuel["coeffs"].items():
                gen = gen_by_name.get(g_name)
                if gen is None:
                    raise ValueError(
                        f"set_fuel_supply_limit({fuel['name']!r}): "
                        f"generator {g_name!r} not in system")
                if heat_rate == 0.0:
                    continue
                for t in range(T):
                    term = heat_rate * gen._p_vars[t] * (w[t] * dts[t])
                    total_fuel = term if total_fuel is None else total_fuel + term
            if total_fuel is not None:
                model.add(
                    total_fuel <= fuel["cap"],
                    name=f"fuel_supply_{fuel['name']}_{f_idx}")

        # CES: sum(score[tech] × p[gen,t]) ≥ fraction × total_load_energy.
        if self._ces is not None:
            weighted = None
            for gen in self._generators:
                score = self._ces["scores"].get(gen.tech, 0.0)
                if score == 0.0:
                    continue
                for t in range(T):
                    term = score * gen._p_vars[t] * (w[t] * dts[t])
                    weighted = term if weighted is None else weighted + term
            if weighted is not None:
                ces_pen = self._ces.get("slack_penalty")
                if ces_pen is not None:
                    ces_slack = model.variable("ces_slack", lower=0.0,
                                               upper=self._ces["fraction"] * max(total_load_energy, 0.0))
                    model.add(
                        weighted + ces_slack >= self._ces["fraction"] * total_load_energy,
                        name="ces")
                    policy_slack_terms.append(ces_pen * ces_slack)
                else:
                    model.add(
                        weighted >= self._ces["fraction"] * total_load_energy,
                        name="ces")

        # 24/7 hourly matching: qualifying dispatch into the tracked
        # load's bus ≥ that load's demand at every timestep.
        if self._hourly_matching is not None:
            target_name = self._hourly_matching["load_name"]
            target_load = next((ld for ld in self._loads if ld.name == target_name), None)
            if target_load is None:
                raise ValueError(
                    f"set_hourly_matching: load {target_name!r} not found.")
            techs = self._hourly_matching["techs"]
            target_bus = target_load.bus
            for t in range(T):
                d = target_load.amount
                d_t = float(d[t]) if isinstance(d, (np.ndarray, list, tuple)) else float(d)
                if d_t <= 0:
                    continue
                qsum = None
                for gen in self._generators:
                    if gen.bus is not target_bus or gen.tech not in techs:
                        continue
                    term = gen._p_vars[t]
                    qsum = term if qsum is None else qsum + term
                if qsum is None:
                    raise ValueError(
                        f"set_hourly_matching: no qualifying generators on "
                        f"bus {target_bus.name!r}.")
                model.add(qsum >= d_t, name=f"hourly_match_{t}")

        # Capacity reserve margin: derated firm cap ≥ (1 + margin) × peak load.
        if self._reserve_margin is not None:
            peak_override = self._reserve_margin.get("peak_override")
            if peak_override is not None:
                peak_load = float(peak_override)
            else:
                peak_load = 0.0
                for load in self._loads:
                    d = load.amount
                    if isinstance(d, (np.ndarray, list, tuple)):
                        peak_load += float(np.max(d))
                    else:
                        peak_load += float(d)
            firm_req = (1.0 + self._reserve_margin["margin"]) * peak_load
            firm_expr = None
            for gen in self._generators:
                credit = self._reserve_margin["firm_credit"].get(gen.tech, 0.0)
                if credit == 0.0:
                    continue
                cap_expr = gen._cap_var if (gen.extendable and gen._cap_var is not None) else gen.capacity
                term = credit * cap_expr
                firm_expr = term if firm_expr is None else firm_expr + term
            if firm_expr is None:
                # No generators tagged with firm credit — interpret as unmet.
                if firm_req > 0:
                    raise ValueError(
                        "set_reserve_margin: no generators match the firm_credit tags.")
            else:
                model.add(firm_expr >= firm_req, name="reserve_margin")

        # Spinning reserve: total dispatchable headroom ≥ fraction * load per timestep.
        # Only counts dispatchable (non-variable) generators by default — VRE with
        # carrier_factor doesn't contribute headroom. Applies across the system, not per-bus.
        if self._spinning_reserve is not None and self._spinning_reserve > 0:
            for t in range(T):
                total_load_t = 0.0
                for load in self._loads:
                    d = load.amount
                    if isinstance(d, (np.ndarray, list, tuple)):
                        d = float(d[t])
                    total_load_t += d
                if total_load_t <= 0:
                    continue
                headroom = None
                for gen in self._generators:
                    if gen.carrier_factor is not None:
                        continue  # VRE contributes no firm reserve
                    if gen.extendable and gen._cap_var is not None:
                        term = gen._cap_var - gen._p_vars[t]
                    else:
                        cap_base = gen.capacity
                        if gen.clustered and gen.n_units > 1:
                            cap_base = gen.capacity * gen.n_units
                        term = cap_base - gen._p_vars[t]
                    headroom = term if headroom is None else headroom + term
                if headroom is not None:
                    model.add(
                        headroom >= self._spinning_reserve * total_load_t,
                        name=f"spinning_reserve_{t}"
                    )

        # Phase 2.3 — single-largest-unit (N-1 generation) contingency reserve.
        # For each contingency unit g and timestep t, spinnable headroom from
        # the *other* units must cover g's output. Headroom counts only
        # committed-but-unused capacity (avail_cap·u − p for committable units;
        # avail_cap − p for always-on dispatchable units). VRE (carrier_factor)
        # contributes no firm reserve and is never a contingency unit.
        if self._contingency_reserve is not None:
            cset = self._contingency_reserve.get("generators")

            def _avail_cap(gen, t):
                if gen.extendable and gen._cap_var is not None:
                    cap = gen._cap_var
                else:
                    cap = gen.capacity * gen.n_units if (gen.clustered and gen.n_units > 1) \
                        else gen.capacity
                return cap

            firm = [g for g in self._generators if g.carrier_factor is None]
            contingency = [g for g in firm
                           if cset is None or g.name in cset]
            for g in contingency:
                for t in range(T):
                    others = None
                    for h in firm:
                        if h is g:
                            continue
                        cap_h = _avail_cap(h, t)
                        if h.committable and h._status_vars:
                            term = cap_h * h._status_vars[t] - h._p_vars[t]
                        else:
                            term = cap_h - h._p_vars[t]
                        others = term if others is None else others + term
                    if others is not None:
                        model.add(others >= g._p_vars[t],
                                  name=f"contingency_{g.name}_{t}")

        # Phase 4.4 — shared-capacity converter locks.
        if self._shared_caps:
            links_by_name = {lk.name: lk for lk in self._links}
            for spec in self._shared_caps:
                members = []
                for nm in spec["links"]:
                    lk = links_by_name.get(nm)
                    if lk is None:
                        raise KeyError(f"set_shared_capacity: no link named {nm!r}")
                    members.append(lk)
                # Tie extendable members to one shared rating.
                ext = [lk for lk in members if lk._cap_var is not None]
                ref = ext[0]._cap_var if ext else None
                for lk in ext[1:]:
                    model.add(lk._cap_var == ref,
                              name=f"sharedcap_eq_{lk.name}")
                if spec["mutex"]:
                    # Combined per-timestep throughput ≤ shared converter rating.
                    if ref is not None:
                        shared_cap = ref
                    else:
                        shared_cap = min(lk.capacity for lk in members)
                    def _flow_t(lk, t):
                        if lk._flow_signed_vars:
                            return lk._flow_signed_vars[t]
                        return lk._flow_vars[t]
                    for t in range(T):
                        acc = None
                        for lk in members:
                            term = _flow_t(lk, t)
                            acc = term if acc is None else acc + term
                        model.add(acc <= shared_cap,
                                  name=f"sharedcap_mutex_{spec['links'][0]}_{t}")

        # ---- Benders subproblem fix-caps (Phase 8) ----
        # Bind each listed cap_var to a scalar value via a named equality
        # constraint. The dual on that equality is the marginal operational
        # cost of one more MW of that capacity — the Benders cut coefficient
        # β_j. We record the constraint index so we can look up the dual by
        # position after solve.
        benders_fix_idx: dict[str, int] = {}
        if benders_fix_caps:
            for gen in self._generators:
                if gen.extendable and gen._cap_var is not None \
                        and gen.name in benders_fix_caps:
                    val = float(benders_fix_caps[gen.name])
                    benders_fix_idx[gen.name] = model.num_constraints
                    model.add(gen._cap_var == val,
                              name=f"benders_fix_{gen.name}")
            for sto in self._storages:
                pname = f"{sto.name}_power"
                ename = f"{sto.name}_energy"
                if sto._cap_power_var is not None and pname in benders_fix_caps:
                    val = float(benders_fix_caps[pname])
                    benders_fix_idx[pname] = model.num_constraints
                    model.add(sto._cap_power_var == val,
                              name=f"benders_fix_{pname}")
                if sto._cap_energy_var is not None and ename in benders_fix_caps:
                    val = float(benders_fix_caps[ename])
                    benders_fix_idx[ename] = model.num_constraints
                    model.add(sto._cap_energy_var == val,
                              name=f"benders_fix_{ename}")
            for link in self._links:
                if link.extendable and link._cap_var is not None \
                        and link.name in benders_fix_caps:
                    val = float(benders_fix_caps[link.name])
                    benders_fix_idx[link.name] = model.num_constraints
                    model.add(link._cap_var == val,
                              name=f"benders_fix_{link.name}")
            unresolved = set(benders_fix_caps) - set(benders_fix_idx)
            if unresolved:
                raise ValueError(
                    f"benders_fix_caps: no extendable cap_var matches "
                    f"{sorted(unresolved)!r}"
                )

        # ---- Phase 11 UC warm-start / variable-fix hook ----
        # ``uc_fix_schedule`` is an optional {gen_name: np.ndarray of length T}.
        # Entries that are ``np.nan`` leave the committable status free; finite
        # entries pin u[t] via equality. This is how ML predictors (and the
        # learned variable fixer) push high-confidence decisions into the MIP.
        if uc_fix_schedule:
            unresolved_uc: list[str] = []
            for name, schedule in uc_fix_schedule.items():
                gen = next((g for g in self._generators if g.name == name), None)
                link = next((l for l in self._links if l.name == name), None)
                target = None
                target_label = None
                if gen is not None and gen.committable and gen._status_vars:
                    target = gen
                    target_label = "gen"
                elif link is not None and link.committable and link._status_vars:
                    target = link
                    target_label = "link"
                if target is None:
                    unresolved_uc.append(name)
                    continue
                sched = np.asarray(schedule, dtype=float)
                if sched.shape[0] < T:
                    raise ValueError(
                        f"uc_fix_schedule[{name!r}] has length "
                        f"{sched.shape[0]} < {T}"
                    )
                for t, u_var in enumerate(target._status_vars):
                    val = sched[t]
                    if np.isnan(val):
                        continue
                    model.add(u_var == float(val),
                              name=f"uc_fix_{target_label}_{target.name}_{t}")
            if unresolved_uc:
                raise ValueError(
                    f"uc_fix_schedule: no committable generator/link matches "
                    f"{sorted(unresolved_uc)!r}"
                )

        # ---- Objective ----

        obj = None

        if objective == "min_cost":
            # CO2 price: add $/tCO2 × emission_factor to each emitting generator's effective mc.
            co2_price = self._co2_price or 0.0

            # PWL fuel cost terms (one per generator that has heat_rate_segments).
            # pwl_cost_terms was populated in the UC / PWL pass above.
            for term in pwl_cost_terms:
                obj = term if obj is None else obj + term

            for gen in self._generators:
                # Flat marginal cost applies to non-PWL generators. For PWL
                # generators the segment cost already covers fuel; the
                # ``marginal_cost`` value still represents a $/MWh var-OM
                # line-item, so we add it separately.
                if gen.heat_rate_segments is None:
                    eff_mc = gen.marginal_cost + co2_price * gen.emission_factor
                else:
                    # Fuel already counted via PWL segments — just var-OM + CO2.
                    eff_mc = gen.marginal_cost + co2_price * gen.emission_factor
                    # NOTE: if the user builds segments whose slopes already
                    # include CO2 cost, they should pass emission_factor=0.
                # Phase 6 — PTC: subsidy reduces the effective marginal cost.
                ptc_credit = self._ptc.get(gen.tech, 0.0) if gen.tech else 0.0
                eff_mc = eff_mc - ptc_credit
                mc_is_array = isinstance(eff_mc, np.ndarray)
                for t in range(T):
                    mc_t = float(eff_mc[t]) if mc_is_array else eff_mc
                    if mc_t == 0.0:
                        continue
                    cost_term = mc_t * gen._p_vars[t] * (w[t] * dts[t])
                    obj = cost_term if obj is None else obj + cost_term
                # Startup cost via the v[t] 3-bin startup indicator. ``startup_fuel_cost``
                # is Phase 2.x fuel-per-start (GenX ``Start_Fuel_MMBTU_per_MW × cap_size ×
                # fuel_price``); kept as a distinct field for reporting but fused
                # onto the same coefficient.
                startup_coef = gen.startup_cost + gen.startup_fuel_cost
                if gen.start_up_segments and gen._start_type_vars:
                    # Phase 2.2 — multi-state start: cost depends on the start
                    # type chosen (δ[t,s]); overrides the flat startup_cost.
                    segs = sorted(gen.start_up_segments, key=lambda x: x[0])
                    for t in range(1, T):
                        for s, (_off, seg_cost) in enumerate(segs):
                            c = (seg_cost + gen.startup_fuel_cost) * w[t]
                            if c == 0.0:
                                continue
                            term = c * gen._start_type_vars[t][s]
                            obj = term if obj is None else obj + term
                elif startup_coef > 0.0 and gen._startup_vars:
                    for t in range(1, T):
                        term = (startup_coef * w[t]) * gen._startup_vars[t]
                        obj = term if obj is None else obj + term
                # Shutdown cost via the w[t] 3-bin shutdown indicator.
                if gen.shutdown_cost > 0.0 and gen._shutdown_vars:
                    for t in range(1, T):
                        term = (gen.shutdown_cost * w[t]) * gen._shutdown_vars[t]
                        obj = term if obj is None else obj + term
                # Phase 16.x — no-load / idling cost applied to u[t] so a
                # committed unit pays even when dispatching zero (Tulipa
                # ``units_on_cost``). Weighted by snapshot weight × dt like
                # operational costs.
                if gen.no_load_cost > 0.0 and gen._status_vars:
                    for t in range(T):
                        term = (gen.no_load_cost * w[t] * dts[t]) * gen._status_vars[t]
                        obj = term if obj is None else obj + term
                if gen.extendable and gen._cap_var is not None and not benders_skip_capex:
                    # Phase 6 — ITC reduces effective capital cost.
                    itc = self._itc.get(gen.tech, 0.0) if gen.tech else 0.0
                    itc_mult = (1.0 - itc)
                    # PWL CapEx overrides flat capital_cost when segments are set.
                    if gen._capex_seg_vars:
                        for seg, slope in zip(gen._capex_seg_vars, gen._capex_seg_slopes):
                            term = (itc_mult * slope) * seg
                            obj = term if obj is None else obj + term
                    elif gen.capital_cost != 0.0:
                        term = (itc_mult * gen.capital_cost) * gen._cap_var
                        obj = term if obj is None else obj + term
                    # Fixed O&M ($/MW/year) — paid on the chosen cap even when
                    # idle; enables endogenous retirement with min_capacity=0.
                    if gen.fixed_om != 0.0:
                        term = gen.fixed_om * gen._cap_var
                        obj = term if obj is None else obj + term
                elif gen.fixed_om != 0.0 and not benders_skip_capex:
                    # Non-extendable: fixed_om is a constant but still shows up
                    # in total_cost so the user sees the retained-cost line.
                    term = gen.fixed_om * gen.capacity
                    obj = term if obj is None else obj + term

            for sto in self._storages:
                # Discharge var-OM (PyPSA convention). Charge var-OM is separate (GenX applies both).
                # Store-mode storages have empty _charge/_discharge_vars and
                # zero marginal_cost (the eligibility check ensures this), so
                # the loops are no-ops for them.
                if sto._discharge_vars:
                    for t in range(T):
                        cost_term = sto.marginal_cost * sto._discharge_vars[t] * (w[t] * dts[t])
                        obj = cost_term if obj is None else obj + cost_term
                if sto.marginal_cost_charge > 0.0 and sto._charge_vars:
                    for t in range(T):
                        obj = obj + sto.marginal_cost_charge * sto._charge_vars[t] * (w[t] * dts[t])
                # Spill cost (Phase 4) — applies when inflow is set; defaults to
                # a tiny tie-breaker so the LP doesn't waste water gratuitously.
                if sto._spill_vars and sto.spill_cost != 0.0:
                    for t in range(T):
                        obj = obj + sto.spill_cost * sto._spill_vars[t] * (w[t] * dts[t])
                if sto.ramp_cost > 0.0 and sto._ramp_up_vars:
                    for t in range(T):
                        if sto._ramp_down_vars:
                            term = sto.ramp_cost * (sto._ramp_up_vars[t] + sto._ramp_down_vars[t]) * w[t]
                        else:  # signed form: one var already equals |Δ|
                            term = sto.ramp_cost * sto._ramp_up_vars[t] * w[t]
                        obj = term if obj is None else obj + term
                if sto.extendable and not benders_skip_capex:
                    if sto._cap_power_var is not None:
                        obj = obj + sto.capital_cost_power * sto._cap_power_var
                    if sto._cap_energy_var is not None:
                        obj = obj + sto.capital_cost_energy * sto._cap_energy_var
                # 18.P2 Lagrangian boundary prices (default 0.0 → no-op).
                if sto.soc_start_cost != 0.0 and sto._soc_start_var is not None:
                    obj = obj + sto.soc_start_cost * sto._soc_start_var
                if sto.soc_terminal_cost != 0.0:
                    _term_var = sto._e_vars[-1] if sto._e_vars else sto._soc_vars[-1]
                    obj = obj + sto.soc_terminal_cost * _term_var
                # 18.P2 v4 — boundary net-flow prices (full-mode storages).
                if sto.net_t0_cost != 0.0 and sto._discharge_vars:
                    obj = obj + sto.net_t0_cost * (
                        sto._discharge_vars[0] - sto._charge_vars[0])
                if sto.net_terminal_cost != 0.0 and sto._discharge_vars:
                    obj = obj + sto.net_terminal_cost * (
                        sto._discharge_vars[-1] - sto._charge_vars[-1])
            for link in self._links:
                # DC-OPF / PTDF lines have no transport flow var; their marginal
                # cost (typically zero for transmission) doesn't apply here. If
                # users want a per-MW dispatch cost on a DC line they should add
                # it in the transport branch.
                if link._flow_vars:
                    lmc_is_array = isinstance(link.marginal_cost, np.ndarray)
                    for t in range(T):
                        lmc_t = float(link.marginal_cost[t]) if lmc_is_array else link.marginal_cost
                        cost_term = lmc_t * link._flow_vars[t] * (w[t] * dts[t])
                        obj = cost_term if obj is None else obj + cost_term
                if link.extendable and link._cap_var is not None and not benders_skip_capex:
                    obj = obj + link.capital_cost * link._cap_var
                if link.startup_cost > 0.0 and link._startup_vars:
                    for t in range(1, T):
                        term = (link.startup_cost * w[t]) * link._startup_vars[t]
                        obj = term if obj is None else obj + term
                if link.shutdown_cost > 0.0 and link._shutdown_vars:
                    for t in range(1, T):
                        term = (link.shutdown_cost * w[t]) * link._shutdown_vars[t]
                        obj = term if obj is None else obj + term
                if link.ramp_cost > 0.0 and link._ramp_up_vars:
                    for t in range(T):
                        if link._ramp_down_vars:
                            term = link.ramp_cost * (link._ramp_up_vars[t] + link._ramp_down_vars[t]) * w[t]
                        else:  # signed form: one var already equals |Δ|
                            term = link.ramp_cost * link._ramp_up_vars[t] * w[t]
                        obj = term if obj is None else obj + term
                # 18.P2 v4 — boundary flow prices (default 0.0 → no-op).
                if link.flow_t0_cost != 0.0 and link._flow_vars:
                    obj = obj + link.flow_t0_cost * link._flow_vars[0]
                if link.flow_terminal_cost != 0.0 and link._flow_vars:
                    obj = obj + link.flow_terminal_cost * link._flow_vars[-1]

            # Phase 6.2 — soft-policy slack penalties ($/MWh of shortfall).
            for term in policy_slack_terms:
                obj = term if obj is None else obj + term

            # 18.P2 v4b — proximal boundary V-terms (charges and rebates).
            for term in v_obj_terms:
                obj = term if obj is None else obj + term

            # Phase 14 — capex_mode="incremental": subtract the constant
            # `capital_cost × min_capacity` baseline so the reported total
            # matches PyPSA's convention of charging only new capacity above
            # the floor. Constant terms don't affect argmin; this is an
            # accounting flag, not a physics change.
            if self._capex_mode == "incremental" and not benders_skip_capex:
                baseline = 0.0
                for gen in self._generators:
                    if gen.extendable and gen._cap_var is not None:
                        itc = self._itc.get(gen.tech, 0.0) if gen.tech else 0.0
                        baseline += (1.0 - itc) * gen.capital_cost * gen.min_capacity
                for sto in self._storages:
                    if sto.extendable:
                        if sto._cap_power_var is not None:
                            baseline += sto.capital_cost_power * sto.min_power_capacity
                        if sto._cap_energy_var is not None:
                            baseline += sto.capital_cost_energy * sto.min_energy_capacity
                for link in self._links:
                    if link.extendable and link._cap_var is not None:
                        baseline += link.capital_cost * link.min_capacity
                if baseline != 0.0 and obj is not None:
                    obj = obj - baseline

        elif objective == "min_emissions":
            for gen in self._generators:
                for t in range(T):
                    em_term = gen.emission_factor * gen._p_vars[t] * (w[t] * dts[t])
                    obj = em_term if obj is None else obj + em_term
        else:
            raise ValueError(f"Unknown objective: {objective!r}")

        if model_hook is not None:
            extra = model_hook(model, self, obj)
            if extra is not None:
                obj = extra

        if obj is not None:
            model.minimize(obj)

        # ---- Solve ----

        # ---- Lossless solver-path selection ----
        # Every choice here is bit-exact: it changes the *path* to the
        # optimum, never the optimum itself.
        #
        # Two levers, very different risk profiles (both measured 2026-05-29):
        #   * parallel dual simplex (`parallel=on`) — lossless and
        #     non-regressing across cases, so default it on when the caller
        #     asked for >1 thread.
        #   * interior-point + crossover (`solver_method=ipm`) — a BIG win on
        #     network-dominated LPs (scigrid native 13.85→5.87 s, 2.36×) but a
        #     BIG loss on large temporal-staircase LPs, where the crossover
        #     over hundreds of thousands of columns dominates (CINDER LP
        #     158→406 s). Problem *size* is the wrong signal and an earlier
        #     "≥50 K cols ⇒ ipm" auto-rule REGRESSED CINDER, so IPM is now
        #     strictly opt-in. If a robust structural signal (e.g. bus-count ≫
        #     snapshot-count) is found later, re-enable auto-selection here.
        # Resolve lp_backend when the caller didn't set it explicitly:
        #   explicit lp_backend=  >  .nexus_solver.json sidecar  >  built-in default.
        # The sidecar is written by nexus_energy.solver_tuner (the method
        # calibrator); it is advisory and silently ignored if absent/unreadable.
        if lp_backend is None:
            from nexus_energy.solver_tuner import sidecar_lp_backend
            lp_backend = sidecar_lp_backend() or "ipm_fast"

        eff_method = solver_method
        eff_crossover = run_crossover
        eff_parallel = parallel
        _ipm_fast_lp = False   # set when the ipm_fast LP path is taken (fallback hook)

        # P1 (18.u) — `lp_backend` convenience switch. A user-facing layer over
        # the solver_method/crossover plumbing so callers can pick an LP engine
        # without knowing HiGHS option names. Default "auto" = current behaviour
        # (no method forced) → byte-identical, no regression. An explicit
        # `solver_method` always wins over `lp_backend`.
        #   simplex : dual simplex (the legacy default path)
        #   ipm     : interior point + crossover (exact vertex)
        #   pdlp    : HiGHS first-order PDLP + crossover (CPU; for huge sparse LP)
        #   gpu     : cuOpt GPU PDLP + crossover, FALLING BACK to CPU pdlp when
        #             no CUDA device is present (e.g. this macOS box). The switch
        #             is real and future-ready; it never silently runs simplex.
        # Crossover is forced on for ipm/pdlp/gpu so the returned point is a
        # true vertex (bit-exact + valid MILP root) — accuracy over speed.
        if eff_method is None and lp_backend and lp_backend != "auto":
            lb = lp_backend.lower()
            if lb == "gpu":
                # Probe for cuOpt + a CUDA device. Absent on CPU-only boxes
                # (e.g. macOS) → fall back to CPU first-order PDLP. The switch
                # never silently degrades to simplex.
                _gpu_ok = False
                try:
                    import importlib
                    _cu = importlib.util.find_spec("cuopt")
                    if _cu is not None:
                        import importlib as _il
                        _cuda = _il.import_module("cuopt")  # noqa: F841
                        _gpu_ok = True
                except Exception:
                    _gpu_ok = False
                if _gpu_ok:
                    eff_method = "cuopt_pdlp"   # routed in nexus-opt solve path
                else:
                    if verbose:
                        print("[nexus] lp_backend=gpu: no cuOpt/CUDA device → "
                              "falling back to CPU PDLP")
                    eff_method = "pdlp"
                if eff_crossover is None:
                    eff_crossover = "on"
            elif lb == "pdlp":
                eff_method = "pdlp"
                if eff_crossover is None:
                    eff_crossover = "on"
            elif lb == "ipm":
                eff_method = "ipm"
            elif lb == "ipm_fast":
                # IPM WITHOUT crossover. ~1.5x faster than dual simplex on
                # well-conditioned expansion LPs (PyPSA-Eur 730h 29.6->19.7s,
                # exact objective), but returns an INTERIOR (non-vertex) point:
                # individual dispatch values + duals on a degenerate LP can
                # differ from the simplex vertex, and there is no basis for a
                # MILP root. So: pure-LP only — for MILP (integers present and
                # not relaxed) fall back to the unforced default. A non-optimal
                # result triggers a simplex re-solve below (handles IPM-hostile
                # conditioning, e.g. CINDER, which stalls/fails under IPM).
                if _has_integers and not _relax_integers:
                    eff_method = None          # MILP -> leave unforced
                else:
                    eff_method = "ipm"
                    if eff_crossover is None:
                        eff_crossover = "off"
                    _ipm_fast_lp = True
            elif lb == "simplex":
                eff_method = "simplex"
            else:
                raise ValueError(
                    f"unknown lp_backend {lp_backend!r}; expected one of "
                    "auto|simplex|ipm|ipm_fast|pdlp|gpu")

        if eff_method == "ipm" and eff_crossover is None:
            eff_crossover = "on"   # exact vertex + duals — keeps IPM lossless
        if eff_parallel is None and threads is not None and threads > 1:
            eff_parallel = "on"

        solve_kwargs = {}
        if solver is not None:
            # Phase 10.9 — third-party LP/MILP solvers are reached through the
            # explicit LP-export bridge, not this native path, because they are
            # not part of the nexus-opt core. Fail loudly with a pointer rather
            # than silently handing an unknown solver name to HiGHS.
            if solver.lower() in ("gurobi", "cplex", "scip", "mosek", "xpress"):
                raise NotImplementedError(
                    f"Solver {solver!r} is an external third-party solver. Use "
                    f"nexus_energy.external_solvers.solve_system_external("
                    f"system, {solver!r}) (LP-export bridge), or solve natively "
                    f"with HiGHS / OSQP / Clarabel / Ipopt.")
            solve_kwargs["solver"] = solver
        if time_limit is not None:
            solve_kwargs["time_limit"] = time_limit
        if gap is not None:
            solve_kwargs["gap"] = gap
        if threads is not None:
            solve_kwargs["threads"] = threads
        if eff_method is not None:
            solve_kwargs["solver_method"] = eff_method
        if eff_crossover is not None:
            solve_kwargs["run_crossover"] = eff_crossover
        if eff_parallel is not None:
            solve_kwargs["parallel"] = eff_parallel
        if scale_cleanup:
            solve_kwargs["scale_cleanup"] = True
        if simplex_scale_strategy is not None:
            solve_kwargs["simplex_scale_strategy"] = simplex_scale_strategy
        if eliminate_redundant:
            solve_kwargs["eliminate_redundant"] = True
        solve_kwargs["verbose"] = verbose
        solve_kwargs["presolve"] = presolve
        if warm_start is not None:
            solve_kwargs["warm_start"] = warm_start
        if basis is not None:
            solve_kwargs["basis"] = basis

        t_start = time.perf_counter()
        raw_result = model.solve(**solve_kwargs)
        solve_time = time.perf_counter() - t_start

        # ipm_fast safety net: if IPM-without-crossover did not return a clean
        # optimum (IPM-hostile conditioning can stall or fail), re-solve once
        # with dual simplex so the default never silently degrades correctness.
        if _ipm_fast_lp and raw_result.status not in ("optimal", "time_limit"):
            fallback_kwargs = dict(solve_kwargs)
            fallback_kwargs["solver_method"] = "simplex"
            fallback_kwargs.pop("run_crossover", None)
            t_start = time.perf_counter()
            raw_result = model.solve(**fallback_kwargs)
            solve_time = time.perf_counter() - t_start

        # ---- Extract results ----

        result = OptimisationResult(
            status=raw_result.status,
            total_cost=raw_result.objective if raw_result.objective is not None else float("nan"),
            solve_time=solve_time,
            _raw=raw_result,
            _balance_row_idx=dict(balance_idx),
        )

        if raw_result.status in ("optimal", "time_limit"):
            # Generator dispatch
            for gen in self._generators:
                vals = np.array([raw_result.value(v) for v in gen._p_vars])
                result.generator_dispatch[gen.name] = vals
                if gen.extendable and gen._cap_var is not None:
                    result.capacity_additions[gen.name] = raw_result.value(gen._cap_var)
                # UC schedule extraction (Phase 11 warm-start hook).
                if gen.committable and gen._status_vars:
                    result.unit_status[gen.name] = np.array(
                        [raw_result.value(u) for u in gen._status_vars])

            # Storage. For store-mode storages, derive charge/discharge
            # series from the signed Δe between successive timesteps so
            # downstream consumers (tests, comparison scripts) see the
            # same shape of arrays regardless of the underlying model.
            for sto in self._storages:
                if sto._e_vars:
                    e_vals = np.array(
                        [raw_result.value(v) for v in sto._e_vars])
                    result.storage_soc[sto.name] = e_vals
                    sd_factor = 1.0 - sto.self_discharge * dt
                    charge_arr = np.zeros(T)
                    discharge_arr = np.zeros(T)
                    for t in range(T):
                        if t == 0:
                            if sto.cyclic and T > 1:
                                e_prev = e_vals[T - 1]
                            elif sto.extendable and sto._cap_energy_var is not None:
                                e_prev = sto.soc_initial * raw_result.value(sto._cap_energy_var)
                            elif sto._soc_start_var is not None:
                                e_prev = raw_result.value(sto._soc_start_var)
                            else:
                                e_prev = sto.soc_initial * sto.energy_capacity
                        else:
                            e_prev = e_vals[t - 1]
                        delta = e_vals[t] - sd_factor * e_prev
                        # Δe > 0 → bus → storage (charge);
                        # Δe < 0 → storage → bus (discharge).
                        if delta > 0:
                            charge_arr[t] = delta / dt
                        elif delta < 0:
                            discharge_arr[t] = -delta / dt
                    result.storage_charge[sto.name] = charge_arr
                    result.storage_discharge[sto.name] = discharge_arr
                else:
                    result.storage_charge[sto.name] = np.array(
                        [raw_result.value(v) for v in sto._charge_vars])
                    result.storage_discharge[sto.name] = np.array(
                        [raw_result.value(v) for v in sto._discharge_vars])
                    result.storage_soc[sto.name] = np.array(
                        [raw_result.value(v) for v in sto._soc_vars])

            # Storage investment
            for sto in self._storages:
                if sto.extendable:
                    if sto._cap_power_var is not None:
                        result.capacity_additions[f"{sto.name}_power"] = raw_result.value(sto._cap_power_var)
                    if sto._cap_energy_var is not None:
                        result.capacity_additions[f"{sto.name}_energy"] = raw_result.value(sto._cap_energy_var)

            # Links
            for link in self._links:
                if link._flow_signed_vars:
                    # DC-OPF / PTDF: signed flow (positive = bus_from → bus_to).
                    result.link_flow[link.name] = np.array(
                        [raw_result.value(v) for v in link._flow_signed_vars])
                else:
                    result.link_flow[link.name] = np.array(
                        [raw_result.value(v) for v in link._flow_vars])
                if link.extendable and link._cap_var is not None:
                    result.capacity_additions[link.name] = raw_result.value(link._cap_var)

            # Bus shadow prices: look up dual values directly by constraint index.
            # Avoids nexus-opt's full sensitivity() pass, which is O(C*V).
            try:
                duals = raw_result.duals
            except Exception:
                duals = None
            if duals is not None and balance_idx:
                for bus in self._buses:
                    arr = np.empty(T, dtype=float)
                    arr.fill(np.nan)
                    for t in range(T):
                        idx = balance_idx.get((bus.name, t))
                        if idx is not None and 0 <= idx < len(duals):
                            # Divide by dt so prices are in $/MWh, not $/MW·dt.
                            arr[t] = float(duals[idx]) / dt
                    result.bus_shadow_prices[bus.name] = arr
            if duals is not None and soc_idx:
                for sto in self._storages:
                    if not sto._soc_vars:
                        continue
                    arr = np.empty(T, dtype=float)
                    arr.fill(np.nan)
                    for t in range(T):
                        idx = soc_idx.get((sto.name, t))
                        if idx is not None and 0 <= idx < len(duals):
                            arr[t] = float(duals[idx])  # $/MWh of stored energy
                    result.storage_soc_duals[sto.name] = arr
            if duals is not None and soc_fixed_idx:
                for key, idx in soc_fixed_idx.items():
                    if 0 <= idx < len(duals):
                        result.soc_fixed_duals[key] = float(duals[idx])

            # Phase 8 — Benders cut coefficients. The dual on the
            # ``cap_var == fixed`` equality is ∂(operational cost) / ∂cap_j,
            # which is exactly the coefficient β_j used in the optimality
            # cut ``θ ≥ sub_cost + Σ_j β_j * (cap_j - cap_fixed_j)``.
            if duals is not None and benders_fix_idx:
                for name, idx in benders_fix_idx.items():
                    if 0 <= idx < len(duals):
                        result.cap_dual[name] = float(duals[idx])

        return result
