"""
Phase 11: Diagnostics, Reporting & Visualisation.

Post-processing of optimisation results:
- Curtailment analysis
- Bottleneck detection
- Duration curves
- Dispatch summaries
- Energy balance verification
- Infeasibility explanation

Visualisations require plotly (optional).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem, OptimisationResult


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CurtailmentReport:
    """Per-generator curtailment analysis."""
    curtailment: dict[str, np.ndarray]  # MW curtailed per timestep
    curtailment_fraction: dict[str, float]  # fraction of potential generation
    total_curtailed: dict[str, float]  # total MWh
    message: str


@dataclass
class BottleneckReport:
    """Binding constraint / capacity bottleneck analysis."""
    at_capacity: dict[str, np.ndarray]  # components at capacity, per timestep
    utilisation: dict[str, float]  # fraction of time at capacity
    saturated_hours: dict[str, int]
    message: str


@dataclass
class DispatchSummary:
    """Aggregate dispatch statistics."""
    generator_total: dict[str, float]  # total MWh over horizon
    generator_capacity_factor: dict[str, float]  # CF achieved
    storage_cycles: dict[str, float]  # equivalent full cycles
    link_utilisation: dict[str, float]
    total_generation: float
    total_demand: float
    total_curtailment: float
    total_cost: float


# ---------------------------------------------------------------------------
# Diagnostics engine
# ---------------------------------------------------------------------------

class Diagnostics:
    """
    Attach post-processing diagnostics to an EnergySystem + its result.

    Usage:
        >>> result = sys.optimise()
        >>> diag = Diagnostics(sys, result)
        >>> print(diag.summary())
        >>> print(diag.curtailment_report())
    """

    def __init__(self, system: "EnergySystem", result: "OptimisationResult"):
        self.system = system
        self.result = result
        self._T = system._timesteps
        self._dt = system._dt

    # ---- Curtailment ----

    def curtailment_report(self, threshold: float = 1e-4) -> CurtailmentReport:
        """
        Analyse curtailment for each generator with a carrier_factor.

        Curtailment = (potential output) - (actual output).
        Only meaningful for generators with time-varying availability (solar/wind).
        """
        curtail = {}
        fraction = {}
        total = {}

        for gen in self.system._generators:
            if gen.carrier_factor is None or gen.name not in self.result.generator_dispatch:
                continue
            actual = self.result.generator_dispatch[gen.name]
            capacity = gen.capacity
            if gen.extendable and gen.name in self.result.capacity_additions:
                capacity = self.result.capacity_additions[gen.name]
            if capacity < 1e-6:
                continue
            cf = np.asarray(gen.carrier_factor[:len(actual)])
            potential = cf * capacity
            curt = np.maximum(potential - actual, 0.0)
            total_pot = potential.sum() * self._dt
            total_curt = curt.sum() * self._dt
            curtail[gen.name] = curt
            fraction[gen.name] = total_curt / total_pot if total_pot > threshold else 0.0
            total[gen.name] = total_curt

        # Message
        lines = ["Curtailment Report:"]
        if not curtail:
            lines.append("  No variable-output generators found.")
        else:
            for name, frac in sorted(fraction.items()):
                lines.append(
                    f"  {name:25s}  curtailed: {total[name]:8.1f} MWh "
                    f"({100*frac:5.1f}% of potential)"
                )

        return CurtailmentReport(
            curtailment=curtail,
            curtailment_fraction=fraction,
            total_curtailed=total,
            message="\n".join(lines),
        )

    # ---- Bottlenecks ----

    def bottleneck_report(self, tol: float = 1e-3) -> BottleneckReport:
        """
        Identify components operating at their capacity limit.
        These are candidates for expansion.
        """
        at_cap = {}
        util = {}
        sat_hours = {}

        for gen in self.system._generators:
            if gen.name not in self.result.generator_dispatch:
                continue
            actual = self.result.generator_dispatch[gen.name]
            cap = gen.capacity
            if gen.extendable and gen.name in self.result.capacity_additions:
                cap = self.result.capacity_additions[gen.name]
            if cap < 1e-6:
                continue
            # Account for time-varying CF
            if gen.carrier_factor is not None:
                cf = np.asarray(gen.carrier_factor[:len(actual)])
                limit = cap * cf
            else:
                limit = np.full(len(actual), cap)
            at = (actual >= limit - tol) & (limit > 1e-6)
            at_cap[gen.name] = at
            util[gen.name] = float(at.mean())
            sat_hours[gen.name] = int(at.sum())

        for link in self.system._links:
            if link.name not in self.result.link_flow:
                continue
            flow = self.result.link_flow[link.name]
            cap = link.capacity
            if link.extendable and link.name in self.result.capacity_additions:
                cap = self.result.capacity_additions[link.name]
            if cap < 1e-6:
                continue
            at = flow >= cap - tol
            at_cap[link.name] = at
            util[link.name] = float(at.mean())
            sat_hours[link.name] = int(at.sum())

        lines = ["Bottleneck Report:"]
        sorted_by_util = sorted(util.items(), key=lambda x: -x[1])
        for name, u in sorted_by_util:
            if u > 0.01:
                lines.append(
                    f"  {name:25s}  at capacity {100*u:5.1f}% of time "
                    f"({sat_hours[name]}/{self._T} timesteps)"
                )
        if not any(u > 0.01 for u in util.values()):
            lines.append("  No significant bottlenecks found.")

        return BottleneckReport(
            at_capacity=at_cap,
            utilisation=util,
            saturated_hours=sat_hours,
            message="\n".join(lines),
        )

    # ---- Dispatch summary ----

    def dispatch_summary(self) -> DispatchSummary:
        """Aggregate dispatch statistics."""
        gen_total = {}
        gen_cf = {}
        total_gen = 0.0

        for gen in self.system._generators:
            if gen.name not in self.result.generator_dispatch:
                continue
            dispatch = self.result.generator_dispatch[gen.name]
            total_mwh = dispatch.sum() * self._dt
            gen_total[gen.name] = total_mwh
            total_gen += total_mwh
            cap = gen.capacity
            if gen.extendable and gen.name in self.result.capacity_additions:
                cap = self.result.capacity_additions[gen.name]
            if cap > 1e-6 and self._T > 0:
                potential = cap * self._T * self._dt
                gen_cf[gen.name] = total_mwh / potential
            else:
                gen_cf[gen.name] = 0.0

        # Storage cycling
        storage_cycles = {}
        for sto in self.system._storages:
            if sto.name not in self.result.storage_discharge:
                continue
            total_discharged = self.result.storage_discharge[sto.name].sum() * self._dt
            e_cap = sto.energy_capacity
            if sto.extendable and f"{sto.name}_energy" in self.result.capacity_additions:
                e_cap = self.result.capacity_additions[f"{sto.name}_energy"]
            storage_cycles[sto.name] = total_discharged / e_cap if e_cap > 1e-6 else 0.0

        # Link utilisation
        link_util = {}
        for link in self.system._links:
            if link.name not in self.result.link_flow:
                continue
            flow = self.result.link_flow[link.name]
            cap = link.capacity
            if link.extendable and link.name in self.result.capacity_additions:
                cap = self.result.capacity_additions[link.name]
            if cap > 1e-6:
                link_util[link.name] = float(flow.mean() / cap)
            else:
                link_util[link.name] = 0.0

        # Total demand
        total_demand = 0.0
        for load in self.system._loads:
            amount = load.amount
            if isinstance(amount, np.ndarray):
                total_demand += amount.sum() * self._dt
            else:
                total_demand += amount * self._T * self._dt

        # Total curtailment
        curt = self.curtailment_report()
        total_curtailed = sum(curt.total_curtailed.values())

        return DispatchSummary(
            generator_total=gen_total,
            generator_capacity_factor=gen_cf,
            storage_cycles=storage_cycles,
            link_utilisation=link_util,
            total_generation=total_gen,
            total_demand=total_demand,
            total_curtailment=total_curtailed,
            total_cost=self.result.total_cost,
        )

    # ---- Energy balance verification ----

    def verify_energy_balance(self, tol: float = 1e-3) -> dict:
        """
        Verify that energy balance holds at every bus, every timestep.
        Returns dict of bus names to max imbalance.
        """
        imbalances = {}
        for bus in self.system._buses:
            bus_imbalance = np.zeros(self._T)
            for t in range(self._T):
                supply = 0.0
                # Generators on this bus
                for gen in self.system._generators:
                    if gen.bus is bus and gen.name in self.result.generator_dispatch:
                        supply += self.result.generator_dispatch[gen.name][t]
                # Storage discharge
                for sto in self.system._storages:
                    if sto.bus is bus and sto.name in self.result.storage_discharge:
                        supply += self.result.storage_discharge[sto.name][t]
                # Link imports
                for link in self.system._links:
                    if link.bus_to is bus and link.name in self.result.link_flow:
                        supply += self.result.link_flow[link.name][t] * link.efficiency

                demand = 0.0
                # Loads
                for load in self.system._loads:
                    if load.bus is bus:
                        d = load.amount
                        if isinstance(d, np.ndarray):
                            demand += float(d[t])
                        else:
                            demand += float(d)
                # Storage charge
                for sto in self.system._storages:
                    if sto.bus is bus and sto.name in self.result.storage_charge:
                        demand += self.result.storage_charge[sto.name][t]
                # Link exports
                for link in self.system._links:
                    if link.bus_from is bus and link.name in self.result.link_flow:
                        demand += self.result.link_flow[link.name][t]

                bus_imbalance[t] = supply - demand
            imbalances[bus.name] = float(np.abs(bus_imbalance).max())

        return imbalances

    # ---- Why infeasible ----

    def why_infeasible(self) -> str:
        """
        Provide a plain-English explanation of infeasibility.
        Only meaningful if result.status == 'infeasible'.
        """
        if self.result.status != "infeasible":
            return f"Problem is {self.result.status}, not infeasible. No explanation needed."

        # Check total supply capacity vs max demand
        max_demand = 0.0
        for load in self.system._loads:
            d = load.amount
            if isinstance(d, np.ndarray):
                max_demand = max(max_demand, float(d.max()))
            else:
                max_demand = max(max_demand, float(d))

        total_capacity = sum(g.capacity for g in self.system._generators)

        lines = ["Infeasibility Analysis:"]
        lines.append(f"  Max instantaneous demand: {max_demand:.1f} MW")
        lines.append(f"  Total generator capacity: {total_capacity:.1f} MW")
        if total_capacity < max_demand:
            lines.append(f"  ⚠ Capacity shortage of {max_demand - total_capacity:.1f} MW.")
            lines.append(f"  Suggest: add more generation capacity or allow extendable gens.")

        # Check emission cap
        if self.system._emission_limit is not None:
            min_em = 0.0  # lower bound based on clean gens
            lines.append(f"  Emission cap: {self.system._emission_limit} tCO2")
            clean_cap = sum(
                g.capacity for g in self.system._generators
                if g.emission_factor < 0.01)
            if clean_cap < max_demand * 0.5:  # heuristic
                lines.append(f"  ⚠ Emission cap may be too tight for available clean capacity "
                             f"({clean_cap:.1f} MW).")

        # Check bus isolation
        connected_buses = set()
        for link in self.system._links:
            connected_buses.add(link.bus_from.name)
            connected_buses.add(link.bus_to.name)
        isolated = [b.name for b in self.system._buses
                   if b.name not in connected_buses
                   and not any(g.bus is b for g in self.system._generators)
                   and any(l.bus is b for l in self.system._loads)]
        if isolated:
            lines.append(f"  ⚠ Buses with demand but no generation/links: {isolated}")

        return "\n".join(lines)

    # ---- Master summary ----

    def summary(self) -> str:
        """Human-readable summary of the optimisation result."""
        ds = self.dispatch_summary()
        lines = [
            "=" * 60,
            f"Optimisation Result: {self.system.name}",
            "=" * 60,
            f"Status:        {self.result.status}",
            f"Total cost:    ${self.result.total_cost:,.2f}",
            f"Solve time:    {self.result.solve_time:.3f}s",
            f"Horizon:       {self._T} timesteps × {self._dt}h",
            "",
            "Generation Mix:",
        ]
        for name, total in sorted(ds.generator_total.items(), key=lambda x: -x[1]):
            cf = ds.generator_capacity_factor[name]
            share = 100 * total / ds.total_generation if ds.total_generation > 1e-6 else 0
            lines.append(
                f"  {name:25s}  {total:>10.1f} MWh  "
                f"({share:5.1f}% of gen, CF={100*cf:5.1f}%)"
            )

        if ds.storage_cycles:
            lines.append("")
            lines.append("Storage:")
            for name, cycles in ds.storage_cycles.items():
                lines.append(f"  {name:25s}  {cycles:.1f} equivalent full cycles")

        if self.result.capacity_additions:
            lines.append("")
            lines.append("Investment Decisions:")
            for name, cap in sorted(self.result.capacity_additions.items()):
                if cap > 0.01:
                    lines.append(f"  {name:25s}  {cap:8.1f}")

        if ds.total_curtailment > 1e-3:
            lines.append("")
            lines.append(f"Total curtailment: {ds.total_curtailment:.1f} MWh")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting (optional — requires plotly)
# ---------------------------------------------------------------------------

def plot_dispatch(system: "EnergySystem", result: "OptimisationResult",
                  bus: Optional[str] = None) -> object:
    """
    Create an interactive stacked-area dispatch plot via Plotly.

    Args:
        system: the EnergySystem
        result: optimisation result
        bus: if provided, only plot generators on this bus

    Returns:
        A plotly Figure object (user calls .show() or .write_html()).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotting requires plotly. Install with: pip install plotly")

    T = system._timesteps
    hours = np.arange(T) * system._dt

    fig = go.Figure()

    # Generators (stacked positive)
    for gen in system._generators:
        if bus is not None and gen.bus.name != bus:
            continue
        if gen.name not in result.generator_dispatch:
            continue
        fig.add_trace(go.Scatter(
            x=hours, y=result.generator_dispatch[gen.name],
            name=gen.name, mode="lines", stackgroup="supply",
            fill="tonexty",
        ))

    # Storage discharge (stacked positive)
    for sto in system._storages:
        if bus is not None and sto.bus.name != bus:
            continue
        if sto.name not in result.storage_discharge:
            continue
        fig.add_trace(go.Scatter(
            x=hours, y=result.storage_discharge[sto.name],
            name=f"{sto.name} discharge", mode="lines",
            stackgroup="supply", fill="tonexty",
        ))

    # Storage charge (negative)
    for sto in system._storages:
        if bus is not None and sto.bus.name != bus:
            continue
        if sto.name not in result.storage_charge:
            continue
        fig.add_trace(go.Scatter(
            x=hours, y=-result.storage_charge[sto.name],
            name=f"{sto.name} charge", mode="lines",
            stackgroup="demand", fill="tonexty",
        ))

    # Demand (line)
    for load in system._loads:
        if bus is not None and load.bus.name != bus:
            continue
        amount = load.amount
        if not isinstance(amount, np.ndarray):
            amount = np.full(T, amount)
        fig.add_trace(go.Scatter(
            x=hours, y=amount, name=f"{load.name} (demand)",
            line=dict(color="black", width=2, dash="dash"),
        ))

    fig.update_layout(
        title=f"Dispatch — {system.name}" + (f" ({bus})" if bus else ""),
        xaxis_title="Hour",
        yaxis_title="Power [MW]",
        template="plotly_dark",
    )
    return fig


def plot_duration_curve(series: np.ndarray, name: str = "") -> object:
    """Create a duration curve (sorted descending)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotting requires plotly.")

    sorted_desc = np.sort(series)[::-1]
    percentile = np.arange(len(sorted_desc)) / len(sorted_desc) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentile, y=sorted_desc, mode="lines", name=name))
    fig.update_layout(
        title=f"Duration Curve — {name}",
        xaxis_title="Percentile [%]",
        yaxis_title="Value",
        template="plotly_dark",
    )
    return fig


def plot_energy_sankey(system: "EnergySystem",
                       result: "OptimisationResult") -> object:
    """
    Energy flow Sankey diagram across the whole horizon.
    Aggregates over time.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotting requires plotly.")

    # Build node list: generators, buses, storage, links, loads
    nodes = []
    node_idx = {}

    def _add_node(name: str) -> int:
        if name not in node_idx:
            node_idx[name] = len(nodes)
            nodes.append(name)
        return node_idx[name]

    # Add all buses
    for bus in system._buses:
        _add_node(f"bus:{bus.name}")

    sources, targets, values, labels = [], [], [], []

    # Generator → bus
    for gen in system._generators:
        if gen.name not in result.generator_dispatch:
            continue
        total = result.generator_dispatch[gen.name].sum() * system._dt
        if total < 1e-3:
            continue
        src = _add_node(f"gen:{gen.name}")
        tgt = node_idx[f"bus:{gen.bus.name}"]
        sources.append(src); targets.append(tgt); values.append(total)
        labels.append(f"{total:.0f} MWh")

    # Bus → link → bus
    for link in system._links:
        if link.name not in result.link_flow:
            continue
        total = result.link_flow[link.name].sum() * system._dt
        if total < 1e-3:
            continue
        mid = _add_node(f"link:{link.name}")
        src = node_idx[f"bus:{link.bus_from.name}"]
        tgt = node_idx[f"bus:{link.bus_to.name}"]
        sources.append(src); targets.append(mid); values.append(total)
        labels.append(f"{total:.0f} MWh")
        sources.append(mid); targets.append(tgt); values.append(total * link.efficiency)
        labels.append(f"{total * link.efficiency:.0f} MWh")

    # Bus → load
    for load in system._loads:
        amount = load.amount
        if isinstance(amount, np.ndarray):
            total = amount.sum() * system._dt
        else:
            total = float(amount) * system._T * system._dt
        if total < 1e-3:
            continue
        src = node_idx[f"bus:{load.bus.name}"]
        tgt = _add_node(f"load:{load.name}")
        sources.append(src); targets.append(tgt); values.append(total)
        labels.append(f"{total:.0f} MWh")

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=15, thickness=15),
        link=dict(source=sources, target=targets, value=values, label=labels),
    ))
    fig.update_layout(
        title=f"Energy Flow Sankey — {system.name}",
        template="plotly_dark",
    )
    return fig
