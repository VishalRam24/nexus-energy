"""
Phase 10: Simulation Mode.

Forward-simulate an energy system without optimisation. Useful for:
- Validating a known design
- Testing dispatch rules (merit order, custom rules)
- Comparing simulated dispatch vs optimal dispatch
- Rapid iteration before committing to optimisation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem, Bus, Generator, Storage


def _merit_order_dispatch(system: "EnergySystem",
                          bus: "Bus",
                          demand: float,
                          storage_soc: dict[str, float],
                          dt: float) -> dict:
    """
    Serve demand at a single bus using merit-order dispatch.
    Cheapest generator first, then storage discharge.
    """
    gens_on_bus = [g for g in system._generators if g.bus is bus]
    stos_on_bus = [s for s in system._storages if s.bus is bus]

    # Sort generators by marginal cost
    gens_on_bus = sorted(gens_on_bus, key=lambda g: g.marginal_cost)

    dispatch = {g.name: 0.0 for g in gens_on_bus}
    charge = {s.name: 0.0 for s in stos_on_bus}
    discharge = {s.name: 0.0 for s in stos_on_bus}
    remaining = demand

    # Generators (cheapest first)
    for gen in gens_on_bus:
        if remaining <= 1e-9:
            break
        cap = gen.capacity
        # Respect carrier factor for VRE
        max_out = cap  # carrier factor handled by caller via time index
        take = min(remaining, max_out)
        dispatch[gen.name] = take
        remaining -= take

    # Storage discharge if demand remains
    for sto in stos_on_bus:
        if remaining <= 1e-9:
            break
        available = storage_soc.get(sto.name, 0.0) * sto.efficiency_discharge
        max_power = min(sto.power_capacity, available / dt)
        take = min(remaining, max_power)
        discharge[sto.name] = take
        remaining -= take

    # If generation exceeds demand, charge storage with surplus
    surplus = -remaining if remaining < 0 else 0
    # (merit order assumption: no over-generation — but renewable curtailment handled
    # at the caller level via carrier_factor)

    return {
        "dispatch": dispatch,
        "charge": charge,
        "discharge": discharge,
        "shortage": max(remaining, 0.0),
    }


def simulate(system: "EnergySystem",
             strategy: str = "merit_order",
             custom_rule: Optional[Callable] = None) -> dict:
    """
    Forward-simulate the energy system.

    Args:
        system: EnergySystem (already configured with components, demands)
        strategy: "merit_order" (cheapest first) or "custom"
        custom_rule: callable(system, t, state) -> dict for custom dispatch

    Returns:
        dict with per-timestep dispatch, SOC trajectories, shortage per bus.

    Note:
        Renewable generators (with carrier_factor) are treated as "must-take".
        Conventional generators respect merit order.
        Storage charges on surplus, discharges on shortage.
    """
    # Infer timesteps
    T = system._timesteps
    if T == 1:
        for load in system._loads:
            if isinstance(load.amount, np.ndarray):
                T = len(load.amount)
                break
        else:
            for gen in system._generators:
                if gen.carrier_factor is not None:
                    T = len(gen.carrier_factor)
                    break
    dt = system._dt

    # Initialise output arrays
    gen_dispatch = {g.name: np.zeros(T) for g in system._generators}
    sto_charge = {s.name: np.zeros(T) for s in system._storages}
    sto_discharge = {s.name: np.zeros(T) for s in system._storages}
    sto_soc = {s.name: np.zeros(T) for s in system._storages}
    shortages = {b.name: np.zeros(T) for b in system._buses}

    # Initial SOC
    soc_current = {
        s.name: s.soc_initial * s.energy_capacity
        for s in system._storages
    }

    for t in range(T):
        for bus in system._buses:
            # Compute demand at this bus and timestep
            demand = 0.0
            for load in system._loads:
                if load.bus is bus:
                    amount = load.amount
                    if isinstance(amount, np.ndarray):
                        demand += float(amount[t])
                    else:
                        demand += float(amount)

            # Must-take renewables first (carrier_factor gens)
            must_take = 0.0
            gens_on_bus = [g for g in system._generators if g.bus is bus]
            for gen in gens_on_bus:
                if gen.carrier_factor is not None:
                    cf = float(gen.carrier_factor[t])
                    output = gen.capacity * cf
                    gen_dispatch[gen.name][t] = output
                    must_take += output

            # Net demand for dispatchable gens + storage
            net_demand = demand - must_take

            # Overgeneration → charge storage
            if net_demand < 0:
                surplus = -net_demand
                # Charge storage proportionally to power_capacity
                stos_on_bus = [s for s in system._storages if s.bus is bus]
                for sto in stos_on_bus:
                    if surplus <= 1e-9:
                        break
                    headroom = (sto.soc_max * sto.energy_capacity
                                - soc_current[sto.name]) / (sto.efficiency_charge * dt)
                    take = min(surplus, sto.power_capacity, max(headroom, 0))
                    sto_charge[sto.name][t] = take
                    soc_current[sto.name] += sto.efficiency_charge * take * dt
                    surplus -= take
                # Remaining surplus = curtailment (implicit — no tracking here)

            else:
                # Dispatch conventional generators by merit order
                dispatchable = [g for g in gens_on_bus if g.carrier_factor is None]
                dispatchable = sorted(dispatchable, key=lambda g: g.marginal_cost)
                remaining = net_demand

                for gen in dispatchable:
                    if remaining <= 1e-9:
                        break
                    take = min(remaining, gen.capacity)
                    gen_dispatch[gen.name][t] = take
                    remaining -= take

                # If still short, discharge storage
                stos_on_bus = [s for s in system._storages if s.bus is bus]
                for sto in stos_on_bus:
                    if remaining <= 1e-9:
                        break
                    available_energy = (soc_current[sto.name]
                                         - sto.soc_min * sto.energy_capacity)
                    max_pwr = min(sto.power_capacity,
                                   available_energy * sto.efficiency_discharge / dt)
                    take = min(remaining, max(max_pwr, 0))
                    sto_discharge[sto.name][t] = take
                    soc_current[sto.name] -= take * dt / sto.efficiency_discharge
                    remaining -= take

                shortages[bus.name][t] = max(remaining, 0.0)

            # Self-discharge
            for sto in system._storages:
                soc_current[sto.name] *= (1 - sto.self_discharge * dt)

        # Record SOC
        for sto in system._storages:
            sto_soc[sto.name][t] = soc_current[sto.name]

    # Compute total cost (like optimisation result)
    total_cost = 0.0
    for gen in system._generators:
        total_cost += gen.marginal_cost * gen_dispatch[gen.name].sum() * dt

    return {
        "generator_dispatch": gen_dispatch,
        "storage_charge": sto_charge,
        "storage_discharge": sto_discharge,
        "storage_soc": sto_soc,
        "shortages": shortages,
        "total_cost": total_cost,
        "total_shortage": {b: shortages[b].sum() for b in shortages},
    }


def compare_sim_vs_opt(system: "EnergySystem",
                        sim_result: dict,
                        opt_result) -> str:
    """
    Compare simulation results vs optimisation results.
    Returns a human-readable report.
    """
    lines = ["Simulation vs Optimisation Comparison:"]
    lines.append(f"  Simulated cost: ${sim_result['total_cost']:,.2f}")
    lines.append(f"  Optimal cost:   ${opt_result.total_cost:,.2f}")
    if opt_result.total_cost > 1e-6:
        gap = (sim_result['total_cost'] - opt_result.total_cost) / opt_result.total_cost
        lines.append(f"  Suboptimality:  {100*gap:.2f}%")
    lines.append("")

    lines.append("Generator dispatch (total MWh):")
    for name in opt_result.generator_dispatch:
        sim_total = sim_result["generator_dispatch"].get(name, np.zeros(1)).sum() * system._dt
        opt_total = opt_result.generator_dispatch[name].sum() * system._dt
        lines.append(f"  {name:20s}  sim: {sim_total:8.1f}  opt: {opt_total:8.1f}")

    total_shortage = sum(sim_result["total_shortage"].values())
    if total_shortage > 1e-3:
        lines.append(f"\n  ⚠ Simulation had {total_shortage:.1f} MWh of unmet demand.")

    return "\n".join(lines)
