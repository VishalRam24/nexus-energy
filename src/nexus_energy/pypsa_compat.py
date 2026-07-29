"""
Phase 13: PyPSA Compatibility Layer.

Import and export PyPSA Network objects. Enables:
- Running PyPSA test cases on nexus-energy for head-to-head benchmarks
- Gradual migration of existing PyPSA workflows to nexus-energy

Mapping of PyPSA concepts to nexus-energy:
- Network → EnergySystem
- Bus → Bus (with carrier name)
- Generator → Generator
- StorageUnit → Storage
- Store → Storage (variant with energy-only)
- Load → Load
- Link → Link
- Line → Link (with efficiency = 1 - losses)
- Carrier → Carrier
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem


def from_pypsa(network, name: Optional[str] = None,
               line_model: str = "auto"):
    """
    Convert a PyPSA Network to a nexus-energy EnergySystem.

    Args:
        network: a pypsa.Network instance
        name: optional name for the EnergySystem
        line_model: how to translate ``network.lines``:
            * ``"auto"`` (default) — Lines with reactance ``x > 0`` are
              imported as DC-OPF links (``model_type='dc_opf'``); lines
              with ``x = 0`` fall back to lossless transport (the
              pre-Phase-3 behaviour).
            * ``"transport"`` — force every Line to transport, ignoring
              reactance. Use to reproduce the legacy obj-diff numbers.
            * ``"dc_opf"`` — force every Line to DC-OPF; lines with no
              reactance get a small default (0.001 p.u.).

    Returns:
        EnergySystem ready to optimise.

    Notes:
        - Snapshots are used to infer timesteps
        - Time-varying data in network.<component>_t is mapped to arrays
        - Only supports: buses, generators, storage_units, stores, loads, links, lines
        - DC-OPF auto-routing closes the KVL gap reported in
          ``test_projects/.../FLAGSHIP_COMPARISON.md`` (PyPSA-Earth /
          PyPSA-Eur tutorials).
    """
    from nexus_energy.core import EnergySystem

    sys_name = name or getattr(network, "name", "pypsa_import") or "pypsa_import"
    system = EnergySystem(sys_name)

    # Make sure derived per-unit quantities (``x_pu_eff`` etc.) exist as
    # columns. Newer PyPSA versions only populate them lazily; calling
    # ``calculate_dependent_values`` is a no-op if they're already set.
    if line_model in ("auto", "dc_opf") and hasattr(network, "calculate_dependent_values"):
        try:
            network.calculate_dependent_values()
        except Exception:
            pass  # don't block import if PyPSA's pre-compute trips on edge data

    # ---- Timesteps from snapshots ----
    snapshots = getattr(network, "snapshots", None)
    if snapshots is not None and len(snapshots) > 1:
        T = len(snapshots)
        # Timestep weight (for non-uniform snapshots)
        weights = getattr(network, "snapshot_weightings", None)
        if weights is not None and hasattr(weights, "generators"):
            dt = float(weights["generators"].iloc[0])
        else:
            dt = 1.0
        system.set_timesteps(T, dt=dt)
    else:
        T = 1
        dt = 1.0

    # ---- Carriers ----
    if hasattr(network, "carriers") and len(network.carriers) > 0:
        for carrier_name in network.carriers.index:
            if carrier_name and carrier_name not in system._carriers:
                unit = "MWh"  # PyPSA default
                system.add_carrier(carrier_name, unit=unit)

    # ---- Buses ----
    bus_map = {}
    for bus_name, bus_row in network.buses.iterrows():
        carrier = bus_row.get("carrier", "electricity") or "electricity"
        # Map common aliases
        if carrier == "AC" or carrier == "DC":
            carrier = "electricity"
        if carrier not in system._carriers:
            system.add_carrier(carrier, unit="MWh")
        bus = system.add_bus(str(bus_name), carrier=carrier)
        bus_map[bus_name] = bus

    # ---- Helper for time-varying attributes ----
    def _time_series(df_static, df_t, name, attr, default, T):
        """Get attribute as array of length T."""
        if hasattr(df_t, attr) and name in df_t[attr].columns:
            series = df_t[attr][name]
            vals = np.asarray(series.values, dtype=float)
            if len(vals) == T:
                return vals
        val = df_static.get(attr, default)
        if val is None:
            val = default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # ---- Generators ----
    for gen_name, gen_row in network.generators.iterrows():
        bus = bus_map.get(gen_row["bus"])
        if bus is None:
            continue
        p_nom = float(gen_row.get("p_nom", 0.0) or 0.0)
        p_nom_extendable = bool(gen_row.get("p_nom_extendable", False))
        marginal_cost = float(gen_row.get("marginal_cost", 0.0) or 0.0)
        capital_cost = float(gen_row.get("capital_cost", 0.0) or 0.0)
        efficiency = float(gen_row.get("efficiency", 1.0) or 1.0)

        # Time-varying p_max_pu (capacity factor)
        cf = None
        gens_t = getattr(network, "generators_t", None)
        if gens_t is not None and hasattr(gens_t, "p_max_pu"):
            if gen_name in gens_t.p_max_pu.columns:
                cf_series = gens_t.p_max_pu[gen_name]
                cf = np.asarray(cf_series.values, dtype=float)
        # Static (scalar) availability derating. PyPSA stores constant
        # de-ratings — e.g. nuclear p_max_pu=0.781 — as a scalar column on
        # ``network.generators`` (no time-series). Without mapping it,
        # dispatchable units would run at full nameplate and over-dispatch
        # cheap baseload, displacing pricier peakers and under-pricing the
        # system. Synthesize a constant carrier_factor so the cap
        # ``p[t] <= capacity · p_max_pu`` is enforced exactly as in PyPSA.
        if cf is None:
            p_max_pu_static = float(gen_row.get("p_max_pu", 1.0) or 1.0)
            if 0.0 <= p_max_pu_static < 1.0:
                cf = np.full(T, p_max_pu_static, dtype=float)

        # Emission factor from carrier
        emission = 0.0
        if "carrier" in gen_row and hasattr(network, "carriers"):
            cname = gen_row["carrier"]
            if cname in network.carriers.index:
                emission = float(network.carriers.loc[cname].get("co2_emissions", 0.0) or 0.0)
                # Apply efficiency: tCO2/MWh_electric = emission_factor_fuel / efficiency
                if efficiency > 1e-6:
                    emission = emission / efficiency

        capacity_for_nexus = p_nom if not p_nom_extendable else 0.0
        max_cap = float(gen_row.get("p_nom_max", float("inf")) or float("inf"))
        if max_cap == 0 or np.isnan(max_cap):
            max_cap = float("inf")
        min_cap = float(gen_row.get("p_nom_min", 0.0) or 0.0)
        if np.isnan(min_cap) or min_cap < 0:
            min_cap = 0.0

        gen_kwargs = dict(
            bus=bus,
            capacity=capacity_for_nexus,
            marginal_cost=marginal_cost,
            capital_cost=capital_cost,
            carrier_factor=cf,
            emission_factor=emission,
            extendable=p_nom_extendable,
            max_capacity=max_cap,
            min_capacity=min_cap,
        )

        if bool(gen_row.get("committable", False)):
            p_min_pu = float(gen_row.get("p_min_pu", 0.0) or 0.0)
            up_before = int(gen_row.get("up_time_before", 1) or 0)
            down_before = int(gen_row.get("down_time_before", 0) or 0)
            gen_kwargs.update(
                committable=True,
                min_up_time=int(gen_row.get("min_up_time", 0) or 0),
                min_down_time=int(gen_row.get("min_down_time", 0) or 0),
                startup_cost=float(gen_row.get("start_up_cost", 0.0) or 0.0),
                shutdown_cost=float(gen_row.get("shut_down_cost", 0.0) or 0.0),
                p_min=p_nom * p_min_pu,
                initial_status=1 if up_before > 0 else 0,
                up_time_before=up_before,
                down_time_before=down_before,
            )

        system.add_generator(str(gen_name), **gen_kwargs)

    # ---- Loads ----
    for load_name, load_row in network.loads.iterrows():
        bus = bus_map.get(load_row["bus"])
        if bus is None:
            continue
        # Time-varying p_set
        amount = 0.0
        loads_t = getattr(network, "loads_t", None)
        if loads_t is not None and hasattr(loads_t, "p_set"):
            if load_name in loads_t.p_set.columns:
                amount = np.asarray(loads_t.p_set[load_name].values, dtype=float)
        if isinstance(amount, float):
            amount = float(load_row.get("p_set", 0.0) or 0.0)
        system.add_load(str(load_name), bus=bus, amount=amount)

    # ---- Storage Units (power + energy capacity) ----
    if hasattr(network, "storage_units"):
        for sto_name, sto_row in network.storage_units.iterrows():
            bus = bus_map.get(sto_row["bus"])
            if bus is None:
                continue
            p_nom = float(sto_row.get("p_nom", 0.0) or 0.0)
            p_nom_extendable = bool(sto_row.get("p_nom_extendable", False))
            max_hours = float(sto_row.get("max_hours", 6.0) or 6.0)
            energy_cap = p_nom * max_hours
            eta_ch = float(sto_row.get("efficiency_store", 0.9) or 0.9)
            eta_dis = float(sto_row.get("efficiency_dispatch", 0.9) or 0.9)
            soc_init = float(sto_row.get("state_of_charge_initial", 0.5 * energy_cap) or 0)
            soc_init_frac = soc_init / energy_cap if energy_cap > 1e-6 else 0.5
            cyclic = bool(sto_row.get("cyclic_state_of_charge", True))
            capital_cost = float(sto_row.get("capital_cost", 0.0) or 0.0)
            marginal_cost = float(sto_row.get("marginal_cost", 0.0) or 0.0)
            p_nom_max = float(sto_row.get("p_nom_max", float("inf")) or float("inf"))
            if p_nom_max == 0 or np.isnan(p_nom_max):
                p_nom_max = float("inf")
            p_nom_min = float(sto_row.get("p_nom_min", 0.0) or 0.0)
            if np.isnan(p_nom_min) or p_nom_min < 0:
                p_nom_min = 0.0

            # Standing loss / self discharge mapping
            standing_loss = float(sto_row.get("standing_loss", 0.0) or 0.0)
            self_discharge = standing_loss * dt

            # Time-varying inflow mapping
            inflow = None
            if hasattr(network, "storage_units_t") and "inflow" in network.storage_units_t and sto_name in network.storage_units_t.inflow.columns:
                inflow = network.storage_units_t.inflow[sto_name].values

            # Time-varying state of charge pinning mapping
            soc_fixed = {}
            if hasattr(network, "storage_units_t") and "state_of_charge_set" in network.storage_units_t and sto_name in network.storage_units_t.state_of_charge_set.columns:
                series = network.storage_units_t.state_of_charge_set[sto_name]
                for t_idx, val in enumerate(series.values):
                    if not np.isnan(val):
                        soc_fixed[t_idx] = float(val)

            spill_cost = float(sto_row.get("spill_cost", 1e-3) or 1e-3)

            system.add_storage(
                str(sto_name), bus=bus,
                power_capacity=p_nom if not p_nom_extendable else 0,
                energy_capacity=energy_cap if not p_nom_extendable else 0,
                efficiency_charge=eta_ch,
                efficiency_discharge=eta_dis,
                self_discharge=self_discharge,
                soc_initial=min(max(soc_init_frac, 0), 1),
                cyclic=cyclic,
                cyclic_level="free",  # PyPSA cyclic_state_of_charge: free level
                marginal_cost=marginal_cost,
                capital_cost_power=capital_cost,
                capital_cost_energy=0.0,
                extendable=p_nom_extendable,
                max_power_capacity=p_nom_max,
                max_energy_capacity=p_nom_max * max_hours if p_nom_max != float("inf") else float("inf"),
                min_power_capacity=p_nom_min,
                min_energy_capacity=p_nom_min * max_hours,
                max_hours=max_hours,
                inflow=inflow,
                soc_fixed=soc_fixed if soc_fixed else None,
                spill_cost=spill_cost,
            )

    # ---- Stores (energy-only, for hydrogen/gas stores) ----
    # PyPSA Store: the Store itself has no inherent power limit — charge/discharge are
    # gated by the attached Link. We model this by giving the nexus Storage a huge
    # internal power capacity so the Link is always the binding constraint, and put
    # the full capital_cost on the energy dimension.
    if hasattr(network, "stores"):
        for store_name, store_row in network.stores.iterrows():
            bus = bus_map.get(store_row["bus"])
            if bus is None:
                continue
            e_nom = float(store_row.get("e_nom", 0.0) or 0.0)
            e_nom_extendable = bool(store_row.get("e_nom_extendable", False))
            e_nom_max = float(store_row.get("e_nom_max", float("inf")) or float("inf"))
            if e_nom_max == 0 or np.isnan(e_nom_max):
                e_nom_max = float("inf")
            e_nom_min = float(store_row.get("e_nom_min", 0.0) or 0.0)
            if np.isnan(e_nom_min) or e_nom_min < 0:
                e_nom_min = 0.0
            cyclic = bool(store_row.get("e_cyclic", True))
            capital_cost = float(store_row.get("capital_cost", 0.0) or 0.0)
            marginal_cost = float(store_row.get("marginal_cost", 0.0) or 0.0)
            e_init_frac = float(store_row.get("e_initial", 0.0) or 0) / e_nom if e_nom > 1e-6 else 0.5

            # Standing loss / self discharge mapping
            standing_loss = float(store_row.get("standing_loss", 0.0) or 0.0)
            self_discharge = standing_loss * dt

            # Time-varying inflow mapping
            inflow = None
            if hasattr(network, "stores_t") and "inflow" in network.stores_t and store_name in network.stores_t.inflow.columns:
                inflow = network.stores_t.inflow[store_name].values

            # Time-varying state of charge pinning mapping
            soc_fixed = {}
            if hasattr(network, "stores_t") and "e_set" in network.stores_t and store_name in network.stores_t.e_set.columns:
                series = network.stores_t.e_set[store_name]
                for t_idx, val in enumerate(series.values):
                    if not np.isnan(val):
                        soc_fixed[t_idx] = float(val)

            system.add_storage(
                str(store_name), bus=bus,
                power_capacity=1e12 if e_nom_extendable else max(e_nom, 1e12),
                energy_capacity=e_nom if not e_nom_extendable else 0,
                efficiency_charge=1.0,
                efficiency_discharge=1.0,
                self_discharge=self_discharge,
                soc_initial=min(max(e_init_frac, 0), 1),
                cyclic=cyclic,
                cyclic_level="free",  # PyPSA e_cyclic: free level
                marginal_cost=marginal_cost,
                capital_cost_power=0.0,
                capital_cost_energy=capital_cost if e_nom_extendable else 0.0,
                extendable=e_nom_extendable,
                max_power_capacity=1e12,
                max_energy_capacity=e_nom_max,
                min_energy_capacity=e_nom_min,
                inflow=inflow,
                soc_fixed=soc_fixed if soc_fixed else None,
                storage_model="store",
            )

    # ---- Links ----
    if hasattr(network, "links"):
        for link_name, link_row in network.links.iterrows():
            bus_from = bus_map.get(link_row["bus0"])
            bus_to = bus_map.get(link_row["bus1"])
            if bus_from is None or bus_to is None:
                continue
            p_nom = float(link_row.get("p_nom", 0.0) or 0.0)
            p_nom_extendable = bool(link_row.get("p_nom_extendable", False))
            efficiency = float(link_row.get("efficiency", 1.0) or 1.0)
            marginal_cost = float(link_row.get("marginal_cost", 0.0) or 0.0)
            capital_cost = float(link_row.get("capital_cost", 0.0) or 0.0)
            max_cap = float(link_row.get("p_nom_max", float("inf")) or float("inf"))
            if max_cap == 0 or np.isnan(max_cap):
                max_cap = float("inf")
            min_cap = float(link_row.get("p_nom_min", 0.0) or 0.0)
            if np.isnan(min_cap) or min_cap < 0:
                min_cap = 0.0

            system.add_link(
                str(link_name),
                bus_from=bus_from, bus_to=bus_to,
                capacity=p_nom if not p_nom_extendable else 0,
                efficiency=efficiency,
                marginal_cost=marginal_cost,
                capital_cost=capital_cost,
                extendable=p_nom_extendable,
                max_capacity=max_cap,
                min_capacity=min_cap,
            )

    # ---- Lines (transmission lines — simplified as lossless links) ----
    if hasattr(network, "lines"):
        for line_name, line_row in network.lines.iterrows():
            bus_from = bus_map.get(line_row["bus0"])
            bus_to = bus_map.get(line_row["bus1"])
            if bus_from is None or bus_to is None:
                continue
            s_nom = float(line_row.get("s_nom", 0.0) or 0.0)
            s_nom_extendable = bool(line_row.get("s_nom_extendable", False))
            capital_cost = float(line_row.get("capital_cost", 0.0) or 0.0)
            s_nom_max = float(line_row.get("s_nom_max", float("inf")) or float("inf"))
            if s_nom_max == 0 or np.isnan(s_nom_max):
                s_nom_max = float("inf")
            s_nom_min = float(line_row.get("s_nom_min", 0.0) or 0.0)
            if np.isnan(s_nom_min) or s_nom_min < 0:
                s_nom_min = 0.0
            # For extendable lines with pre-built s_nom, s_nom_min defaults to s_nom.
            if s_nom_extendable and s_nom_min == 0.0 and s_nom > 0.0:
                s_nom_min = s_nom

            efficiency = 1.0  # lossless transport; piecewise-linear losses not modelled

            # Pick the most-canonical reactance available. PyPSA's KVL uses
            # ``x_pu_eff`` (computed by ``calculate_dependent_values`` from
            # ``x``, ``v_nom``, and ``num_parallel``); when that's missing
            # fall back to ``x_pu``, then to ``x`` (assumed already p.u.).
            x_pu = float(
                line_row.get("x_pu_eff", 0.0)
                or line_row.get("x_pu", 0.0)
                or line_row.get("x", 0.0)
                or 0.0
            )
            if line_model == "auto":
                use_dc_opf = x_pu > 0
            elif line_model == "dc_opf":
                use_dc_opf = True
                if x_pu <= 0:
                    x_pu = 1e-3  # sensible default to keep KVL well-conditioned
            else:  # "transport"
                use_dc_opf = False

            link_kwargs = dict(
                bus_from=bus_from, bus_to=bus_to,
                capacity=s_nom if not s_nom_extendable else 0,
                efficiency=efficiency,
                capital_cost=capital_cost,
                bidirectional=True,
                extendable=s_nom_extendable,
                max_capacity=s_nom_max,
                min_capacity=s_nom_min,
            )
            if use_dc_opf:
                link_kwargs["reactance"] = x_pu
                link_kwargs["model_type"] = "dc_opf"
                # DC-OPF flow is signed and obeys KVL; don't double-add the
                # PyPSA bidirectional mutex (signed flow already covers both
                # directions). Setting bidirectional=False here only affects
                # the transport-style mutex; it does not stop the line from
                # carrying flow either way.
                link_kwargs["bidirectional"] = False
            system.add_link(f"line_{line_name}", **link_kwargs)

    # ---- Transformers (treated as lossless bidirectional Links, transport model) ----
    if hasattr(network, "transformers"):
        for xfmr_name, xfmr_row in network.transformers.iterrows():
            bus_from = bus_map.get(xfmr_row["bus0"])
            bus_to = bus_map.get(xfmr_row["bus1"])
            if bus_from is None or bus_to is None:
                continue
            s_nom = float(xfmr_row.get("s_nom", 0.0) or 0.0)
            s_nom_extendable = bool(xfmr_row.get("s_nom_extendable", False))
            capital_cost = float(xfmr_row.get("capital_cost", 0.0) or 0.0)
            s_nom_max = float(xfmr_row.get("s_nom_max", float("inf")) or float("inf"))
            if s_nom_max == 0 or np.isnan(s_nom_max):
                s_nom_max = float("inf")
            system.add_link(
                f"xfmr_{xfmr_name}",
                bus_from=bus_from, bus_to=bus_to,
                capacity=s_nom if not s_nom_extendable else 0,
                efficiency=1.0,
                capital_cost=capital_cost,
                bidirectional=True,
                extendable=s_nom_extendable,
                max_capacity=s_nom_max,
            )

    # ---- Global constraints (CO2 caps) ----
    if hasattr(network, "global_constraints"):
        for gc_name, gc_row in network.global_constraints.iterrows():
            if gc_row.get("type") == "primary_energy" and "co2" in gc_name.lower():
                limit = float(gc_row.get("constant", 0.0) or 0.0)
                if limit > 0:
                    system.set_emission_limit(limit)
                    break

    return system


def to_pypsa(system: "EnergySystem", result=None):
    """
    Convert a nexus-energy EnergySystem back to a PyPSA Network.
    Optionally populate optimisation results.

    Requires: pip install pypsa

    Args:
        system: EnergySystem to export
        result: optional OptimisationResult to populate pypsa_network.*_t

    Returns:
        pypsa.Network
    """
    try:
        import pypsa
    except ImportError:
        raise ImportError("to_pypsa requires pypsa. Install with: pip install pypsa")

    import pandas as pd

    n = pypsa.Network()
    T = system._timesteps

    if T > 1:
        # Create a dummy DatetimeIndex
        snapshots = pd.date_range("2024-01-01", periods=T, freq="h")
        n.set_snapshots(snapshots)

    # Carriers
    carriers_seen = set()
    for bus in system._buses:
        if bus.carrier.name not in carriers_seen:
            n.add("Carrier", bus.carrier.name)
            carriers_seen.add(bus.carrier.name)

    # Buses
    for bus in system._buses:
        n.add("Bus", bus.name, carrier=bus.carrier.name)

    # Generators
    for gen in system._generators:
        kwargs = dict(
            bus=gen.bus.name,
            p_nom=gen.capacity,
            marginal_cost=gen.marginal_cost,
            capital_cost=gen.capital_cost,
            p_nom_extendable=gen.extendable,
        )
        if gen.max_capacity != float("inf"):
            kwargs["p_nom_max"] = gen.max_capacity
        n.add("Generator", gen.name, **kwargs)
        if gen.carrier_factor is not None and T > 1:
            n.generators_t.p_max_pu[gen.name] = gen.carrier_factor

    # Loads
    for load in system._loads:
        if isinstance(load.amount, np.ndarray) and T > 1:
            n.add("Load", load.name, bus=load.bus.name)
            n.loads_t.p_set[load.name] = load.amount
        else:
            amt = load.amount if not isinstance(load.amount, np.ndarray) else load.amount[0]
            n.add("Load", load.name, bus=load.bus.name, p_set=float(amt))

    # Storage
    for sto in system._storages:
        if sto.power_capacity >= 1e11:
            n.add("Store", sto.name,
                  bus=sto.bus.name,
                  e_nom=sto.energy_capacity,
                  e_cyclic=sto.cyclic,
                  e_nom_extendable=sto.extendable,
                  capital_cost=sto.capital_cost_energy)
        else:
            max_hours = sto.max_hours or (sto.energy_capacity / sto.power_capacity if sto.power_capacity > 1e-6 else 6.0)
            n.add("StorageUnit", sto.name,
                  bus=sto.bus.name,
                  p_nom=sto.power_capacity,
                  max_hours=max_hours,
                  efficiency_store=sto.efficiency_charge,
                  efficiency_dispatch=sto.efficiency_discharge,
                  cyclic_state_of_charge=sto.cyclic,
                  p_nom_extendable=sto.extendable,
                  capital_cost=sto.capital_cost_power)

    # Links
    for link in system._links:
        n.add("Link", link.name,
              bus0=link.bus_from.name,
              bus1=link.bus_to.name,
              p_nom=link.capacity,
              efficiency=link.efficiency,
              marginal_cost=link.marginal_cost,
              capital_cost=link.capital_cost,
              p_nom_extendable=link.extendable)

    # Populate results if provided
    if result is not None and result.status == "optimal":
        for gen_name, dispatch in result.generator_dispatch.items():
            if T > 1:
                n.generators_t.p[gen_name] = dispatch
        for sto_name, discharge in result.storage_discharge.items():
            if T > 1:
                sto_comp = next((s for s in system._storages if s.name == sto_name), None)
                if sto_comp is not None and sto_comp.power_capacity >= 1e11:
                    if hasattr(n, "stores_t"):
                        n.stores_t.e[sto_name] = result.storage_soc.get(sto_name, np.zeros(T))
                else:
                    charge = result.storage_charge.get(sto_name, np.zeros(T))
                    n.storage_units_t.p[sto_name] = discharge - charge
                    n.storage_units_t.state_of_charge[sto_name] = (
                        result.storage_soc.get(sto_name, np.zeros(T)))

    return n
