"""
Phase 6: Sector Coupling & Multi-Carrier Networks.

Provides pre-built sector coupling patterns and convenience functions
for creating multi-carrier energy systems.
"""

from __future__ import annotations

import numpy as np

from nexus_energy.core import EnergySystem, Bus
from nexus_energy.components import add_component


def create_power_to_hydrogen(
    system: EnergySystem,
    elec_bus: Bus,
    name_prefix: str = "p2h",
    electrolyser_type: str = "EC008",
    electrolyser_capacity: float = 50.0,  # MW
    h2_storage_capacity: float = 0.0,  # MW (0 = no storage)
    h2_storage_duration: float = 24.0,  # hours
    fuel_cell_capacity: float = 0.0,  # MW (0 = no fuel cell)
    fuel_cell_type: str = "EC001",
    extendable: bool = False,
    max_capacity: float = float("inf"),
) -> dict[str, object]:
    """
    Create a Power-to-Hydrogen chain: electricity → electrolyser → H2 → (storage) → (fuel cell).

    Returns dict of created components: {"h2_bus", "electrolyser", "h2_storage", "fuel_cell"}.
    """
    h2_bus = system.add_bus(f"{name_prefix}_h2", carrier="hydrogen")
    components = {"h2_bus": h2_bus}

    # Electrolyser: electricity → hydrogen
    elz = add_component(system, f"{name_prefix}_elz", electrolyser_type,
                        bus=elec_bus, bus_to=h2_bus,
                        capacity=electrolyser_capacity,
                        extendable=extendable, max_capacity=max_capacity)
    components["electrolyser"] = elz

    # H2 Storage (optional)
    if h2_storage_capacity > 0:
        h2_sto = add_component(system, f"{name_prefix}_h2_store", "EC012",
                               bus=h2_bus, capacity=h2_storage_capacity,
                               energy_to_power_ratio=h2_storage_duration,
                               extendable=extendable)
        components["h2_storage"] = h2_sto

    # Fuel cell: hydrogen → electricity (optional)
    if fuel_cell_capacity > 0:
        fc = add_component(system, f"{name_prefix}_fc", fuel_cell_type,
                           bus=h2_bus, bus_to=elec_bus,
                           capacity=fuel_cell_capacity,
                           extendable=extendable)
        components["fuel_cell"] = fc

    return components


def create_heat_system(
    system: EnergySystem,
    elec_bus: Bus,
    name_prefix: str = "heat",
    heat_pump_capacity: float = 20.0,
    heat_pump_type: str = "EC068",
    gas_boiler_capacity: float = 0.0,
    tes_capacity: float = 0.0,
    tes_duration: float = 6.0,
    extendable: bool = False,
) -> dict[str, object]:
    """
    Create a heat supply system: electricity/gas → heat pumps/boilers → heat bus → (TES).

    Returns dict of created components.
    """
    heat_bus = system.add_bus(f"{name_prefix}_bus", carrier="heat")
    components = {"heat_bus": heat_bus}

    # Heat pump: electricity → heat
    if heat_pump_capacity > 0:
        hp = add_component(system, f"{name_prefix}_hp", heat_pump_type,
                           bus=elec_bus, bus_to=heat_bus,
                           capacity=heat_pump_capacity,
                           extendable=extendable)
        components["heat_pump"] = hp

    # Gas boiler: gas → heat (optional)
    if gas_boiler_capacity > 0:
        gas_bus = system.add_bus(f"{name_prefix}_gas", carrier="natural_gas")
        # Infinite gas supply (generator on gas bus)
        system.add_generator(f"{name_prefix}_gas_supply", bus=gas_bus,
                             capacity=gas_boiler_capacity * 2,
                             marginal_cost=30)
        boiler = add_component(system, f"{name_prefix}_boiler", "EC085",
                               bus=gas_bus, bus_to=heat_bus,
                               capacity=gas_boiler_capacity,
                               extendable=extendable)
        components["gas_bus"] = gas_bus
        components["gas_boiler"] = boiler

    # Thermal energy storage (optional)
    if tes_capacity > 0:
        tes = add_component(system, f"{name_prefix}_tes", "EC078",
                            bus=heat_bus, capacity=tes_capacity,
                            energy_to_power_ratio=tes_duration,
                            extendable=extendable)
        components["tes"] = tes

    return components


def create_temperature_heat_network(
    system: EnergySystem,
    elec_bus: Bus,
    name_prefix: str = "dh",
    hot_carrier: str = "heat_high",
    cold_carrier: str = "heat_low",
    hx_capacity: float = 100.0,           # MW heat downgradable high→low
    hx_efficiency: float = 0.98,          # heat-exchanger thermal efficiency
    booster_cop: float = 0.0,             # >0 adds a low→high heat-pump booster (COP)
    booster_capacity: float = 0.0,        # MW_thermal output of the booster
    extendable: bool = False,
) -> dict[str, object]:
    """Temperature-tiered heat network (Phase 4.6).

    District-heating and industrial heat split into a *high* and a *low*
    temperature tier on separate buses. High-grade heat can always cascade
    down to the low tier through a heat exchanger (one-directional, with a
    small thermal loss); upgrading low→high requires work, modelled as an
    electric *booster heat pump* whose thermal output is ``COP × electricity``.
    This packages the otherwise-verbose ``heat_high`` / ``heat_low`` + HX-Link
    pattern (Calliope / oemof multi-temperature heat, SpineOpt commodity
    tiers).

    Returns dict with ``{hot_bus, cold_bus, heat_exchanger[, booster]}``.

    Args:
        elec_bus: electricity bus feeding the optional booster heat pump.
        hx_capacity / hx_efficiency: rating and thermal efficiency of the
            high→low cascade exchanger.
        booster_cop / booster_capacity: if both > 0, add a low→high booster
            heat pump consuming electricity (``thermal_out = COP × elec``).
    """
    # Register the temperature-tier carriers if the system doesn't know them.
    for c in (hot_carrier, cold_carrier):
        if c not in system._carriers:
            system.add_carrier(c)
    hot_bus = system.add_bus(f"{name_prefix}_{hot_carrier}", carrier=hot_carrier)
    cold_bus = system.add_bus(f"{name_prefix}_{cold_carrier}", carrier=cold_carrier)
    components = {"hot_bus": hot_bus, "cold_bus": cold_bus}

    # Heat exchanger: high-temperature heat cascades down to low temperature.
    hx = system.add_link(
        f"{name_prefix}_hx", bus_from=hot_bus, bus_to=cold_bus,
        capacity=hx_capacity, efficiency=hx_efficiency,
        extendable=extendable,
        max_capacity=(float("inf") if extendable else hx_capacity),
    )
    components["heat_exchanger"] = hx

    # Optional electric booster heat pump: low → high (consumes work).
    # Two inputs (low-grade heat + electricity) are not co-fired in a single
    # Link; the standard reduced form drives the high-grade output purely from
    # electricity at the heat-pump COP, which is the dominant cost term.
    if booster_cop > 0 and booster_capacity > 0:
        booster = system.add_link(
            f"{name_prefix}_booster", bus_from=elec_bus, bus_to=hot_bus,
            capacity=booster_capacity / booster_cop,  # electric input rating
            efficiency=booster_cop,
            extendable=extendable,
            max_capacity=(float("inf") if extendable
                          else booster_capacity / booster_cop),
        )
        components["booster"] = booster

    return components


def create_power_to_gas(
    system: EnergySystem,
    elec_bus: Bus,
    name_prefix: str = "p2g",
    electrolyser_capacity: float = 50.0,
    methanation_capacity: float = 30.0,
    gas_storage_capacity: float = 0.0,
    extendable: bool = False,
) -> dict[str, object]:
    """
    Create a Power-to-Gas chain:
    electricity → electrolyser → H2 → methanation → synthetic gas.

    Returns dict of created components.
    """
    h2_bus = system.add_bus(f"{name_prefix}_h2", carrier="hydrogen")
    gas_bus = system.add_bus(f"{name_prefix}_gas", carrier="natural_gas")
    components = {"h2_bus": h2_bus, "gas_bus": gas_bus}

    # Electrolyser
    elz = add_component(system, f"{name_prefix}_elz", "EC008",
                        bus=elec_bus, bus_to=h2_bus,
                        capacity=electrolyser_capacity,
                        extendable=extendable)
    components["electrolyser"] = elz

    # Methanation: H2 → synthetic natural gas
    meth = add_component(system, f"{name_prefix}_meth", "EC193",
                         bus=h2_bus, bus_to=gas_bus,
                         capacity=methanation_capacity,
                         extendable=extendable)
    components["methanation"] = meth

    return components


def create_multi_carrier_system(
    name: str = "multi_carrier",
    carriers: list[str] | None = None,
) -> tuple[EnergySystem, dict[str, Bus]]:
    """
    Create an EnergySystem pre-configured with multiple carrier buses.

    Args:
        name: system name
        carriers: list of carrier names. Default: ["electricity", "heat", "hydrogen"]

    Returns:
        (system, buses_dict) where buses_dict maps carrier name to Bus.
    """
    if carriers is None:
        carriers = ["electricity", "heat", "hydrogen"]

    system = EnergySystem(name)
    buses = {}
    for carrier in carriers:
        buses[carrier] = system.add_bus(f"{carrier}_bus", carrier=carrier)

    return system, buses
