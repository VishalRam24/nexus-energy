"""
Phase 4: Component Registry & F0 Integration.

Provides a registry of energy component templates (EC001-EC223) that can be
instantiated on an EnergySystem. Each template defines default parameters,
port types, and cost data.

F0 components use constant efficiency — no physics model needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nexus_energy.core import Bus, EnergySystem, Generator, Storage, Link


@dataclass
class ComponentTemplate:
    """Definition of an energy component type."""
    ec_id: str
    name: str
    sector: str
    category: str  # "generator", "storage", "converter", "load"
    # Default parameters
    default_capacity: float = 0.0
    default_efficiency: float = 1.0
    marginal_cost: float = 0.0
    capital_cost: float = 0.0  # $/MW/yr or $/MWh/yr
    lifetime: int = 25  # years
    emission_factor: float = 0.0  # tCO2/MWh
    # Port types
    input_carrier: str = "electricity"
    output_carrier: str = "electricity"
    # Storage-specific
    energy_to_power_ratio: float = 4.0  # MWh per MW (duration in hours)
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.95
    self_discharge_per_hour: float = 0.0
    # Operational
    p_min_fraction: float = 0.0
    ramp_up_fraction: Optional[float] = None  # fraction of capacity per timestep
    ramp_down_fraction: Optional[float] = None


# ---------------------------------------------------------------------------
# Component Database — F0 defaults for all sectors
# ---------------------------------------------------------------------------

# Sector 01: Hydrogen & Fuel Cells
_HYDROGEN = [
    ComponentTemplate("EC001", "PEM Fuel Cell", "hydrogen", "converter",
                      default_efficiency=0.50, marginal_cost=5, capital_cost=1500,
                      lifetime=15, input_carrier="hydrogen", output_carrier="electricity"),
    ComponentTemplate("EC002", "SOFC", "hydrogen", "converter",
                      default_efficiency=0.60, marginal_cost=4, capital_cost=3000,
                      lifetime=20, input_carrier="hydrogen", output_carrier="electricity"),
    ComponentTemplate("EC008", "PEM Electrolyser", "hydrogen", "converter",
                      default_efficiency=0.65, marginal_cost=2, capital_cost=800,
                      lifetime=20, input_carrier="electricity", output_carrier="hydrogen"),
    ComponentTemplate("EC009", "Alkaline Electrolyser", "hydrogen", "converter",
                      default_efficiency=0.63, marginal_cost=1.5, capital_cost=500,
                      lifetime=25, input_carrier="electricity", output_carrier="hydrogen"),
    ComponentTemplate("EC012", "Compressed H2 Storage", "hydrogen", "storage",
                      capital_cost=15, lifetime=30, energy_to_power_ratio=24,
                      efficiency_charge=0.95, efficiency_discharge=1.0,
                      input_carrier="hydrogen", output_carrier="hydrogen"),
]

# Sector 02: Batteries
_BATTERIES = [
    ComponentTemplate("EC018", "LFP Battery", "batteries", "storage",
                      capital_cost=200, lifetime=20, energy_to_power_ratio=4,
                      efficiency_charge=0.96, efficiency_discharge=0.96,
                      self_discharge_per_hour=0.00002),
    ComponentTemplate("EC019", "NMC Battery", "batteries", "storage",
                      capital_cost=250, lifetime=15, energy_to_power_ratio=4,
                      efficiency_charge=0.95, efficiency_discharge=0.95,
                      self_discharge_per_hour=0.00004),
    ComponentTemplate("EC028", "Lead-Acid Battery", "batteries", "storage",
                      capital_cost=150, lifetime=8, energy_to_power_ratio=4,
                      efficiency_charge=0.90, efficiency_discharge=0.90,
                      self_discharge_per_hour=0.0001),
    ComponentTemplate("EC031", "Sodium-Ion Battery", "batteries", "storage",
                      capital_cost=120, lifetime=15, energy_to_power_ratio=4,
                      efficiency_charge=0.94, efficiency_discharge=0.94),
    ComponentTemplate("EC035", "Iron-Air Battery", "batteries", "storage",
                      capital_cost=25, lifetime=25, energy_to_power_ratio=100,
                      efficiency_charge=0.70, efficiency_discharge=0.70),
    ComponentTemplate("EC036", "VRFB Flow Battery", "batteries", "storage",
                      capital_cost=350, lifetime=25, energy_to_power_ratio=8,
                      efficiency_charge=0.88, efficiency_discharge=0.88),
]

# Sector 03: Solar
_SOLAR = [
    ComponentTemplate("EC044", "Mono-Si PV", "solar", "generator",
                      marginal_cost=0, capital_cost=800, lifetime=30,
                      emission_factor=0),
    ComponentTemplate("EC048", "Perovskite PV", "solar", "generator",
                      marginal_cost=0, capital_cost=400, lifetime=15,
                      emission_factor=0),
]

# Sector 04: Wind
_WIND = [
    ComponentTemplate("EC062", "Onshore Wind (HAWT)", "wind", "generator",
                      marginal_cost=0, capital_cost=1200, lifetime=25,
                      emission_factor=0),
    ComponentTemplate("EC065", "Offshore Wind (Fixed)", "wind", "generator",
                      marginal_cost=0, capital_cost=2500, lifetime=25,
                      emission_factor=0),
]

# Sector 05: Thermal
_THERMAL = [
    ComponentTemplate("EC068", "Air-Source Heat Pump", "thermal", "converter",
                      default_efficiency=3.0, marginal_cost=1, capital_cost=600,
                      lifetime=20, input_carrier="electricity", output_carrier="heat"),
    ComponentTemplate("EC078", "Hot Water TES", "thermal", "storage",
                      capital_cost=30, lifetime=30, energy_to_power_ratio=6,
                      efficiency_charge=0.95, efficiency_discharge=0.95,
                      self_discharge_per_hour=0.005,
                      input_carrier="heat", output_carrier="heat"),
    ComponentTemplate("EC085", "Natural Gas Boiler", "thermal", "converter",
                      default_efficiency=0.92, marginal_cost=30, capital_cost=80,
                      lifetime=25, emission_factor=0.2,
                      input_carrier="natural_gas", output_carrier="heat"),
    ComponentTemplate("EC091", "Vapor Compression Chiller", "thermal", "converter",
                      default_efficiency=4.0, marginal_cost=2, capital_cost=400,
                      lifetime=20, input_carrier="electricity", output_carrier="heat"),
]

# Sector 06: Conventional Generation
_CONVENTIONAL = [
    ComponentTemplate("EC101", "CCGT", "conventional", "generator",
                      default_efficiency=0.58, marginal_cost=35, capital_cost=900,
                      lifetime=30, emission_factor=0.37, p_min_fraction=0.4,
                      ramp_up_fraction=0.05, ramp_down_fraction=0.05,
                      input_carrier="natural_gas"),
    ComponentTemplate("EC109", "Simple Cycle Gas Turbine", "conventional", "generator",
                      default_efficiency=0.38, marginal_cost=55, capital_cost=500,
                      lifetime=25, emission_factor=0.55, p_min_fraction=0.2,
                      ramp_up_fraction=0.20, ramp_down_fraction=0.20,
                      input_carrier="natural_gas"),
    ComponentTemplate("EC116", "Nuclear PWR", "conventional", "generator",
                      marginal_cost=10, capital_cost=5000, lifetime=60,
                      emission_factor=0, p_min_fraction=0.5,
                      ramp_up_fraction=0.01, ramp_down_fraction=0.01),
]

# Sector 07: Mechanical Storage
_MECHANICAL = [
    ComponentTemplate("EC122", "Pumped Hydro Storage", "mechanical_storage", "storage",
                      capital_cost=50, lifetime=60, energy_to_power_ratio=10,
                      efficiency_charge=0.87, efficiency_discharge=0.87),
]

# Sector 08: Hydro & Marine
_HYDRO = [
    ComponentTemplate("EC128", "Conventional Hydro Dam", "hydro", "generator",
                      marginal_cost=2, capital_cost=2000, lifetime=80, emission_factor=0),
]

# Sector 09: Biomass
_BIOMASS = [
    ComponentTemplate("EC140", "Biogas CHP", "biomass", "generator",
                      default_efficiency=0.40, marginal_cost=20, capital_cost=2500,
                      lifetime=20, emission_factor=0.05,
                      input_carrier="biomass"),
]

# Sector 10: Geothermal
_GEOTHERMAL = [
    ComponentTemplate("EC153", "Binary Geothermal Plant", "geothermal", "generator",
                      marginal_cost=5, capital_cost=4000, lifetime=30, emission_factor=0),
]

# Sector 11: Power Electronics (simplified — these are efficiency multipliers)
_POWER_ELECTRONICS = [
    ComponentTemplate("EC164", "Three-Phase Inverter", "power_electronics", "converter",
                      default_efficiency=0.98, marginal_cost=0, capital_cost=50,
                      lifetime=15, input_carrier="electricity", output_carrier="electricity"),
]

# Sector 12: Gas Systems
_GAS = [
    ComponentTemplate("EC193", "Methanation Reactor", "gas_systems", "converter",
                      default_efficiency=0.60, marginal_cost=5, capital_cost=1200,
                      lifetime=20, input_carrier="hydrogen", output_carrier="natural_gas"),
]

# Sector 13: Carbon Capture
_CCUS = [
    ComponentTemplate("EC198", "Post-Combustion Capture", "carbon_capture", "converter",
                      default_efficiency=0.90, marginal_cost=40, capital_cost=2000,
                      lifetime=25, input_carrier="electricity", output_carrier="co2"),
    ComponentTemplate("EC201", "DAC Solid Sorbent", "carbon_capture", "converter",
                      default_efficiency=0.85, marginal_cost=200, capital_cost=4000,
                      lifetime=20, input_carrier="electricity", output_carrier="co2"),
]

# Sector 14: Desalination
_DESAL = [
    ComponentTemplate("EC209", "Reverse Osmosis", "desalination", "converter",
                      default_efficiency=0.40, marginal_cost=1, capital_cost=1500,
                      lifetime=25, input_carrier="electricity", output_carrier="water"),
]

# Sector 15: Thermoelectric
_THERMOELECTRIC = [
    ComponentTemplate("EC216", "Thermoelectric Generator", "thermoelectric", "converter",
                      default_efficiency=0.08, marginal_cost=0, capital_cost=5000,
                      lifetime=20, input_carrier="heat", output_carrier="electricity"),
]


# ---------------------------------------------------------------------------
# Component Registry
# ---------------------------------------------------------------------------

class ComponentRegistry:
    """Registry of all available component templates."""

    def __init__(self):
        self._templates: dict[str, ComponentTemplate] = {}
        self._sectors: dict[str, list[str]] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load all built-in component templates."""
        all_templates = (
            _HYDROGEN + _BATTERIES + _SOLAR + _WIND + _THERMAL +
            _CONVENTIONAL + _MECHANICAL + _HYDRO + _BIOMASS +
            _GEOTHERMAL + _POWER_ELECTRONICS + _GAS + _CCUS +
            _DESAL + _THERMOELECTRIC
        )
        for tmpl in all_templates:
            self._templates[tmpl.ec_id] = tmpl
            sector = tmpl.sector
            if sector not in self._sectors:
                self._sectors[sector] = []
            self._sectors[sector].append(tmpl.ec_id)

    def get(self, ec_id: str) -> ComponentTemplate:
        """Get a component template by EC ID."""
        if ec_id not in self._templates:
            raise KeyError(f"Unknown component: {ec_id!r}. "
                           f"Available: {list(self._templates.keys())}")
        return self._templates[ec_id]

    def list_sectors(self) -> list[str]:
        """List all available sectors."""
        return sorted(self._sectors.keys())

    def list_components(self, sector: str | None = None) -> list[str]:
        """List component IDs, optionally filtered by sector."""
        if sector is None:
            return sorted(self._templates.keys())
        if sector not in self._sectors:
            raise KeyError(f"Unknown sector: {sector!r}")
        return self._sectors[sector]

    def info(self, ec_id: str) -> dict:
        """Get component info as a dictionary."""
        tmpl = self.get(ec_id)
        return {
            "ec_id": tmpl.ec_id,
            "name": tmpl.name,
            "sector": tmpl.sector,
            "category": tmpl.category,
            "efficiency": tmpl.default_efficiency,
            "marginal_cost": tmpl.marginal_cost,
            "capital_cost": tmpl.capital_cost,
            "lifetime": tmpl.lifetime,
            "emission_factor": tmpl.emission_factor,
            "input_carrier": tmpl.input_carrier,
            "output_carrier": tmpl.output_carrier,
        }

    def register(self, template: ComponentTemplate):
        """Register a custom component template."""
        self._templates[template.ec_id] = template
        sector = template.sector
        if sector not in self._sectors:
            self._sectors[sector] = []
        if template.ec_id not in self._sectors[sector]:
            self._sectors[sector].append(template.ec_id)

    @property
    def count(self) -> int:
        return len(self._templates)


# Module-level registry instance
registry = ComponentRegistry()


# ---------------------------------------------------------------------------
# Convenience: add component by EC ID
# ---------------------------------------------------------------------------

def add_component(system: EnergySystem, name: str, ec_id: str,
                  bus: Bus, capacity: float,
                  bus_to: Bus | None = None,
                  carrier_factor: np.ndarray | None = None,
                  extendable: bool = False,
                  max_capacity: float = float("inf"),
                  **overrides):
    """
    Add a component to the system using a registered template.

    For generators (solar, wind, gas, etc.): only `bus` needed.
    For converters (electrolyser, heat pump, etc.): `bus` = input, `bus_to` = output.
    For storage (battery, TES, etc.): only `bus` needed.

    Returns the created component object.
    """
    tmpl = registry.get(ec_id)

    marginal_cost = overrides.pop("marginal_cost", tmpl.marginal_cost)
    capital_cost = overrides.pop("capital_cost", tmpl.capital_cost)
    efficiency = overrides.pop("efficiency", tmpl.default_efficiency)
    emission_factor = overrides.pop("emission_factor", tmpl.emission_factor)

    if tmpl.category == "generator":
        ramp_up = None
        ramp_down = None
        if tmpl.ramp_up_fraction is not None:
            ramp_up = tmpl.ramp_up_fraction * capacity
        if tmpl.ramp_down_fraction is not None:
            ramp_down = tmpl.ramp_down_fraction * capacity

        return system.add_generator(
            name, bus=bus, capacity=capacity,
            marginal_cost=marginal_cost,
            capital_cost=capital_cost,
            emission_factor=emission_factor,
            carrier_factor=carrier_factor,
            p_min=tmpl.p_min_fraction * capacity,
            ramp_up=ramp_up,
            ramp_down=ramp_down,
            extendable=extendable,
            max_capacity=max_capacity,
        )

    elif tmpl.category == "storage":
        e2p = overrides.pop("energy_to_power_ratio", tmpl.energy_to_power_ratio)
        energy_capacity = capacity * e2p

        return system.add_storage(
            name, bus=bus,
            power_capacity=capacity,
            energy_capacity=energy_capacity,
            efficiency_charge=overrides.pop("efficiency_charge", tmpl.efficiency_charge),
            efficiency_discharge=overrides.pop("efficiency_discharge", tmpl.efficiency_discharge),
            self_discharge=tmpl.self_discharge_per_hour,
            marginal_cost=marginal_cost,
            capital_cost_power=capital_cost,
            capital_cost_energy=overrides.pop("capital_cost_energy", capital_cost * 0.3),
            extendable=extendable,
            max_power_capacity=max_capacity,
            max_energy_capacity=max_capacity * e2p if max_capacity < float("inf") else float("inf"),
        )

    elif tmpl.category == "converter":
        if bus_to is None:
            raise ValueError(
                f"Component {ec_id} ({tmpl.name}) is a converter and requires bus_to. "
                f"It converts {tmpl.input_carrier} → {tmpl.output_carrier}."
            )
        return system.add_link(
            name, bus_from=bus, bus_to=bus_to,
            capacity=capacity,
            efficiency=efficiency,
            marginal_cost=marginal_cost,
            capital_cost=capital_cost,
            extendable=extendable,
            max_capacity=max_capacity,
        )

    else:
        raise ValueError(f"Unknown component category: {tmpl.category!r}")
