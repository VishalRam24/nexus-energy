"""Phase 2.x / 6 — ``set_fuel_supply_limit`` annual fuel bucket."""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def _two_gas_system(T=8, demand=80.0):
    """Two cheap gas gens share a fuel bucket; an expensive oil peaker
    is the escape valve when the gas budget is tight."""
    sys = ne.EnergySystem("fuel_supply")
    sys.set_timesteps(T)
    b = sys.add_bus("e")
    # Gas gens: heat rate 8 MMBtu/MWh, $20/MWh marginal (folds heat_rate × price)
    sys.add_generator("gas_a", bus=b, capacity=100.0, marginal_cost=20.0)
    sys.add_generator("gas_b", bus=b, capacity=100.0, marginal_cost=20.0)
    # Oil peaker: no heat_rate coefficient in fuel bucket; expensive.
    sys.add_generator("oil", bus=b, capacity=200.0, marginal_cost=200.0)
    sys.add_load("ld", bus=b, amount=np.full(T, float(demand)))
    return sys


def test_fuel_supply_unlimited_runs_cheapest():
    """Baseline — no bucket set, LP picks gas at $20 over oil at $200."""
    sys = _two_gas_system()
    res = sys.optimise()
    assert res.status == "optimal"
    gas_a = sys._generators[0]
    gas_b = sys._generators[1]
    oil = sys._generators[2]
    oil_energy = sum(res._raw.value(p) for p in oil._p_vars)
    assert oil_energy < 1e-6


def test_fuel_supply_cap_forces_peaker():
    """Bucket tight enough that gas can't cover full load → oil kicks in."""
    T = 8
    demand = 80.0
    # Total fuel needed if ALL 640 MWh goes to gas @ heat_rate=8:
    # 640 × 8 = 5120 MMBtu. Cap to 3000 MMBtu (≤ 375 MWh of gas).
    sys = _two_gas_system(T=T, demand=demand)
    sys.set_fuel_supply_limit(
        "gas", max_fuel=3000.0,
        generators={"gas_a": 8.0, "gas_b": 8.0},
    )
    res = sys.optimise()
    assert res.status == "optimal"

    gas_a = sys._generators[0]
    gas_b = sys._generators[1]
    oil = sys._generators[2]
    gas_energy = sum(res._raw.value(p) for p in gas_a._p_vars) \
               + sum(res._raw.value(p) for p in gas_b._p_vars)
    oil_energy = sum(res._raw.value(p) for p in oil._p_vars)
    # Total gas fuel must not exceed 3000 MMBtu.
    assert gas_energy * 8.0 <= 3000.0 + 1e-6
    # Oil had to fill the gap.
    assert oil_energy > 1e-3


def test_fuel_supply_binds_exactly_when_cheaper_than_peaker():
    """When peaker > gas cost, LP runs the gas bucket to its cap."""
    T = 8
    sys = _two_gas_system(T=T, demand=80.0)
    cap = 2400.0  # 300 MWh of gas
    sys.set_fuel_supply_limit(
        "gas", max_fuel=cap, generators={"gas_a": 8.0, "gas_b": 8.0})
    res = sys.optimise()
    assert res.status == "optimal"
    gas_a = sys._generators[0]
    gas_b = sys._generators[1]
    gas_mmbtu = 8.0 * (sum(res._raw.value(p) for p in gas_a._p_vars)
                       + sum(res._raw.value(p) for p in gas_b._p_vars))
    assert abs(gas_mmbtu - cap) < 1e-3


def test_fuel_supply_missing_generator_raises():
    sys = _two_gas_system()
    with pytest.raises(ValueError, match="not in system"):
        sys.set_fuel_supply_limit(
            "gas", max_fuel=1000.0, generators={"ghost": 8.0})
        sys.optimise()


def test_fuel_supply_empty_generators_raises():
    sys = _two_gas_system()
    with pytest.raises(ValueError, match="at least one generator"):
        sys.set_fuel_supply_limit("gas", max_fuel=1000.0, generators={})
