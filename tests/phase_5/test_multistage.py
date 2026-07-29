"""Phase 5 — perfect-foresight multi-stage investment with vintages.

A single LP that spans multiple decade snapshots (e.g. 2030 / 2040 /
2050). Generators built at stage S live for ``lifetime_years`` and
then retire; scheduled retirement via ``retire_at_year`` is enforced.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def _build_stage(name: str, demand: float,
                 capital_cost_solar: float = 100_000.0) -> ne.EnergySystem:
    sys = ne.EnergySystem(name)
    sys.set_timesteps(1, dt=1.0)
    bus = sys.add_bus("e")
    sys.add_load("ld", bus=bus, amount=demand)
    sys.add_generator(
        "solar", bus=bus, capacity=0,
        marginal_cost=0.0,
        capital_cost=capital_cost_solar,
        extendable=True, max_capacity=1_000.0,
        lifetime_years=15,
    )
    return sys


class TestMultiStage:
    def test_vintage_carries_forward_until_lifetime_expires(self):
        # Two stages 10 years apart; solar lifetime 15yr means stage-0
        # build is still live at stage 1 (year+10 < year+15). Stage-1
        # CapEx is half of stage-0 (reflects real solar learning curve)
        # so the LP strictly prefers to defer any capacity it can.
        mss = ne.MultiStageSystem("vintage")
        mss.add_stage(year=2030,
                      system=_build_stage("2030", demand=100.0,
                                          capital_cost_solar=100_000.0))
        mss.add_stage(year=2040,
                      system=_build_stage("2040", demand=150.0,
                                          capital_cost_solar=50_000.0))

        result = mss.optimise()
        assert result.status == "optimal"
        # 2030: must build 100 (no other way to serve stage-0 demand).
        # 2040: vintage 100 still live → build only 50 more, at cheaper cost.
        assert result.new_builds["solar"][0] == pytest.approx(100.0, abs=1e-3)
        assert result.new_builds["solar"][1] == pytest.approx(50.0, abs=1e-3)
        # Active cap at stage 1 == 100 (vintage) + 50 (new) = 150
        assert result.capacity_active["solar"][1] == pytest.approx(150.0, abs=1e-3)
        # Objective = 100 × $100k + 50 × $50k = $12.5M
        assert result.total_cost == pytest.approx(12_500_000.0, rel=1e-4)

    def test_vintage_retires_after_lifetime(self):
        # Three stages 10 years apart, lifetime=15 → stage-0 vintage
        # expires by stage 2 (2050 >= 2030 + 15). LP must rebuild.
        mss = ne.MultiStageSystem("expire")
        mss.add_stage(year=2030, system=_build_stage("2030", demand=100.0))
        mss.add_stage(year=2040, system=_build_stage("2040", demand=100.0))
        mss.add_stage(year=2050, system=_build_stage("2050", demand=100.0))

        result = mss.optimise()
        assert result.status == "optimal"
        # Stage 0 vintage live at stage 1 (age 10 < 15), dead at stage 2 (age 20 ≥ 15).
        # Stage 1 vintage live at stage 2 (age 10 < 15). So LP wants to
        # minimise total build. Cheapest: build 100 in 2030, 0 in 2040
        # (vintage covers), 100 in 2050 (both 2030 + 2040 vintages dead
        # or dying? 2040 still live at 2050 since 2040+15=2055>2050).
        # Ah wait — 2040 vintage IS live at 2050. So stage 2's 100 MW
        # demand can be met by stage-1's vintage if stage 1 built 100.
        # But stage 1's demand is also 100 and its vintage from stage 0
        # covers it → stage 1 needn't build. But then stage 2 has
        # nothing live (stage 0 expired, stage 1 didn't build). LP is
        # forced to either build at stage 1 (covers both 1 and 2) or
        # build at stage 0 and then again at stage 2. Both strategies
        # buy 200 MW of capital cost total → tie broken either way.
        nb = result.new_builds["solar"]
        total_built = nb.sum()
        assert total_built == pytest.approx(200.0, abs=1e-3)
        # Every stage must be served
        assert result.capacity_active["solar"][0] >= 100.0 - 1e-3
        assert result.capacity_active["solar"][1] >= 100.0 - 1e-3
        assert result.capacity_active["solar"][2] >= 100.0 - 1e-3

    def test_scheduled_retirement_forces_rebuild(self):
        # Coal exists (brownfield) at 2030; retires in 2040 via
        # retire_at_year. Solar takes over.
        mss = ne.MultiStageSystem("retire")

        s0 = ne.EnergySystem("2030")
        s0.set_timesteps(1, dt=1.0)
        b0 = s0.add_bus("e")
        s0.add_load("ld", bus=b0, amount=80.0)
        s0.add_generator("coal", bus=b0, capacity=100,
                         marginal_cost=40.0, retire_at_year=2040)
        s0.add_generator("solar", bus=b0, capacity=0,
                         marginal_cost=0.0, capital_cost=50_000.0,
                         extendable=True, max_capacity=500.0,
                         lifetime_years=25)

        s1 = ne.EnergySystem("2040")
        s1.set_timesteps(1, dt=1.0)
        b1 = s1.add_bus("e")
        s1.add_load("ld", bus=b1, amount=80.0)
        # Coal is still in the stage but marked retired
        s1.add_generator("coal", bus=b1, capacity=100,
                         marginal_cost=40.0, retire_at_year=2040)
        s1.add_generator("solar", bus=b1, capacity=0,
                         marginal_cost=0.0, capital_cost=50_000.0,
                         extendable=True, max_capacity=500.0,
                         lifetime_years=25)

        mss.add_stage(year=2030, system=s0)
        mss.add_stage(year=2040, system=s1)

        result = mss.optimise()
        assert result.status == "optimal"
        # At stage 1 (2040), coal is gone → solar must cover 80 MW.
        assert result.capacity_active["solar"][1] >= 80.0 - 1e-3
        # Coal active at stage 1 should be zero (retired).
        assert result.capacity_active["coal"][1] == pytest.approx(0.0, abs=1e-3)
