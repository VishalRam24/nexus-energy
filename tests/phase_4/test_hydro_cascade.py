"""Phase 4 — multi-reservoir hydro cascade.

Cascade semantics: an upstream reservoir's spill becomes a downstream
reservoir's natural inflow. SOC evolution at the downstream reservoir
adds (upstream_spill[t-delay] + natural_inflow[t]) on top of the
charge / discharge balance. Without this, modelling a Yangtze /
Columbia / Snowy-style chain is impossible — each reservoir would have
to be solved as if hydrologically independent.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def _two_reservoir_cascade(spill_to_downstream: bool):
    """Upper reservoir gets natural inflow; lower reservoir gets either
    nothing (spill_to=None) or upstream spill (spill_to=lower).

    Three timesteps, demand at the same bus the lower reservoir feeds.
    """
    sys = ne.EnergySystem("hydro")
    sys.set_timesteps(3, dt=1.0)
    bus = sys.add_bus("e")

    # Upper reservoir: starts at 50 MWh of stored energy; gets 30 MWh/h
    # of natural inflow; physical capacity 60 MWh forces it to spill if
    # not generated quickly enough.
    upper = sys.add_storage(
        "upper", bus=bus, power_capacity=20, energy_capacity=60,
        soc_initial=0.5, cyclic=False, inflow=np.array([30.0, 30.0, 30.0]),
    )
    lower = sys.add_storage(
        "lower", bus=bus, power_capacity=20, energy_capacity=60,
        soc_initial=0.5, cyclic=False,
    )
    if spill_to_downstream:
        upper.spill_to = lower

    # Discourage discharging from the upper directly (high marginal cost
    # on its discharge leg) so the LP prefers to spill upstream and
    # discharge downstream when cascade is active.
    upper.marginal_cost = 100.0
    lower.marginal_cost = 0.0

    sys.add_load("ld", bus=bus, amount=np.array([10.0, 10.0, 10.0]))
    return sys, upper, lower


class TestHydroCascade:
    def test_cascade_carries_spill_into_downstream_soc(self):
        sys, upper, lower = _two_reservoir_cascade(spill_to_downstream=True)
        result = sys.optimise()
        assert result.status == "optimal"
        soc_low = result.storage_soc["lower"]
        # Cascade transfers: lower SOC must increase from inflow it didn't
        # otherwise have. Without cascade, the lower reservoir starts at
        # 30 MWh and (lacking natural inflow) can only discharge down.
        # With cascade, upper's spill lifts the lower's SOC.
        assert soc_low[-1] > 30.0

    def test_no_cascade_means_no_lower_inflow(self):
        sys, upper, lower = _two_reservoir_cascade(spill_to_downstream=False)
        result = sys.optimise()
        assert result.status == "optimal"
        soc_low = result.storage_soc["lower"]
        # Without cascade, lower reservoir has no inflow. At best it stays
        # at its initial SOC (30 MWh) by neither charging nor discharging.
        assert soc_low[-1] <= 30.0 + 1e-6

    def test_inflow_alone_lifts_soc_when_no_dispatch(self):
        # Single reservoir with inflow, no load → SOC must rise by inflow.
        sys = ne.EnergySystem("inflow_only")
        sys.set_timesteps(3, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_storage(
            "r", bus=bus, power_capacity=10, energy_capacity=200,
            soc_initial=0.0, cyclic=False, inflow=np.array([10.0, 10.0, 10.0]),
        )
        # Tiny load to keep the LP non-trivial.
        sys.add_generator("dummy", bus=bus, capacity=10, marginal_cost=0)
        sys.add_load("ld", bus=bus, amount=np.array([0.0, 0.0, 0.0]))
        # Force simplex: the zero-cost dummy generator makes the optimum
        # non-unique (cost is 0 whether it charges storage or not), so only a
        # vertex solution pins SOC to the inflow-only value. The ipm_fast
        # default returns an interior point that loads the dummy arbitrarily.
        result = sys.optimise(lp_backend="simplex")
        assert result.status == "optimal"
        # 3 hours × 10 MWh inflow → final SOC ≈ 30 (less spill).
        assert result.storage_soc["r"][-1] == pytest.approx(30.0, abs=0.5)
