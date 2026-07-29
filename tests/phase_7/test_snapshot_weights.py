"""Phase 7 — Snapshot weights foundation.

A snapshot weight ``w[t]`` is the multiplicative factor by which timestep
``t`` represents real-world time in cost / emission / policy aggregations.
Per-timestep physics (SOC, bus balance, ramps, capacity) ignore weights —
they bind on each snapshot independently.

Equivalence pattern: a 2-step horizon with uniform weight=1 must produce
the same total cost as a 1-step horizon with weight=2 when all physics
hold per-step. That is the foundational invariant for representative
periods (Phase 7 next step).
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


class TestSnapshotWeights:
    def test_uniform_weight_one_matches_no_weights(self):
        """w=[1,1,...] must produce the exact same objective as no weights."""
        def build():
            sys = ne.EnergySystem("base")
            sys.set_timesteps(4, dt=1.0)
            bus = sys.add_bus("e")
            sys.add_load("ld", bus=bus, amount=np.array([50.0, 80.0, 60.0, 40.0]))
            sys.add_generator("g", bus=bus, capacity=200,
                              marginal_cost=20.0)
            return sys

        baseline = build().optimise()
        weighted = build()
        weighted.set_snapshot_weights(np.ones(4))
        weighted_res = weighted.optimise()

        assert baseline.status == "optimal"
        assert weighted_res.status == "optimal"
        assert weighted_res.total_cost == pytest.approx(baseline.total_cost,
                                                        rel=1e-6)

    def test_weight_two_doubles_cost(self):
        """w=2 on every snapshot doubles the objective vs w=1."""
        def build():
            sys = ne.EnergySystem("dbl")
            sys.set_timesteps(3, dt=1.0)
            bus = sys.add_bus("e")
            sys.add_load("ld", bus=bus, amount=100.0)
            sys.add_generator("g", bus=bus, capacity=150,
                              marginal_cost=30.0)
            return sys

        s1 = build()
        s1.set_snapshot_weights(np.ones(3))
        r1 = s1.optimise()

        s2 = build()
        s2.set_snapshot_weights(2.0 * np.ones(3))
        r2 = s2.optimise()

        assert r1.status == "optimal"
        assert r2.status == "optimal"
        # 3 timesteps × 100 MW × $30 × dt = $9k for w=1; double for w=2.
        assert r1.total_cost == pytest.approx(9_000.0, rel=1e-6)
        assert r2.total_cost == pytest.approx(18_000.0, rel=1e-6)

    def test_two_snapshots_collapse_to_one_with_weight_two(self):
        """A 1-snapshot system with w=[2] must match a 2-snapshot
        system with w=[1,1] when both snapshots see the same load.
        This is the equivalence representative-period TDR depends on."""
        def build_full():
            sys = ne.EnergySystem("full")
            sys.set_timesteps(2, dt=1.0)
            bus = sys.add_bus("e")
            sys.add_load("ld", bus=bus, amount=np.array([75.0, 75.0]))
            sys.add_generator("g", bus=bus, capacity=200,
                              marginal_cost=15.0)
            return sys

        def build_collapsed():
            sys = ne.EnergySystem("rep")
            sys.set_timesteps(1, dt=1.0)
            bus = sys.add_bus("e")
            sys.add_load("ld", bus=bus, amount=np.array([75.0]))
            sys.add_generator("g", bus=bus, capacity=200,
                              marginal_cost=15.0)
            sys.set_snapshot_weights(np.array([2.0]))
            return sys

        rfull = build_full().optimise()
        rrep = build_collapsed().optimise()

        assert rfull.status == "optimal"
        assert rrep.status == "optimal"
        assert rrep.total_cost == pytest.approx(rfull.total_cost, rel=1e-6)

    def test_weights_scale_emission_cap(self):
        """Weighted emissions must respect the cap: a w=10 snapshot
        consumes 10× the emission budget of a w=1 snapshot."""
        sys = ne.EnergySystem("em")
        sys.set_timesteps(2, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("dirty", bus=bus, capacity=200,
                          marginal_cost=10, emission_factor=1.0)
        sys.add_generator("clean", bus=bus, capacity=200,
                          marginal_cost=50, emission_factor=0.0)
        sys.set_snapshot_weights(np.array([1.0, 10.0]))
        # Cap = 100 tCO2. Weighted emissions = 1 * p[0] + 10 * p[1].
        # If LP runs dirty fully: 1*100 + 10*100 = 1100 — way over.
        # Cap forces clean dispatch in the high-weight snapshot.
        sys.set_emission_limit(100.0)
        result = sys.optimise()

        assert result.status == "optimal"
        dirty = result.generator_dispatch["dirty"]
        clean = result.generator_dispatch["clean"]
        weighted_em = 1.0 * dirty[0] + 10.0 * clean[0] * 0.0 \
                      + 10.0 * dirty[1] + 1.0 * clean[1] * 0.0
        assert weighted_em <= 100.0 + 1e-3
        # Bus balance still holds per snapshot.
        assert dirty[0] + clean[0] == pytest.approx(100.0, abs=1e-3)
        assert dirty[1] + clean[1] == pytest.approx(100.0, abs=1e-3)

    def test_weights_scale_rps_target(self):
        """RPS fraction is on weighted load — weight increase on a
        clean-friendly snapshot does not break the LP."""
        sys = ne.EnergySystem("rps")
        sys.set_timesteps(2, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=np.array([100.0, 100.0]))
        sys.add_generator("gas", bus=bus, capacity=200, marginal_cost=20,
                          tech="gas")
        sys.add_generator("wind", bus=bus, capacity=200, marginal_cost=0,
                          tech="wind")
        sys.set_snapshot_weights(np.array([1.0, 3.0]))
        sys.set_rps(fraction=0.5, qualifying_techs=["wind"])
        # Total weighted load = 100*1 + 100*3 = 400 MWh.
        # 50% RPS → wind ≥ 200 MWh weighted = 1*w0 + 3*w1 ≥ 200.
        # Wind is cheaper → LP picks wind everywhere up to load cap.
        result = sys.optimise()
        assert result.status == "optimal"
        wind = result.generator_dispatch["wind"]
        weighted_wind = 1.0 * wind[0] + 3.0 * wind[1]
        assert weighted_wind >= 200.0 - 1e-3

    def test_invalid_length_raises(self):
        sys = ne.EnergySystem("bad")
        sys.set_timesteps(3, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=10.0)
        sys.add_generator("g", bus=bus, capacity=50, marginal_cost=1)
        sys.set_snapshot_weights(np.array([1.0, 1.0]))  # length 2 ≠ T=3
        with pytest.raises(ValueError, match="length"):
            sys.optimise()

    def test_negative_weight_rejected(self):
        sys = ne.EnergySystem("neg")
        sys.set_timesteps(2, dt=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            sys.set_snapshot_weights(np.array([1.0, -0.5]))
