"""Phase 19 (Paper 1) — certified a-posteriori error bounds for
reduced-order (representative-period) energy optimisation.

`temporal.certify_reduction` returns a `CertifiedBound` that provably
brackets the true full-resolution optimum ``C*`` with a valid lower bound
(weighted optimistic-surrogate relaxation) and a valid upper bound (reduced
capacities fixed into the full model = a feasible point). On a tiny instance
we also solve the full model exactly to recover ``C*`` and confirm:

    lower_bound  <=  C*  <=  upper_bound        (the certificate is valid)
    certified_gap_pct  >=  actual_gap_pct       (the envelope dominates)

These are additive / default-off helpers; nothing else in the suite changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_energy.core import EnergySystem
from nexus_energy.temporal import (
    CertifiedBound,
    aggregate_to_representative_days,
    apply_representative_days,
    certified_reduction_demo,
    certify_reduction,
)


def _make_instance(n_days=8, hpd=6, seed=3):
    hours = np.arange(hpd)
    rng = np.random.default_rng(seed)
    loads, solars = [], []
    for _ in range(n_days):
        base = rng.uniform(40, 90)
        amp = rng.uniform(0, 60)
        pk = rng.integers(0, hpd)
        loads.append(base + amp * np.exp(-((hours - pk) ** 2) / 2.0))
        solars.append(np.clip(
            rng.uniform(0, 1) * np.sin((hours - 1) / hpd * np.pi), 0, 1))
    return np.concatenate(loads), np.concatenate(solars)


def _factory(full_load, full_solar, n_days, hpd):
    def make():
        s = EnergySystem("cert_test")
        b = s.add_bus("elec")
        g = s.add_generator("solar", b, capacity=1.0, marginal_cost=0.0,
                            extendable=True, max_capacity=500.0,
                            capital_cost=50.0)
        g.carrier_factor = full_solar.copy()
        s.add_generator("gas", b, capacity=1.0, marginal_cost=40.0,
                        extendable=True, max_capacity=500.0, capital_cost=10.0)
        s.add_load("demand", b, amount=full_load.copy())
        s.set_timesteps(n_days * hpd, dt=1.0)
        return s
    return make


class TestCertifiedReduction:
    def test_bracket_holds_on_tiny_instance(self):
        n_days, hpd, n_rep = 8, 6, 3
        load, solar = _make_instance(n_days, hpd)
        make = _factory(load, solar, n_days, hpd)

        rep = aggregate_to_representative_days(
            {"load": load, "solar": solar}, n_days=n_rep, hours_per_day=hpd)
        red = make()
        apply_representative_days(red, rep, {"load": "demand", "solar": "solar"})
        red_res = red.optimise()

        cert = certify_reduction(
            make, rep, dict(red_res.capacity_additions),
            reduced_cost=red_res.total_cost, period_length=hpd)

        true_opt = make().optimise().total_cost

        assert isinstance(cert, CertifiedBound)
        # The certificate is a valid envelope around the true optimum.
        assert cert.lower_bound <= true_opt + 1e-6
        assert true_opt <= cert.upper_bound + 1e-6
        assert cert.brackets_optimum
        assert cert.gap_abs >= -1e-6
        assert cert.gap_pct >= -1e-9

    def test_certified_gap_dominates_actual_gap(self):
        d = certified_reduction_demo()
        cert = d["certified"]
        # True optimum sits inside the certified envelope.
        assert cert.lower_bound <= d["true_optimum"] + 1e-6
        assert d["true_optimum"] <= cert.upper_bound + 1e-6
        # The certified gap is a guaranteed upper bound on the actual gap.
        assert d["certified_gap_pct"] >= d["actual_gap_pct"] - 1e-9
        # A good reduction's actual gap is tiny.
        assert d["actual_gap_pct"] < 5.0

    def test_lower_bound_valid_across_seeds(self):
        # The optimistic-surrogate LB must never exceed the true optimum, and
        # the feasible-point UB must never fall below it — for ANY seed.
        for seed in range(8):
            n_days, hpd, n_rep = 8, 6, 3
            load, solar = _make_instance(n_days, hpd, seed=seed)
            make = _factory(load, solar, n_days, hpd)
            rep = aggregate_to_representative_days(
                {"load": load, "solar": solar}, n_days=n_rep, hours_per_day=hpd)
            red = make()
            apply_representative_days(
                red, rep, {"load": "demand", "solar": "solar"})
            red_res = red.optimise()
            cert = certify_reduction(
                make, rep, dict(red_res.capacity_additions),
                reduced_cost=red_res.total_cost, period_length=hpd)
            true_opt = make().optimise().total_cost
            assert cert.lower_bound <= true_opt + 1e-6, (
                f"seed {seed}: LB {cert.lower_bound} > true {true_opt}")
            assert true_opt <= cert.upper_bound + 1e-6, (
                f"seed {seed}: UB {cert.upper_bound} < true {true_opt}")

    def test_accepts_reduced_system_carrying_rep_periods(self):
        # certify_reduction should also accept the configured reduced
        # EnergySystem (which stashes `_rep_periods`) instead of the rep object.
        n_days, hpd, n_rep = 6, 6, 2
        load, solar = _make_instance(n_days, hpd, seed=1)
        make = _factory(load, solar, n_days, hpd)
        rep = aggregate_to_representative_days(
            {"load": load, "solar": solar}, n_days=n_rep, hours_per_day=hpd)
        red = make()
        apply_representative_days(red, rep, {"load": "demand", "solar": "solar"})
        red_res = red.optimise()
        cert = certify_reduction(
            make, red, dict(red_res.capacity_additions),
            reduced_cost=red_res.total_cost, period_length=hpd)
        true_opt = make().optimise().total_cost
        assert cert.lower_bound <= true_opt + 1e-6 <= cert.upper_bound + 1e-6

    def test_rejects_bad_reduction_descriptor(self):
        n_days, hpd = 4, 6
        load, solar = _make_instance(n_days, hpd)
        make = _factory(load, solar, n_days, hpd)
        with pytest.raises(ValueError):
            certify_reduction(make, object(), {"solar": 1.0}, period_length=hpd)
