"""Regression: apply_representative_days must align profile columns by the
aggregation's feature order, not by sorted(timeseries_map.keys()).

Bug (found via showcase Case A): when the aggregation `timeseries` dict carries
a selection-only feature (e.g. a residual) absent from the apply `timeseries_map`,
the legacy code indexed columns by sorted(map keys) and silently installed the
WRONG profile (solar got the residual series → infeasible/garbage solves). The
fix carries `RepresentativePeriods.feature_names` (column order) and indexes by it.
"""

from __future__ import annotations

import numpy as np

import nexus_energy as ne
from nexus_energy.temporal import (
    aggregate_to_representative_days,
    apply_representative_days,
)


def test_apply_aligns_by_feature_names_with_selection_only_feature():
    rng = np.random.default_rng(0)
    H = 24 * 20
    # 'resid' sorts BETWEEN 'demand' and 'solar' → it shifts solar's column.
    demand = 100 + 20 * np.sin(np.arange(H) * 2 * np.pi / 24) + rng.standard_normal(H)
    solar = np.clip(np.sin(np.arange(H) * np.pi / 24), 0, 1)          # in [0,1]
    resid = 1000 + rng.standard_normal(H)                              # large, distinct scale
    rep = aggregate_to_representative_days(
        {"demand": demand, "solar": solar, "resid": resid}, n_days=4)
    assert rep.feature_names == ["demand", "resid", "solar"]  # sorted order recorded

    sys = ne.EnergySystem("align");
    b = sys.add_bus("e")
    sys.add_load("L", bus=b, amount=0.0)
    sys.add_generator("PV", bus=b, capacity=200, marginal_cost=0,
                      carrier_factor=np.ones(1))
    sys.add_generator("gas", bus=b, capacity=500, marginal_cost=50)
    # Map only demand+solar (NOT resid). Pre-fix: solar would get the resid
    # column (values ~1000) → carrier_factor >> 1 (garbage). Post-fix: correct.
    apply_representative_days(sys, rep, timeseries_map={"L": "L", "solar": "PV"})
    pv = sys._generators[0]
    # PV carrier_factor must be the SOLAR series (in [0,1]), not resid (~1000).
    assert pv.carrier_factor.max() <= 1.0 + 1e-9, \
        f"wrong profile installed: max cf={pv.carrier_factor.max()} (resid leaked in)"
    assert pv.carrier_factor.min() >= -1e-9
    r = sys.optimise()
    assert r.status == "optimal"
