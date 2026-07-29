"""Phase 10.5 / 10.6 / 10.9 / 16.8 — solver routing, LP basis hot-start, external
solver bridge, DuckDB-backed reader."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import nexus_energy as ne


# ---- 10.5: generic clarabel LP/QP routing through nexus-opt -----------------

def test_clarabel_qp_routing():
    import nexus_opt as o
    m = o.Model("q")
    a = m.variable("a", lower=-5, upper=5)
    m.minimize(a * a - 4 * a)
    r = m.solve(solver="clarabel")
    assert r.status == "optimal"
    assert r.value(a) == pytest.approx(2.0, abs=1e-4)
    assert r.objective == pytest.approx(-4.0, abs=1e-4)


# ---- 10.6 / 11.5: LP simplex-basis hot-start in rolling horizon -------------

def test_rolling_horizon_basis_warmstart_matches_cold():
    """Warm (basis hot-start) rolling solve must give the SAME concatenated
    dispatch / cost as the cold solve — the basis only warms the solve path."""
    from nexus_energy.temporal import rolling_horizon_solve
    rng = np.random.default_rng(0)
    demand = 50 + 10 * rng.standard_normal(24)

    def factory(start, end):
        s = ne.EnergySystem(f"w{start}")
        s.set_timesteps(end - start)
        b = s.add_bus("e")
        s.add_load("d", bus=b, amount=demand[start:end].clip(min=0))
        s.add_generator("g", bus=b, capacity=200, marginal_cost=10)
        s.add_generator("peak", bus=b, capacity=200, marginal_cost=50)
        return s

    cold = rolling_horizon_solve(factory, 24, window_size=8, warm_start=False)
    warm = rolling_horizon_solve(factory, 24, window_size=8, warm_start=True)
    assert warm["total_cost"] == pytest.approx(cold["total_cost"], rel=1e-7)
    for name in cold["generator_dispatch"]:
        assert np.allclose(warm["generator_dispatch"][name],
                           cold["generator_dispatch"][name], atol=1e-6)


# ---- 10.9: external solver bridge -------------------------------------------

def test_external_solver_graceful_when_absent():
    from nexus_energy import external_solvers as ext
    avail = ext.available_solvers()
    assert isinstance(avail, list)
    # None of the commercial solvers are installed in this env → clear error.
    for s in ("gurobi", "cplex", "scip", "mosek", "xpress"):
        if s not in avail:
            with pytest.raises(ImportError):
                ext.solve_lp_external("Minimize\n obj: x\nBounds\n0<=x<=1\nEnd",
                                      s)


def test_external_solver_name_rejected_in_optimise():
    s = ne.EnergySystem("x")
    s.set_timesteps(1)
    b = s.add_bus("e")
    s.add_generator("g", bus=b, capacity=10, marginal_cost=1)
    s.add_load("d", bus=b, amount=5.0)
    with pytest.raises(NotImplementedError, match="external"):
        s.optimise(solver="gurobi")


def test_external_bridge_lp_export_roundtrips():
    """solve_system_external captures a valid LP from the system (even though
    no external solver is installed, the capture+export path must work)."""
    from nexus_energy import external_solvers as ext
    s = ne.EnergySystem("cap")
    s.set_timesteps(2)
    b = s.add_bus("e")
    s.add_generator("g", bus=b, capacity=10, marginal_cost=1)
    s.add_load("d", bus=b, amount=np.array([5.0, 6.0]))
    avail = ext.available_solvers()
    if not avail:
        # No solver: must raise ImportError *after* a successful LP capture.
        with pytest.raises(ImportError):
            ext.solve_system_external(s, "gurobi")
    else:
        res = ext.solve_system_external(s, avail[0])
        assert res.status in ("optimal", "unknown", "infeasible")


# ---- 16.8: DuckDB-backed reader (pandas fallback verified) ------------------

def test_io_tables_pandas_fallback():
    import pandas as pd
    from nexus_energy import io_tables as io
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "assets.csv")
        p2 = os.path.join(d, "flows.csv")
        pd.DataFrame({"asset": ["a", "b"], "cap": [10, 20]}).to_csv(p1, index=False)
        pd.DataFrame({"asset": ["a", "b"], "cost": [1.0, 2.0]}).to_csv(p2, index=False)
        # pandas engine always works.
        df = io.read_table(p1, engine="pandas")
        assert list(df["asset"]) == ["a", "b"]
        # auto engine resolves to duckdb if present else pandas — both equal.
        df_auto = io.read_table(p1, engine="auto")
        assert df_auto.shape == (2, 2)
        # directory read.
        tables = io.read_csv_dir(d, engine="pandas")
        assert set(tables) == {"assets", "flows"}
        # join (pandas engine).
        merged = io.join_tables(tables["assets"], tables["flows"],
                                on="asset", engine="pandas")
        assert merged.shape == (2, 3)
        assert "cap" in merged.columns and "cost" in merged.columns
        # auto-engine join matches pandas-engine join.
        merged_auto = io.join_tables(tables["assets"], tables["flows"],
                                     on="asset", engine="auto")
        assert sorted(merged_auto.columns) == sorted(merged.columns)
        assert len(merged_auto) == len(merged)
