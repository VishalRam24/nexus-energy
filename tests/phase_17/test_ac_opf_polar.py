"""N_En_Phase 17.3 — Polar AC-OPF parity against pandapower PIPS.

These tests rebuild IEEE case9 / case14 / case30 from pandapower's
internal MATPOWER `_ppc` dict (via the shared benchmark helper) and
solve the true polar AC-OPF with `solve_ac_opf_polar`. Both sides are
interior-point NLP solves on the non-convex polar formulation, so
parity is a useful sanity check, but:

- case9 / case14 converge to the same local optimum (within 1e-4
  relative on the objective).
- case30 is non-convex with six generators and has multiple local
  optima of near-identical quality; IPOPT from a flat start lands in
  a basin marginally better than the one PIPS reaches
  (nexus ≈ $577.14 vs pandapower ≈ $578.49, a Δ of ~0.23 %). We
  assert that nexus is *at least as good* as pandapower (minimum-cost
  sense) and that the gap stays under 5e-3 — i.e. nexus does not
  silently regress to a worse basin.

pandapower runs in the isolated venv
`test_projects/test_project_1/pandapower/.venv` (pandapower 2.x,
numpy<2) — skipped if that venv is missing.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

BENCH_DIR = (Path(__file__).resolve().parents[2] / "benchmarks")
sys.path.insert(0, str(BENCH_DIR))

pytest.importorskip("casadi")

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Reference pandapower lives in its own isolated venv outside the package.
# Override with NEXUS_PANDAPOWER_DIR if you keep the clone elsewhere.
PANDAPOWER_PY = Path(
    os.environ.get(
        "NEXUS_PANDAPOWER_DIR",
        _REPO_ROOT / "test_projects/test_project_1/pandapower",
    )
) / ".venv/bin/python"
if not PANDAPOWER_PY.exists():
    pytest.skip("pandapower reference venv not available",
                allow_module_level=True)

from ac_opf_vs_pandapower import run_pandapower, build_nexus_from_ppc  # noqa: E402

import nexus_energy as ne  # noqa: E402


@pytest.fixture(scope="module")
def case9_ref():
    return run_pandapower("case9")


@pytest.fixture(scope="module")
def case14_ref():
    return run_pandapower("case14")


@pytest.fixture(scope="module")
def case30_ref():
    return run_pandapower("case30")


def _solve_polar(ref: dict) -> tuple[float, ne.ACOpfPolarResult]:
    system, cp0 = build_nexus_from_ppc(ref)
    res = ne.solve_ac_opf_polar(system, snapshot=0)
    assert res.status == "optimal"
    return res.total_cost + cp0, res


class TestCase9:
    def test_objective_parity(self, case9_ref):
        """case9 is convex enough in practice for both solvers to match."""
        pm_obj = case9_ref["objective"]
        nx_obj, _ = _solve_polar(case9_ref)
        rel_gap = (pm_obj - nx_obj) / pm_obj
        assert abs(rel_gap) < 1e-4, (
            f"case9 polar gap {rel_gap:+.2e} exceeds 1e-4"
        )

    def test_voltage_magnitudes(self, case9_ref):
        _, res = _solve_polar(case9_ref)
        for pm_bus, pm_v in case9_ref["bus_vm_pu"].items():
            nx_v = res.voltage_mag[f"b{pm_bus}"]
            assert abs(nx_v - pm_v) < 5e-4, (
                f"case9 bus {pm_bus} |V| differ: nexus {nx_v:.5f} vs "
                f"pandapower {pm_v:.5f}"
            )


class TestCase14:
    def test_objective_parity(self, case14_ref):
        pm_obj = case14_ref["objective"]
        nx_obj, _ = _solve_polar(case14_ref)
        rel_gap = (pm_obj - nx_obj) / pm_obj
        assert abs(rel_gap) < 1e-4, (
            f"case14 polar gap {rel_gap:+.2e} exceeds 1e-4"
        )

    def test_voltage_magnitudes(self, case14_ref):
        _, res = _solve_polar(case14_ref)
        for pm_bus, pm_v in case14_ref["bus_vm_pu"].items():
            nx_v = res.voltage_mag[f"b{pm_bus}"]
            assert abs(nx_v - pm_v) < 2e-3, (
                f"case14 bus {pm_bus} |V| differ: nexus {nx_v:.5f} vs "
                f"pandapower {pm_v:.5f}"
            )


class TestCase30:
    """case30 is non-convex; PIPS and IPOPT can reach different local
    optima. Assert nexus is not worse than PIPS, and the absolute gap
    stays under 5e-3 relative — i.e. we haven't silently regressed to
    a far-worse basin.
    """

    def test_objective_not_worse(self, case30_ref):
        pm_obj = case30_ref["objective"]
        nx_obj, _ = _solve_polar(case30_ref)
        rel_gap = (pm_obj - nx_obj) / pm_obj
        assert rel_gap > -1e-4, (
            f"case30 polar regression: nexus worse than pandapower "
            f"by {rel_gap:+.2e}"
        )
        assert abs(rel_gap) < 5e-3, (
            f"case30 polar gap {rel_gap:+.2e} exceeds 5e-3 — "
            f"possible basin shift worth investigating"
        )

    def test_physics_feasible(self, case30_ref):
        """Sanity: voltages in-bounds, no crazy angles."""
        system, _ = build_nexus_from_ppc(case30_ref)
        res = ne.solve_ac_opf_polar(system, snapshot=0)
        assert res.status == "optimal"
        for name, v in res.voltage_mag.items():
            assert 0.9 < v < 1.15, f"bus {name}: |V| = {v:.3f} out of range"
        for name, th in res.voltage_angle.items():
            assert abs(th) < math.pi / 2, (
                f"bus {name}: θ = {math.degrees(th):.2f}° unreasonable"
            )
