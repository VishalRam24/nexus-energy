"""EC195 — Ammonia Synthesis (Haber-Bosch) — F1b Part-Load Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0})
    for k in ["nh3_production_mol_s", "single_pass_conversion", "recycle_ratio",
              "energy_kwh_per_ton", "purge_fraction"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC195"
    assert info["fidelity"] == "F1b"


def test_conversion_at_design(model):
    """At PLR=1.0, T=450C, P=200bar: X_sp should be ~0.15."""
    r = model.predict({"PLR": 1.0})
    X = float(np.atleast_1d(r["single_pass_conversion"])[0])
    assert abs(X - 0.15) < 0.03, f"X_sp at design = {X:.4f}, expected ~0.15"


def test_conversion_decreases_with_part_load(model):
    """Lower PLR -> lower effective pressure -> lower conversion."""
    PLRs = [0.3, 0.5, 0.7, 1.0]
    Xs = []
    for plr in PLRs:
        r = model.predict({"PLR": plr})
        Xs.append(float(np.atleast_1d(r["single_pass_conversion"])[0]))
    assert all(Xs[i] <= Xs[i + 1] + 1e-6 for i in range(len(Xs) - 1)), \
        f"Conversion not non-decreasing with PLR: {Xs}"


def test_recycle_ratio_increases_at_part_load(model):
    """Lower conversion -> higher recycle ratio."""
    PLRs = [0.3, 0.5, 0.7, 1.0]
    RRs = []
    for plr in PLRs:
        r = model.predict({"PLR": plr})
        RRs.append(float(np.atleast_1d(r["recycle_ratio"])[0]))
    assert all(RRs[i] >= RRs[i + 1] - 1e-6 for i in range(len(RRs) - 1)), \
        f"Recycle ratio not non-increasing at part-load: {RRs}"


def test_recycle_ratio_at_design(model):
    """At X_sp=0.15: R = 1/0.15 - 1 = 5.67."""
    r = model.predict({"PLR": 1.0})
    RR = float(np.atleast_1d(r["recycle_ratio"])[0])
    assert 4.0 < RR < 8.0, f"Recycle ratio = {RR:.2f}, expected ~5.67"


def test_energy_increases_at_part_load(model):
    """Specific energy should increase at part-load."""
    PLRs = [0.3, 0.5, 0.7, 1.0]
    Es = []
    for plr in PLRs:
        r = model.predict({"PLR": plr})
        Es.append(float(np.atleast_1d(r["energy_kwh_per_ton"])[0]))
    assert Es[0] > Es[-1], f"Energy not higher at part-load: {Es}"


def test_nh3_production_positive(model):
    """NH3 production must be positive."""
    r = model.predict({"PLR": 0.5})
    nh3 = float(np.atleast_1d(r["nh3_production_mol_s"])[0])
    assert nh3 > 0


def test_nh3_scales_with_n2_flow(model):
    """NH3 production should scale linearly with N2 feed."""
    r1 = model.predict({"n2_flow_mol_s": 1.0, "PLR": 1.0})
    r2 = model.predict({"n2_flow_mol_s": 2.0, "PLR": 1.0})
    nh3_1 = float(np.atleast_1d(r1["nh3_production_mol_s"])[0])
    nh3_2 = float(np.atleast_1d(r2["nh3_production_mol_s"])[0])
    assert abs(nh3_2 / nh3_1 - 2.0) < 0.01


def test_purge_fraction_positive(model):
    """Purge fraction must be positive and small."""
    r = model.predict({"PLR": 1.0})
    pf = float(np.atleast_1d(r["purge_fraction"])[0])
    assert 0.0 < pf < 0.2, f"Purge fraction = {pf:.4f}"


def test_purge_increases_at_part_load(model):
    """Purge fraction should increase at lower PLR."""
    r1 = model.predict({"PLR": 1.0})
    r2 = model.predict({"PLR": 0.3})
    pf1 = float(np.atleast_1d(r1["purge_fraction"])[0])
    pf2 = float(np.atleast_1d(r2["purge_fraction"])[0])
    assert pf2 > pf1


def test_array_input(model):
    """Model should handle array PLR inputs."""
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"PLR": PLRs})
    assert len(np.atleast_1d(r["single_pass_conversion"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLRs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
