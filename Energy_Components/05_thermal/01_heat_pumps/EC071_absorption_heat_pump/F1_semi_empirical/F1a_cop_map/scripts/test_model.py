"""EC071 — Absorption Heat Pump — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0})
    for k in ["cop", "heating_capacity_kw", "driving_heat_kw", "evaporator_heat_kw", "electrical_input_kw"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC071"
    assert info["fidelity"] == "F1a"


def test_cop_positive(model):
    """Heating COP must be > 0 across the operating envelope."""
    Tg = np.linspace(70, 110, 10)
    r = model.predict({"T_gen": Tg, "T_evap": 10.0, "T_cond": 35.0})
    assert np.all(r["cop"] > 0.0)


def test_cop_above_unity(model):
    """A Type-I absorption HP heats by Q_drive + Q_evap, so COP_h > 1."""
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0})
    assert float(r["cop"]) > 1.0, f"COP={float(r['cop']):.2f} should be > 1"


def test_cop_in_expected_range(model):
    """Single-effect AHP COP_h typically ~1.5–1.8 at design conditions."""
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0})
    cop = float(r["cop"])
    assert 1.3 < cop < 2.0, f"COP_h at design = {cop:.2f}, expected 1.3–2.0"


def test_cop_increases_with_T_gen(model):
    """Higher generator temperature improves drive engine efficiency, COP rises."""
    Tg = np.array([70.0, 80.0, 90.0, 100.0, 110.0])
    r = model.predict({"T_gen": Tg, "T_evap": 10.0, "T_cond": 35.0})
    assert np.all(np.diff(r["cop"]) > 0)


def test_cop_decreases_with_T_cond(model):
    """Higher rejection (cond/abs) temperature reduces COP."""
    Tc = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": Tc})
    assert np.all(np.diff(r["cop"]) < 0)


def test_energy_balance(model):
    """Q_heating ≈ Q_drive + Q_evap (first law of the AHP)."""
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0})
    qh   = float(r["heating_capacity_kw"])
    qg   = float(r["driving_heat_kw"])
    qe   = float(r["evaporator_heat_kw"])
    assert abs(qh - (qg + qe)) < 0.05, f"Energy imbalance: {qh:.2f} vs {qg+qe:.2f}"


def test_capacity_scales_with_plr(model):
    r1 = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0, "part_load_ratio": 1.0})
    r2 = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0, "part_load_ratio": 0.5})
    assert abs(float(r2["heating_capacity_kw"]) - 0.5 * float(r1["heating_capacity_kw"])) < 1e-6


def test_benchmark(model):
    Tg = np.random.uniform(70, 110, 1000)
    Te = np.random.uniform(2, 20, 1000)
    Tc = np.random.uniform(28, 45, 1000)
    start = time.perf_counter()
    model.predict({"T_gen": Tg, "T_evap": Te, "T_cond": Tc})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
