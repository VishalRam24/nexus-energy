"""EC091 — Vapor Compression Chiller — F1b Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_chw": 6.7, "T_cw": 29.4, "PLR": 0.5})
    for k in ["cop", "cooling_capacity_kw", "electrical_input_kw", "iplv"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC091"
    assert info["fidelity"] == "F1b"


def test_cop_at_rated_conditions(model):
    """COP at rated conditions (PLR=1, T_cw=29.4C) should be near 5.5."""
    r = model.predict({"T_chw": 6.7, "T_cw": 29.4, "PLR": 1.0})
    cop = float(r["cop"])
    assert 4.0 < cop < 7.0, f"COP at rated = {cop:.2f}"


def test_cop_improves_at_lower_tcw(model):
    """Lower condenser water temp should improve COP."""
    r_hot = model.predict({"T_chw": 6.7, "T_cw": 35.0, "PLR": 1.0})
    r_cool = model.predict({"T_chw": 6.7, "T_cw": 20.0, "PLR": 1.0})
    assert float(r_cool["cop"]) > float(r_hot["cop"])


def test_cop_greater_than_one(model):
    plr = np.linspace(0.1, 1.0, 20)
    r = model.predict({"T_chw": 6.7, "T_cw": 35.0, "PLR": plr})
    assert np.all(r["cop"] > 1.0)


def test_iplv_greater_than_rated_cop(model):
    """IPLV should typically be higher than full-load COP due to part-load benefit."""
    iplv = model._model.iplv()
    assert iplv > model._model.COP_rated, f"IPLV={iplv:.2f} should exceed rated COP={model._model.COP_rated}"


def test_iplv_reasonable_range(model):
    """IPLV for a modern chiller should be in [5, 15]."""
    iplv = model._model.iplv()
    assert 5.0 < iplv < 15.0, f"IPLV = {iplv:.2f}"


def test_cooling_capacity_proportional(model):
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_chw": 6.7, "T_cw": 29.4, "PLR": plr})
    np.testing.assert_allclose(r["cooling_capacity_kw"], 500.0 * plr)


def test_electrical_input_increases_with_plr(model):
    """More cooling = more electricity."""
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_chw": 6.7, "T_cw": 29.4, "PLR": plr})
    assert np.all(np.diff(r["electrical_input_kw"]) > 0)


def test_energy_balance(model):
    """Q_cool / W = COP."""
    r = model.predict({"T_chw": 6.7, "T_cw": 29.4, "PLR": 0.7})
    q = float(r["cooling_capacity_kw"])
    w = float(r["electrical_input_kw"])
    cop = float(r["cop"])
    np.testing.assert_allclose(q / w, cop, rtol=1e-6)


def test_benchmark(model):
    T_chw = np.full(1000, 6.7)
    T_cw = np.random.uniform(18, 40, 1000)
    plr = np.random.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"T_chw": T_chw, "T_cw": T_cw, "PLR": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
