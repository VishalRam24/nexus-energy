"""EC092 — Absorption Chiller — F1b Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_hot": 90.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": 0.5})
    for k in ["cop", "cooling_capacity_kw", "heat_input_kw"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC092"
    assert info["fidelity"] == "F1b"


def test_cop_at_rated_conditions(model):
    """COP at rated should be ~0.70."""
    r = model.predict({"T_hot": 90.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": 1.0})
    cop = float(r["cop"])
    assert 0.5 < cop < 0.80, f"COP at rated = {cop:.3f}"


def test_cop_capped_at_080(model):
    """Single-effect LiBr COP cannot exceed 0.80."""
    plr = np.linspace(0.15, 1.0, 50)
    r = model.predict({"T_hot": 110.0, "T_cw": 25.0, "T_chw": 7.0, "PLR": plr})
    assert np.all(r["cop"] <= 0.80 + 1e-10)


def test_cop_increases_with_higher_thot(model):
    """Higher generator temperature improves COP."""
    r_low = model.predict({"T_hot": 80.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": 1.0})
    r_high = model.predict({"T_hot": 100.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": 1.0})
    assert float(r_high["cop"]) > float(r_low["cop"])


def test_cop_nonnegative(model):
    plr = np.linspace(0.15, 1.0, 50)
    r = model.predict({"T_hot": 80.0, "T_cw": 35.0, "T_chw": 7.0, "PLR": plr})
    assert np.all(r["cop"] >= 0.0)


def test_heat_input_exceeds_cooling(model):
    """For absorption chiller (COP<1): Q_heat > Q_cool always."""
    plr = np.linspace(0.15, 1.0, 20)
    r = model.predict({"T_hot": 90.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": plr})
    assert np.all(r["heat_input_kw"] > r["cooling_capacity_kw"])


def test_cooling_capacity_scales_with_plr(model):
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_hot": 90.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": plr})
    np.testing.assert_allclose(r["cooling_capacity_kw"], 500.0 * plr, rtol=1e-6)


def test_part_load_curve_shape(model):
    """COP should peak at moderate PLR for absorption systems."""
    plr = np.linspace(0.15, 1.0, 100)
    r = model.predict({"T_hot": 90.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": plr})
    cop = np.asarray(r["cop"])
    peak_plr = plr[np.argmax(cop)]
    # Peak should be somewhere around 0.5-1.0
    assert 0.3 < peak_plr < 1.01, f"Peak COP at PLR={peak_plr:.2f}"


def test_benchmark(model):
    n = 1000
    T_hot = np.random.uniform(75, 110, n)
    T_cw = np.random.uniform(25, 40, n)
    T_chw = np.full(n, 7.0)
    plr = np.random.uniform(0.15, 1.0, n)
    start = time.perf_counter()
    model.predict({"T_hot": T_hot, "T_cw": T_cw, "T_chw": T_chw, "PLR": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
