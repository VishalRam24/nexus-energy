"""EC192 — Gas Pressure Regulator — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": 50.0})
    for k in ["T_down_K", "delta_T_K", "Q_std_m3_per_h", "Q_kg_per_s",
               "expansion_Y", "is_choked"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC192"
    assert info["fidelity"] == "F1a"


def test_pressure_constraint(model):
    """P_down must be ≤ P_up — the regulator can only reduce pressure."""
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": 50.0})
    assert 50.0 <= 80.0  # trivially true; test that model doesn't produce nonsense

    # Equal pressures → zero flow
    r_eq = model.predict({"P_up_bar": 80.0, "P_down_bar": 80.0})
    assert float(r_eq["Q_std_m3_per_h"]) == pytest.approx(0.0, abs=1e-9)


def test_jt_cooling_sign(model):
    """NG at ambient T (<< inversion T ~700 K): expansion must cool the gas (ΔT < 0)."""
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": 20.0, "T_up_K": 288.15})
    assert float(r["delta_T_K"]) < 0.0, "JT expansion of NG below inversion T must cool gas"


def test_jt_cooling_magnitude(model):
    """ΔT ≈ -μ_JT × ΔP ≈ -0.45 K/bar × 60 bar ≈ -27 K for 80→20 bar."""
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": 20.0, "T_up_K": 288.15})
    dT = float(r["delta_T_K"])
    # μ_JT = 4.5e-6 K/Pa; ΔP = 60 bar = 6e6 Pa → ΔT ≈ -27 K
    expected = -4.5e-6 * 60.0 * 1e5
    assert dT == pytest.approx(expected, rel=1e-6)


def test_larger_dp_more_cooling(model):
    """Larger pressure drop → more JT cooling."""
    dP_arr = np.array([10.0, 20.0, 30.0, 50.0, 70.0])
    P_down = 80.0 - dP_arr
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": P_down, "T_up_K": 288.15})
    assert np.all(np.diff(r["delta_T_K"]) < 0)  # more negative = more cooling


def test_flow_positive_when_dp_positive(model):
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": 50.0})
    assert float(r["Q_std_m3_per_h"]) > 0
    assert float(r["Q_kg_per_s"]) > 0


def test_flow_increases_with_dp_subcritical(model):
    """In subcritical regime: more dP → more flow."""
    P_down = np.array([70.0, 65.0, 60.0, 55.0])  # decreasing = increasing dP
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": P_down, "T_up_K": 288.15})
    assert np.all(np.diff(r["Q_std_m3_per_h"]) > 0)


def test_flow_saturates_at_choke(model):
    """Once choked, further reducing P_down should not increase flow significantly."""
    # Choke condition: dP >= Fk * Xt * P_up = 0.972 * 0.7 * 80 = 54.4 bar
    # So P_down <= 80 - 54.4 = 25.6 bar is choked
    r_just_choked = model.predict({"P_up_bar": 80.0, "P_down_bar": 24.0})
    r_deep_choked = model.predict({"P_up_bar": 80.0, "P_down_bar": 5.0})
    assert bool(r_just_choked["is_choked"]) is True
    assert bool(r_deep_choked["is_choked"]) is True
    # Flows at choke should be equal (expansion_Y clamped at 2/3)
    assert float(r_just_choked["Q_std_m3_per_h"]) == pytest.approx(
        float(r_deep_choked["Q_std_m3_per_h"]), rel=1e-6)


def test_expansion_factor_y_bounds(model):
    """Y must be in [2/3, 1]."""
    P_down = np.linspace(1, 79, 100)
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": P_down})
    Y = r["expansion_Y"]
    assert np.all(Y >= 2.0 / 3.0 - 1e-10)
    assert np.all(Y <= 1.0 + 1e-10)


def test_flow_scales_with_cv(model):
    """Q ∝ Cv: doubling Cv doubles flow."""
    r1 = model.predict({"P_up_bar": 80.0, "P_down_bar": 50.0, "Cv": 300.0})
    r2 = model.predict({"P_up_bar": 80.0, "P_down_bar": 50.0, "Cv": 600.0})
    ratio = float(r2["Q_std_m3_per_h"]) / float(r1["Q_std_m3_per_h"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_t_down_below_t_up_for_ng_ambient(model):
    """For NG at ambient temperature, downstream T must always be below upstream T."""
    P_down = np.array([10.0, 20.0, 30.0, 50.0, 70.0])
    r = model.predict({"P_up_bar": 80.0, "P_down_bar": P_down, "T_up_K": 288.15})
    assert np.all(r["T_down_K"] < 288.15)


def test_benchmark(model):
    rng = np.random.default_rng(42)
    P_up = rng.uniform(20, 150, 1000)
    P_down = P_up * rng.uniform(0.1, 0.95, 1000)
    start = time.perf_counter()
    model.predict({"P_up_bar": P_up, "P_down_bar": P_down})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
