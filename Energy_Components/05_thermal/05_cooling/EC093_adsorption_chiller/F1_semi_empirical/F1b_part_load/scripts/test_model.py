"""EC093 — Adsorption Chiller — F1b Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": 0.5})
    for k in ["cop", "cooling_power_kw", "driving_heat_kw",
              "heat_rejection_kw", "electrical_input_kw", "cop_degradation_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC093"
    assert info["fidelity"] == "F1b"


def test_cop_range(model):
    """Adsorption chiller COP must be between 0.01 and 0.85."""
    plr = np.linspace(0.1, 1.0, 20)
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": plr})
    assert np.all(r["cop"] >= 0.01)
    assert np.all(r["cop"] <= 0.85)


def test_cop_at_design_full_load(model):
    """COP at design conditions (PLR=1) must be ~0.55.
    RATIONALE: rated_cop=0.55 per Saha et al. (1995).
    # RATIONALE: Tolerance ±0.15 because carnot_fraction is semi-empirical."""
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": 1.0})
    cop = float(r["cop"])
    assert 0.30 < cop < 0.75, f"Design COP={cop:.3f} outside [0.30, 0.75]"


def test_cop_decreases_with_lower_plr(model):
    """COP must decrease as PLR decreases (incomplete adsorption at short cycles)."""
    plr = np.array([0.3, 0.5, 0.75, 1.0])
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": plr})
    assert np.all(np.diff(r["cop"]) > 0), \
        f"COP should increase with PLR: {r['cop']}"


def test_plf_equals_one_at_full_load(model):
    """At PLR=1, cop_degradation_factor must be 1.0."""
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": 1.0})
    np.testing.assert_allclose(float(r["cop_degradation_factor"]), 1.0, atol=1e-10)


def test_plf_less_than_one_at_part_load(model):
    """PLF < 1 at PLR < 1."""
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": 0.5})
    assert float(r["cop_degradation_factor"]) < 1.0


def test_cop_f1b_lower_than_f1a_style_full(model):
    """F1b COP at PLR=0.4 must be lower than at PLR=1 (part-load degrades)."""
    r_full = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                            "part_load_ratio": 1.0})
    r_half = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                            "part_load_ratio": 0.4})
    assert float(r_half["cop"]) < float(r_full["cop"])


def test_cop_decreases_with_lower_T_hot(model):
    """COP must decrease as T_hot decreases (less driving force)."""
    T_hot_arr = np.array([65.0, 70.0, 75.0, 80.0, 85.0])
    r = model.predict({"T_hot": T_hot_arr, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": 1.0})
    assert np.all(np.diff(r["cop"]) > 0), \
        f"COP should increase with T_hot: {r['cop']}"


def test_energy_balance(model):
    """Q_cool + Q_drive = Q_reject (first law)."""
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": 0.7})
    q_cool = float(r["cooling_power_kw"])
    q_driv = float(r["driving_heat_kw"])
    q_rej  = float(r["heat_rejection_kw"])
    np.testing.assert_allclose(q_cool + q_driv, q_rej, rtol=1e-5)


def test_cycling_penalty_below_plr_min(model):
    r_above = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                             "part_load_ratio": 0.35})
    r_below = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                             "part_load_ratio": 0.10})
    assert float(r_below["cop_degradation_factor"]) < float(r_above["cop_degradation_factor"])


def test_cooling_capacity_proportional_to_plr(model):
    plr = np.array([0.3, 0.5, 0.75, 1.0])
    r = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                       "part_load_ratio": plr})
    np.testing.assert_allclose(r["cooling_power_kw"], 50.0 * plr, rtol=1e-10)


def test_benchmark(model):
    rng = np.random.default_rng(99)
    Th  = rng.uniform(60, 90, 1000)
    Tc  = rng.uniform(24, 38, 1000)
    Tch = rng.uniform(7, 18, 1000)
    plr = rng.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"T_hot": Th, "T_cool": Tc, "T_chilled": Tch,
                   "part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
