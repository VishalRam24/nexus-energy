"""EC072 — CO2 Transcritical HP — F1b Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0,
                       "part_load_ratio": 0.5})
    for k in ["cop", "heating_capacity_kw", "electrical_input_kw",
              "cop_degradation_factor", "optimum_P_high_bar"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC072"
    assert info["fidelity"] == "F1b"


def test_cop_range(model):
    """COP must be in [0.8, 8.0] over valid operating range."""
    plr = np.linspace(0.1, 1.0, 20)
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0,
                       "part_load_ratio": plr})
    assert np.all(r["cop"] >= 0.8)
    assert np.all(r["cop"] <= 8.0)


def test_cop_at_rated_conditions(model):
    """COP at design (T_evap=0C, T_w_in=15C, T_w_out=65C, PLR=1) should be ~3.5.
    RATIONALE: rated_cop=3.5 per Sarkar et al. (2004). Tolerance ±1.0 for semi-empirical."""
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0,
                       "part_load_ratio": 1.0})
    cop = float(r["cop"])
    assert 2.0 < cop < 5.0, f"COP at rated = {cop:.3f}"


def test_cop_decreases_with_lower_plr(model):
    """COP must decrease as PLR decreases."""
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0,
                       "part_load_ratio": plr})
    assert np.all(np.diff(r["cop"]) > 0), f"COP vs PLR not monotonic: {r['cop']}"


def test_plf_equals_one_at_full_load_design_twin(model):
    """At PLR=1 and T_water_in=design, degradation factor must be 1.0."""
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0,
                       "part_load_ratio": 1.0})
    np.testing.assert_allclose(float(r["cop_degradation_factor"]), 1.0, atol=1e-10)


def test_cop_decreases_with_higher_T_water_in(model):
    """COP must decrease as T_water_in increases above design value.
    RATIONALE: Higher inlet T shrinks gas-cooler dT, reduces heat extraction
    (Sarkar et al., 2004, Fig.4 — CO2 gas-cooler sensitivity)."""
    Twin_arr = np.array([15.0, 25.0, 35.0, 45.0])
    r = model.predict({"T_evap": 0.0, "T_water_in": Twin_arr,
                       "T_water_out": 65.0, "part_load_ratio": 1.0})
    assert np.all(np.diff(r["cop"]) < 0), \
        f"COP should decrease with increasing T_water_in: {r['cop']}"


def test_cop_decreases_with_lower_T_evap(model):
    """COP must decrease as T_evap decreases (larger lift)."""
    Te_arr = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    r = model.predict({"T_evap": Te_arr, "T_water_in": 15.0,
                       "T_water_out": 65.0, "part_load_ratio": 1.0})
    assert np.all(np.diff(r["cop"]) > 0), \
        f"COP should increase with T_evap: {r['cop']}"


def test_cycling_penalty_below_plr_min(model):
    r_above = model.predict({"T_evap": 0.0, "T_water_in": 15.0,
                             "T_water_out": 65.0, "part_load_ratio": 0.20})
    r_below = model.predict({"T_evap": 0.0, "T_water_in": 15.0,
                             "T_water_out": 65.0, "part_load_ratio": 0.05})
    assert float(r_below["cop_degradation_factor"]) < float(r_above["cop_degradation_factor"])


def test_optimum_pressure_in_transcritical_regime(model):
    """Optimum P_high must be above CO2 critical pressure (73.8 bar) at typical T_gc_out."""
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0,
                       "T_water_out": 65.0, "part_load_ratio": 1.0})
    P_opt = float(r["optimum_P_high_bar"])
    assert P_opt > 73.8, f"P_high_opt={P_opt:.1f}bar should be > 73.8 bar (transcritical)"


def test_heating_capacity_proportional_to_plr(model):
    plr = np.array([0.2, 0.5, 0.75, 1.0])
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0,
                       "T_water_out": 65.0, "part_load_ratio": plr})
    np.testing.assert_allclose(r["heating_capacity_kw"], 50.0 * plr, rtol=1e-10)


def test_benchmark(model):
    rng = np.random.default_rng(0)
    Te   = rng.uniform(-15, 15, 1000)
    Twin = rng.uniform(5, 45, 1000)
    Twout= rng.uniform(45, 85, 1000)
    plr  = rng.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"T_evap": Te, "T_water_in": Twin, "T_water_out": Twout,
                   "part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
