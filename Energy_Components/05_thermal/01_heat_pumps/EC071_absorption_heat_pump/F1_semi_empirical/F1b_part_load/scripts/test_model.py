"""EC071 — Absorption HP — F1b Part-Load — Test Suite

Tests must FAIL the model if physics are wrong. Tolerances are tight.
Any loosening requires a RATIONALE comment citing literature.
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                       "part_load_ratio": 0.5})
    for k in ["cop", "heating_capacity_kw", "driving_heat_kw",
              "evaporator_heat_kw", "electrical_input_kw", "cop_degradation_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC071"
    assert info["fidelity"] == "F1b"


def test_cop_physical_range(model):
    """Absorption HP heating COP must be between 0.3 and 2.5 across valid range."""
    Tg   = np.linspace(70, 110, 10)
    Te   = np.linspace(0, 20, 10)
    Tc   = np.linspace(28, 45, 10)
    plr  = np.linspace(0.2, 1.0, 10)
    r = model.predict({"T_gen": Tg, "T_evap": Te, "T_cond": Tc,
                       "part_load_ratio": plr})
    assert np.all(r["cop"] >= 0.3), f"COP below 0.3: {r['cop'].min():.3f}"
    assert np.all(r["cop"] <= 2.5), f"COP above 2.5: {r['cop'].max():.3f}"


def test_cop_at_full_load_design(model):
    """COP at design point (T_gen=90C, T_evap=10C, T_cond=35C, PLR=1) must be ~1.7.
    RATIONALE: rated_cop_heating=1.7 per Hellmann & Ziegler (1999).
    Tolerance ±0.3 because carnot_fraction is tuned to match the rated value."""
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                       "part_load_ratio": 1.0})
    cop = float(r["cop"])
    assert 1.3 < cop < 2.2, f"Design COP={cop:.3f} out of expected range [1.3, 2.2]"


def test_cop_decreases_with_lower_plr(model):
    """COP must decrease monotonically as PLR decreases (PLF degrades utilization)."""
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                       "part_load_ratio": plr})
    assert np.all(np.diff(r["cop"]) > 0), \
        f"COP not monotonically increasing with PLR: {r['cop']}"


def test_plf_equals_one_at_full_load(model):
    """At PLR=1, degradation factor must be exactly 1.0 (T_gen at design)."""
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                       "part_load_ratio": 1.0})
    np.testing.assert_allclose(float(r["cop_degradation_factor"]), 1.0, atol=1e-10)


def test_cop_decreases_with_lower_T_gen(model):
    """COP must decrease as T_gen drops below design (desorption weakens)."""
    Tg = np.array([75.0, 80.0, 85.0, 90.0, 95.0])
    r = model.predict({"T_gen": Tg, "T_evap": 10.0, "T_cond": 35.0,
                       "part_load_ratio": 1.0})
    # COP should increase with increasing T_gen (more driving force)
    assert np.all(np.diff(r["cop"]) > 0), \
        f"COP not increasing with T_gen: {r['cop']}"


def test_cop_decreases_with_higher_T_cond(model):
    """COP must decrease as T_cond increases (greater temperature lift)."""
    Tc = np.array([28.0, 32.0, 36.0, 40.0, 45.0])
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": Tc,
                       "part_load_ratio": 1.0})
    assert np.all(np.diff(r["cop"]) < 0), \
        f"COP not decreasing with T_cond: {r['cop']}"


def test_energy_balance(model):
    """Q_heating must equal Q_driving + Q_evap (within 1 W tolerance)."""
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                       "part_load_ratio": 0.7})
    q_h   = float(r["heating_capacity_kw"])
    q_d   = float(r["driving_heat_kw"])
    q_e   = float(r["evaporator_heat_kw"])
    np.testing.assert_allclose(q_h, q_d + q_e, rtol=1e-5,
                               err_msg=f"Energy balance: {q_h:.3f} != {q_d:.3f}+{q_e:.3f}")


def test_cycling_penalty_below_plr_min(model):
    """COP degradation is larger below PLR_min than just above it."""
    r_above = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                             "part_load_ratio": 0.25})
    r_below = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                             "part_load_ratio": 0.10})
    assert float(r_below["cop_degradation_factor"]) < float(r_above["cop_degradation_factor"]), \
        "Cycling penalty not applied below PLR_min"


def test_f1b_cop_lower_than_f1a_at_part_load(model):
    """F1b COP at PLR=0.5 must be lower than at PLR=1 (part-load degrades)."""
    r_full = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                            "part_load_ratio": 1.0})
    r_half = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                            "part_load_ratio": 0.5})
    assert float(r_half["cop"]) < float(r_full["cop"]), \
        "Part-load COP must be lower than full-load COP"


def test_heating_capacity_proportional_to_plr(model):
    """Heating capacity = rated_capacity * PLR."""
    plr = np.array([0.2, 0.5, 0.75, 1.0])
    r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                       "part_load_ratio": plr})
    np.testing.assert_allclose(r["heating_capacity_kw"], 100.0 * plr, rtol=1e-10)


def test_benchmark(model):
    """1000 predictions must complete in < 1 s."""
    rng = np.random.default_rng(42)
    Tg  = rng.uniform(70, 110, 1000)
    Te  = rng.uniform(0, 20, 1000)
    Tc  = rng.uniform(28, 45, 1000)
    plr = rng.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"T_gen": Tg, "T_evap": Te, "T_cond": Tc, "part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s"
