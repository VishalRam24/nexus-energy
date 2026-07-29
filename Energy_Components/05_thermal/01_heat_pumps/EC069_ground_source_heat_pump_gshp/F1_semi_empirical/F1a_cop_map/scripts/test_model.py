"""EC069 — GSHP — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

@pytest.fixture
def model():
    return ComponentModel()

def test_predict_keys(model):
    r = model.predict({"T_source": 10.0, "T_sink": 35.0})
    for k in ["cop", "heating_capacity_kw", "electrical_input_kw"]:
        assert k in r

def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC069"
    assert info["fidelity"] == "F1a"

def test_cop_greater_than_one(model):
    """GSHP COP must always exceed 1 (otherwise it's resistance heating)."""
    sources = np.linspace(0, 20, 50)
    r = model.predict({"T_source": sources, "T_sink": 35.0})
    assert np.all(r["cop"] > 1.0), "All COP values must be > 1"

def test_cop_at_rating_G10_W35(model):
    """COP at G10/W35 should be ~4.5 (rated condition per ASHRAE)."""
    r = model.predict({"T_source": 10.0, "T_sink": 35.0})
    cop = float(r["cop"])
    assert 4.0 < cop < 5.5, f"COP at G10/W35 = {cop:.2f}, expected 4.0-5.5"

def test_cop_exceeds_4_at_G10_W35(model):
    """Specific check: COP > 4 at G10/W35 (spec requirement)."""
    r = model.predict({"T_source": 10.0, "T_sink": 35.0})
    assert float(r["cop"]) > 4.0

def test_cop_higher_than_ashp_seasonal(model):
    """
    GSHP seasonal advantage: ground stays at ~10°C while air drops to -5°C in winter.
    GSHP COP at G10/W35 must exceed ASHP COP at A(-5)/W35.
    This is the real-world advantage of ground source over air source.
    """
    r_gshp = model.predict({"T_source": 10.0, "T_sink": 35.0})
    cop_gshp = float(r_gshp["cop"])

    # ASHP at -5°C outside air, same sink
    T_s_K_ashp = -5.0 + 273.15; T_k_K = 35.0 + 273.15
    cop_ashp_cold = 0.45 * T_k_K / (T_k_K - T_s_K_ashp)
    cop_ashp_cold = min(cop_ashp_cold, 15.0)

    assert cop_gshp > cop_ashp_cold, (
        f"GSHP COP at G10/W35 ({cop_gshp:.2f}) must exceed "
        f"ASHP COP at A-5/W35 ({cop_ashp_cold:.2f}) — seasonal advantage test"
    )

def test_cop_decreases_with_temp_lift(model):
    """COP decreases as temperature lift (T_sink - T_source) increases."""
    sources = np.array([15.0, 10.0, 5.0, 0.0])
    r = model.predict({"T_source": sources, "T_sink": 35.0})
    assert np.all(np.diff(r["cop"]) < 0), "COP must decrease with increasing temperature lift"

def test_cop_increases_with_source_temp(model):
    """Higher ground temperature → higher COP (smaller temperature lift)."""
    sources = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    r = model.predict({"T_source": sources, "T_sink": 35.0})
    assert np.all(np.diff(r["cop"]) > 0), "COP must increase as source temperature rises"

def test_heating_capacity_at_full_load(model):
    """At PLR=1, heating capacity should equal rated capacity (15 kW)."""
    r = model.predict({"T_source": 10.0, "T_sink": 35.0, "part_load_ratio": 1.0})
    assert float(r["heating_capacity_kw"]) == pytest.approx(15.0, rel=0.01)

def test_energy_balance(model):
    """Heating output ≈ COP × electrical input (ignoring small aux power)."""
    r = model.predict({"T_source": 10.0, "T_sink": 35.0})
    q = float(r["heating_capacity_kw"])
    w = float(r["electrical_input_kw"])
    cop = float(r["cop"])
    # q/w <= cop (aux power slightly inflates W)
    assert q / w <= cop + 0.2

def test_electrical_increases_with_temp_lift(model):
    """More electrical input needed at higher temperature lifts."""
    sources = np.array([15.0, 10.0, 5.0, 0.0])
    r = model.predict({"T_source": sources, "T_sink": 45.0})
    assert np.all(np.diff(r["electrical_input_kw"]) > 0)

def test_benchmark(model):
    """1000 predictions must complete in <1 second."""
    Ts = np.random.uniform(0, 20, 1000)
    Tk = np.random.uniform(30, 55, 1000)
    start = time.perf_counter()
    model.predict({"T_source": Ts, "T_sink": Tk})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
