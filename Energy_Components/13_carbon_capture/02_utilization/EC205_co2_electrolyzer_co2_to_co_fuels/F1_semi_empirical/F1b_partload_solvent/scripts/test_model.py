"""EC205 — CO2 Electrolyzer — F1b Part-Load Degradation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    for k in ["co_production_rate_g_h", "sec_kwh_t_co", "cell_voltage_V",
              "faradaic_efficiency", "fe_relative_pct"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC205"
    assert info["fidelity"] == "F1b"


def test_fe_100pct_at_start(model):
    """Fresh electrode: FE relative should be 100%."""
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    fe_pct = float(np.atleast_1d(r["fe_relative_pct"])[0])
    assert abs(fe_pct - 100.0) < 1e-6, f"FE at t=0 = {fe_pct:.2f}%"


def test_fe_degrades_with_time(model):
    """FE decreases with operating hours."""
    hours_list = [0, 5000, 10000, 30000]
    fes = [float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": h})["faradaic_efficiency"])[0])
           for h in hours_list]
    assert all(fes[i] >= fes[i + 1] - 1e-6 for i in range(len(fes) - 1)), \
        f"FE not declining: {fes}"


def test_fe_1pct_per_1000h(model):
    """At 1000h: FE relative should be ~99%."""
    r = model.predict({"PLR": 1.0, "operating_hours": 1000})
    fe_pct = float(np.atleast_1d(r["fe_relative_pct"])[0])
    assert abs(fe_pct - 99.0) < 0.5, f"FE at 1000h = {fe_pct:.2f}%, expected 99%"


def test_cell_voltage_design(model):
    """Cell voltage at PLR=1.0 should be ~2.4 V."""
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    V = float(np.atleast_1d(r["cell_voltage_V"])[0])
    assert abs(V - 2.4) < 0.1, f"Cell voltage at design = {V:.3f} V"


def test_voltage_increases_at_low_plr(model):
    """Lower PLR → higher specific voltage (a + b/PLR with a<1, b>0)."""
    V_full = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 0})["cell_voltage_V"])[0])
    V_part = float(np.atleast_1d(model.predict({"PLR": 0.5, "operating_hours": 0})["cell_voltage_V"])[0])
    assert V_part > V_full, f"Voltage at part-load {V_part:.3f} not > full-load {V_full:.3f}"


def test_sec_increases_with_degradation(model):
    """SEC should increase as FE declines."""
    sec_fresh = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 0})["sec_kwh_t_co"])[0])
    sec_aged = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 40000})["sec_kwh_t_co"])[0])
    assert sec_aged >= sec_fresh - 1e-6, f"SEC fresh={sec_fresh:.1f}, aged={sec_aged:.1f}"


def test_co_production_positive(model):
    r = model.predict({"PLR": 0.5, "operating_hours": 0})
    co = float(np.atleast_1d(r["co_production_rate_g_h"])[0])
    assert co > 0


def test_co_scales_with_plr(model):
    """CO production scales with PLR (higher current)."""
    co_1 = float(np.atleast_1d(model.predict({"PLR": 0.5, "operating_hours": 0})["co_production_rate_g_h"])[0])
    co_2 = float(np.atleast_1d(model.predict({"PLR": 1.0, "operating_hours": 0})["co_production_rate_g_h"])[0])
    assert co_2 > co_1


def test_sec_reasonable_range(model):
    """SEC at design: 200-1000 kWh/tCO."""
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_kwh_t_co"])[0])
    assert 200.0 < sec < 1000.0, f"SEC = {sec:.1f} kWh/tCO"


def test_array_input(model):
    PLRs = np.linspace(0.25, 1.0, 10)
    r = model.predict({"PLR": PLRs, "operating_hours": 0})
    assert len(np.atleast_1d(r["sec_kwh_t_co"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.25, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLRs, "operating_hours": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
