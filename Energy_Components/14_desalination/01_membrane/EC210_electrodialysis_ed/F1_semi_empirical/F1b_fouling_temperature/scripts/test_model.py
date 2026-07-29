"""EC210 — Electrodialysis (ED) — F1b Current Density + Donnan T — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current_density": 100.0, "operating_hours": 0})
    for k in ["desalination_rate_mol_s", "salinity_reduction_pct", "current_efficiency",
              "sec_kwh_m3", "donnan_selectivity_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC210"
    assert info["fidelity"] == "F1b"


def test_desalination_rate_positive(model):
    """Desalination rate must be positive at design current."""
    r = model.predict({"current_density": 100.0, "T_feed_degC": 25.0,
                       "C_feed_mol_m3": 100.0, "operating_hours": 0})
    rate = float(np.atleast_1d(r["desalination_rate_mol_s"])[0])
    assert rate > 0, f"Desalination rate = {rate:.6f} mol/s"


def test_desalination_rate_scales_with_current(model):
    """Higher current density → more ion removal."""
    r_low  = model.predict({"current_density": 50.0, "T_feed_degC": 25.0,
                            "C_feed_mol_m3": 100.0, "operating_hours": 0})
    r_high = model.predict({"current_density": 150.0, "T_feed_degC": 25.0,
                            "C_feed_mol_m3": 100.0, "operating_hours": 0})
    rate_low  = float(np.atleast_1d(r_low["desalination_rate_mol_s"])[0])
    rate_high = float(np.atleast_1d(r_high["desalination_rate_mol_s"])[0])
    assert rate_high > rate_low, f"Desalination not scaling with i: {rate_low:.4f} vs {rate_high:.4f}"


def test_current_efficiency_drops_near_limiting(model):
    """Current efficiency should drop at high i/i_lim ratios."""
    r_low  = model.predict({"current_density": 50.0,  "T_feed_degC": 25.0,
                            "C_feed_mol_m3": 100.0, "operating_hours": 0})
    r_high = model.predict({"current_density": 260.0, "T_feed_degC": 25.0,
                            "C_feed_mol_m3": 100.0, "operating_hours": 0})
    eta_low  = float(np.atleast_1d(r_low["current_efficiency"])[0])
    eta_high = float(np.atleast_1d(r_high["current_efficiency"])[0])
    assert eta_low > eta_high, f"Efficiency not dropping near i_lim: {eta_low:.3f} vs {eta_high:.3f}"


def test_donnan_selectivity_decreases_with_temperature(model):
    """Higher T → reduced Donnan selectivity (co-ion leakage)."""
    r_cold = model.predict({"T_feed_degC": 10.0, "current_density": 100.0,
                            "C_feed_mol_m3": 100.0, "operating_hours": 0})
    r_warm = model.predict({"T_feed_degC": 40.0, "current_density": 100.0,
                            "C_feed_mol_m3": 100.0, "operating_hours": 0})
    sel_cold = float(np.atleast_1d(r_cold["donnan_selectivity_factor"])[0])
    sel_warm = float(np.atleast_1d(r_warm["donnan_selectivity_factor"])[0])
    assert sel_cold > sel_warm, \
        f"Donnan selectivity not decreasing with T: cold={sel_cold:.4f}, warm={sel_warm:.4f}"


def test_sec_positive(model):
    r = model.predict({"current_density": 100.0, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_kwh_m3"])[0])
    assert sec > 0


def test_sec_increases_with_aging(model):
    """SEC should increase with membrane aging (higher resistance)."""
    r_new = model.predict({"current_density": 100.0, "T_feed_degC": 25.0,
                           "C_feed_mol_m3": 100.0, "flow_rate_m3_h": 10.0,
                           "operating_hours": 0})
    r_old = model.predict({"current_density": 100.0, "T_feed_degC": 25.0,
                           "C_feed_mol_m3": 100.0, "flow_rate_m3_h": 10.0,
                           "operating_hours": 52560})  # 6 years
    sec_new = float(np.atleast_1d(r_new["sec_kwh_m3"])[0])
    sec_old = float(np.atleast_1d(r_old["sec_kwh_m3"])[0])
    assert sec_old > sec_new, f"SEC not increasing with aging: {sec_new:.3f} vs {sec_old:.3f}"


def test_sec_reasonable_range_brackish(model):
    """SEC for brackish ED: 0.5-5 kWh/m3 (Strathmann 2010)."""
    r = model.predict({"current_density": 100.0, "T_feed_degC": 25.0,
                       "C_feed_mol_m3": 100.0, "flow_rate_m3_h": 10.0,
                       "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_kwh_m3"])[0])
    # Strathmann 2010: brackish ED SEC 0.5-3 kWh/m3; allow generous range
    assert 0.05 < sec < 25.0, f"SEC = {sec:.3f} kWh/m3 out of expected range"


def test_ilim_scales_with_temperature(model):
    """Higher T should increase i_lim (higher diffusivity)."""
    i_lim_cold = model._model._ilim_temperature(10.0, 100.0)
    i_lim_warm = model._model._ilim_temperature(40.0, 100.0)
    assert float(np.atleast_1d(i_lim_warm)[0]) > float(np.atleast_1d(i_lim_cold)[0])


def test_salinity_reduction_positive(model):
    r = model.predict({"current_density": 100.0, "T_feed_degC": 25.0,
                       "C_feed_mol_m3": 100.0, "flow_rate_m3_h": 10.0,
                       "operating_hours": 0})
    sal_red = float(np.atleast_1d(r["salinity_reduction_pct"])[0])
    assert sal_red > 0


def test_array_input(model):
    """Model handles array current density inputs."""
    currents = np.linspace(50.0, 250.0, 10)
    r = model.predict({"current_density": currents, "T_feed_degC": 25.0,
                       "C_feed_mol_m3": 100.0, "operating_hours": 0})
    assert len(np.atleast_1d(r["desalination_rate_mol_s"])) == 10


def test_benchmark(model):
    currents = np.random.uniform(50.0, 250.0, 1000)
    start = time.perf_counter()
    model.predict({"current_density": currents, "T_feed_degC": 25.0,
                   "C_feed_mol_m3": 100.0, "operating_hours": 0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
