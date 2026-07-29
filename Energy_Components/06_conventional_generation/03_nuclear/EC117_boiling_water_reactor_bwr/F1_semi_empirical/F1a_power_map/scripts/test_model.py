"""EC117 — BWR — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0})
    for k in ["electric_power_mw", "thermal_power_mw", "efficiency", "steam_mass_flow_kgs"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC117"
    assert info["fidelity"] == "F1a"


def test_full_power_thermal(model):
    r = model.predict({"part_load_ratio": 1.0})
    assert abs(float(r["thermal_power_mw"]) - 3300.0) < 1.0


def test_full_power_electric(model):
    r = model.predict({"part_load_ratio": 1.0})
    Pe = float(r["electric_power_mw"])
    # 3300 * 0.335 * 0.99 = 1094 MW_e
    assert 1080 < Pe < 1110, f"Pe={Pe:.1f}"


def test_efficiency_realistic(model):
    """BWR direct-cycle eta ~ 0.33-0.34."""
    r = model.predict({"part_load_ratio": 1.0})
    eta = float(r["efficiency"])
    assert 0.32 < eta < 0.35, f"eta={eta:.4f}"


def test_p_electric_equals_p_thermal_times_eta(model):
    PLR = np.linspace(0.6, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR})
    assert np.allclose(r["electric_power_mw"], r["thermal_power_mw"] * r["efficiency"], rtol=1e-10)


def test_plr_min_enforced(model):
    """PLR below 0.6 clamped to 0.6."""
    r_low = model.predict({"part_load_ratio": 0.2})
    r_min = model.predict({"part_load_ratio": 0.6})
    assert abs(float(r_low["thermal_power_mw"]) - float(r_min["thermal_power_mw"])) < 1.0


def test_steam_flow_increases_with_power(model):
    PLR = np.linspace(0.6, 1.0, 10)
    r = model.predict({"part_load_ratio": PLR})
    assert np.all(np.diff(r["steam_mass_flow_kgs"]) > 0)


def test_steam_flow_at_full_power(model):
    """Steam flow at full power should be order of 1900 kg/s."""
    r = model.predict({"part_load_ratio": 1.0})
    m = float(r["steam_mass_flow_kgs"])
    assert 1700 < m < 2100, f"m_steam={m:.0f}"


def test_power_scales_with_plr(model):
    PLR = np.linspace(0.6, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR})
    ratio = r["thermal_power_mw"] / PLR
    assert np.allclose(ratio, ratio[0], rtol=1e-9)


def test_pe_below_pth(model):
    """Electric power must always be less than thermal power."""
    PLR = np.linspace(0.6, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR})
    assert np.all(r["electric_power_mw"] < r["thermal_power_mw"])


def test_benchmark(model):
    PLR = np.random.uniform(0.6, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": PLR})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
