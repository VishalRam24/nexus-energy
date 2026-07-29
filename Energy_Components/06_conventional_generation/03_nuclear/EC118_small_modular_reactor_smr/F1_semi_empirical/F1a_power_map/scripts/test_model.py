"""EC118 — SMR — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0})
    for k in ["electric_power_mw", "thermal_power_mw", "efficiency", "coolant_outlet_temp_c"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC118"
    assert info["fidelity"] == "F1a"


def test_full_power_thermal(model):
    r = model.predict({"part_load_ratio": 1.0})
    assert abs(float(r["thermal_power_mw"]) - 540.0) < 1.0


def test_full_power_electric(model):
    r = model.predict({"part_load_ratio": 1.0})
    Pe = float(r["electric_power_mw"])
    # 540 * 0.34 * 0.99 = 181.7 MW_e
    assert 175 < Pe < 185, f"Pe={Pe:.1f}"


def test_efficiency_realistic(model):
    """SMR efficiency ~ 0.32-0.35."""
    r = model.predict({"part_load_ratio": 1.0})
    eta = float(r["efficiency"])
    assert 0.32 < eta < 0.36, f"eta={eta:.4f}"


def test_p_electric_equals_p_thermal_times_eta(model):
    PLR = np.linspace(0.2, 1.0, 30)
    r = model.predict({"part_load_ratio": PLR})
    assert np.allclose(r["electric_power_mw"], r["thermal_power_mw"] * r["efficiency"], rtol=1e-10)


def test_deep_load_following(model):
    """SMR must operate down to PLR=0.2 (better than traditional LWRs)."""
    r = model.predict({"part_load_ratio": 0.2})
    assert float(r["electric_power_mw"]) > 0
    assert float(r["thermal_power_mw"]) > 0


def test_plr_min_enforced(model):
    r_low = model.predict({"part_load_ratio": 0.05})
    r_min = model.predict({"part_load_ratio": 0.2})
    assert abs(float(r_low["thermal_power_mw"]) - float(r_min["thermal_power_mw"])) < 1.0


def test_coolant_temp_increases_with_power(model):
    PLR = np.linspace(0.3, 1.0, 10)
    r = model.predict({"part_load_ratio": PLR})
    assert np.all(np.diff(r["coolant_outlet_temp_c"]) > 0)


def test_coolant_temp_at_full_power(model):
    r = model.predict({"part_load_ratio": 1.0})
    T = float(r["coolant_outlet_temp_c"])
    assert 310 < T < 335, f"T_out={T:.1f}"


def test_power_scales_with_plr_above_derate(model):
    PLR = np.linspace(0.3, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR})
    ratio = r["thermal_power_mw"] / PLR
    assert np.allclose(ratio, ratio[0], rtol=1e-9)


def test_pe_below_pth(model):
    PLR = np.linspace(0.2, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR})
    assert np.all(r["electric_power_mw"] < r["thermal_power_mw"])


def test_custom_coolant_flow(model):
    r_nom = model.predict({"part_load_ratio": 1.0, "coolant_flow_kgs": 1900.0})
    r_red = model.predict({"part_load_ratio": 1.0, "coolant_flow_kgs": 1000.0})
    assert float(r_red["coolant_outlet_temp_c"]) > float(r_nom["coolant_outlet_temp_c"])


def test_benchmark(model):
    PLR = np.random.uniform(0.2, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": PLR})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
