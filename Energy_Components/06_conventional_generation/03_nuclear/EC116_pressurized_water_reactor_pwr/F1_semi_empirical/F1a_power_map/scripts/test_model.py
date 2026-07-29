"""EC116 — PWR — F1a Steady-State Power Map — Test Suite"""
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
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC116"
    assert info["fidelity"] == "F1a"


def test_full_power_thermal(model):
    """Thermal power at PLR=1 must equal P_thermal_rated = 3000 MW."""
    r = model.predict({"part_load_ratio": 1.0})
    assert abs(float(r["thermal_power_mw"]) - 3000.0) < 1.0


def test_full_power_electric_approx_1000mw(model):
    """Electric power at PLR=1 must be approximately 1000 MW (eta=0.33, gen=0.99)."""
    r = model.predict({"part_load_ratio": 1.0})
    Pe = float(r["electric_power_mw"])
    # 3000 * 0.33 * 0.99 = 980.1 MW_e (generator efficiency slightly reduces output)
    assert 975 < Pe < 1010, f"P_electric at full load = {Pe:.1f} MW (expected ~980-1000)"


def test_efficiency_approximately_33_percent(model):
    """Net efficiency at full power should be ~0.33."""
    r = model.predict({"part_load_ratio": 1.0})
    eta = float(r["efficiency"])
    assert 0.32 < eta < 0.34, f"eta = {eta:.4f}"


def test_p_electric_equals_p_thermal_times_eta(model):
    """P_electric = P_thermal * eta must hold exactly."""
    PLR = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR})
    Pe = r["electric_power_mw"]
    Pt = r["thermal_power_mw"]
    eta = r["efficiency"]
    assert np.allclose(Pe, Pt * eta, rtol=1e-10)


def test_plr_min_enforced(model):
    """PLR below 0.5 should be clamped to 0.5."""
    r_low = model.predict({"part_load_ratio": 0.1})
    r_min = model.predict({"part_load_ratio": 0.5})
    assert abs(float(r_low["thermal_power_mw"]) - float(r_min["thermal_power_mw"])) < 1.0


def test_coolant_temp_increases_with_power(model):
    """Higher power -> higher coolant outlet temperature."""
    PLR = np.linspace(0.5, 1.0, 10)
    r = model.predict({"part_load_ratio": PLR})
    T = r["coolant_outlet_temp_c"]
    assert np.all(np.diff(T) > 0), "Coolant T_out must increase with PLR"


def test_coolant_temp_at_full_power(model):
    """Coolant outlet temp at full power should be ~326 degC."""
    r = model.predict({"part_load_ratio": 1.0})
    T = float(r["coolant_outlet_temp_c"])
    assert 320 < T < 335, f"T_outlet at full power = {T:.1f} degC (expected ~326)"


def test_power_scales_with_plr(model):
    """P_thermal must scale linearly with PLR."""
    PLR = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR})
    Pt = r["thermal_power_mw"]
    # Linear: Pt = P_rated * PLR -> Pt/PLR = const
    ratio = Pt / PLR
    assert np.allclose(ratio, ratio[0], rtol=1e-9), "Thermal power must be linear in PLR"


def test_custom_coolant_flow(model):
    """Custom coolant flow reduces outlet temperature (less heat removal per kg)."""
    r_nominal = model.predict({"part_load_ratio": 1.0, "coolant_flow_kgs": 18000.0})
    r_reduced = model.predict({"part_load_ratio": 1.0, "coolant_flow_kgs": 12000.0})
    assert float(r_reduced["coolant_outlet_temp_c"]) > float(r_nominal["coolant_outlet_temp_c"])


def test_benchmark(model):
    PLR = np.random.uniform(0.5, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": PLR})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
