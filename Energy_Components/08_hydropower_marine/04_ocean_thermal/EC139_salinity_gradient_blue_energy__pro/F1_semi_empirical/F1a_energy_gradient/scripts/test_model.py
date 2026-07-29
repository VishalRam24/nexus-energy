"""EC139 — Salinity Gradient (PRO) — F1a — Test Suite"""

import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

_R = 8.314


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({})
    for k in ["osmotic_pressure_bar", "gibbs_energy_kwh_per_m3",
              "net_energy_kwh_per_m3", "power_kw", "extraction_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC139"
    assert info["fidelity"] == "F1a"


def test_net_energy_below_gibbs(model):
    """Net extractable energy must be less than theoretical Gibbs energy."""
    r = model.predict({})
    assert float(r["net_energy_kwh_per_m3"]) < float(r["gibbs_energy_kwh_per_m3"]), \
        "Net energy exceeds theoretical Gibbs free energy"


def test_net_energy_in_expected_range(model):
    """Net energy should be in published range 0.2-0.4 kWh/m³ freshwater for seawater/river mix."""
    r = model.predict({"C_sw": 35.0, "C_fw": 0.5})
    w_net = float(r["net_energy_kwh_per_m3"])
    # RATIONALE: Yip & Elimelech (2012) and Straub et al. (2016) report net PRO output
    # of 0.2-0.4 kWh/m³ freshwater. This model uses eta_mem=0.45, eta_turb=0.85, eta_px=0.95
    # and Π_avg method; expected ~0.24 kWh/m³. Allow 0.15-0.50 to tolerate parameter variation
    # while still failing models that are physically unreasonable (below 0.15 or above 0.50).
    assert 0.15 <= w_net <= 0.50, f"Net energy {w_net:.4f} kWh/m³_fw outside published range 0.15-0.50"


def test_zero_energy_at_equal_concentrations(model):
    """No salinity gradient → no energy."""
    r = model.predict({"C_sw": 35.0, "C_fw": 35.0})
    assert float(r["osmotic_pressure_bar"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["net_energy_kwh_per_m3"]) == pytest.approx(0.0, abs=1e-10)


def test_osmotic_pressure_formula(model):
    """Cross-check osmotic pressure vs Van't Hoff formula."""
    C_sw, C_fw = 35.0, 0.5
    M_NaCl, nu, T_K = 58.44, 2, 298.15
    dC_mol = (C_sw - C_fw) / M_NaCl * 1000.0  # mol/m³
    Pi_expected = nu * _R * T_K * dC_mol / 1e5   # bar
    r = model.predict({"C_sw": C_sw, "C_fw": C_fw})
    assert abs(float(r["osmotic_pressure_bar"]) - Pi_expected) < 0.01


def test_osmotic_pressure_increases_with_salinity(model):
    """Higher seawater salinity → higher osmotic pressure."""
    C_sw_vals = np.array([25.0, 30.0, 35.0, 40.0])
    r = model.predict({"C_sw": C_sw_vals, "C_fw": 0.5})
    assert np.all(np.diff(r["osmotic_pressure_bar"]) > 0)


def test_net_energy_increases_with_salinity_gradient(model):
    """Higher ΔC → higher net energy."""
    C_sw_vals = np.array([28.0, 32.0, 35.0, 38.0])
    r = model.predict({"C_sw": C_sw_vals, "C_fw": 0.5})
    assert np.all(np.diff(r["net_energy_kwh_per_m3"]) > 0)


def test_power_linear_with_flow_rate(model):
    """Power scales linearly with feed flow rate."""
    r1 = model.predict({"C_sw": 35.0, "C_fw": 0.5, "Q_feed_m3s": 1.0})
    r2 = model.predict({"C_sw": 35.0, "C_fw": 0.5, "Q_feed_m3s": 2.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 2.0) < 0.01, f"2x flow should give 2x power; got ratio={ratio:.4f}"


def test_extraction_efficiency_below_1(model):
    """Extraction efficiency (net/Gibbs) must be < 1."""
    r = model.predict({})
    assert float(r["extraction_efficiency"]) < 1.0


def test_positive_values(model):
    """All outputs positive for valid salinity gradient."""
    r = model.predict({"C_sw": 35.0, "C_fw": 0.5})
    for k in ["osmotic_pressure_bar", "gibbs_energy_kwh_per_m3",
              "net_energy_kwh_per_m3", "power_kw", "extraction_efficiency"]:
        assert float(np.atleast_1d(r[k])[0]) > 0.0, f"{k} should be positive"


def test_benchmark(model):
    C_sw = np.random.uniform(25.0, 40.0, 1000)
    C_fw = np.random.uniform(0.1, 2.0, 1000)
    start = time.perf_counter()
    model.predict({"C_sw": C_sw, "C_fw": C_fw})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
