"""EC159 -- Buck-Boost Converter -- F1b Detailed Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": 5.0})
    expected = ["duty_cycle", "v_out", "efficiency", "p_loss_w", "T_j_degC",
                "p_mosfet_cond_w", "p_diode_cond_w", "p_switching_w", "p_inductor_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC159"
    assert info["fidelity"] == "F1b"


def test_efficiency_bounds(model):
    """Efficiency must be in (0, 1) for positive load."""
    i_range = np.linspace(0.5, 10.0, 50)
    r = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": i_range})
    assert np.all(r["efficiency"] < 1.0), "Efficiency must be < 100%"
    assert np.all(r["efficiency"] > 0.0), "Efficiency must be > 0%"


def test_duty_cycle_formula(model):
    """D = |Vout| / (Vin + |Vout|) for buck-boost."""
    r = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": 5.0})
    D_expected = 12.0 / (24.0 + 12.0)
    assert abs(float(r["duty_cycle"]) - D_expected) < 1e-9


def test_diode_loss_linear_with_current(model):
    """Diode conduction loss = V_f * I_out is linear in current."""
    r1 = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": 2.0})
    r2 = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": 4.0})
    ratio = float(r2["p_diode_cond_w"]) / float(r1["p_diode_cond_w"])
    assert abs(ratio - 2.0) < 1e-6, f"Diode loss ratio {ratio:.6f}, expected 2.0"


def test_loss_breakdown_sums_to_total(model):
    """Sum of all loss components must equal total losses."""
    r = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": 5.0})
    component_sum = (float(r["p_mosfet_cond_w"]) + float(r["p_diode_cond_w"]) +
                     float(r["p_switching_w"]) + float(r["p_inductor_w"]))
    total = float(r["p_loss_w"])
    assert abs(component_sum - total) < 1e-6, \
        f"Component sum {component_sum:.6f} != total {total:.6f}"


def test_zero_load_zero_losses(model):
    """Zero load current should produce zero losses."""
    r = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": 0.0})
    assert float(r["p_loss_w"]) < 1e-9, "Losses must be zero at zero load"


def test_thermal_balance(model):
    """T_j >= T_a when losses are positive."""
    r = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": 8.0})
    T_a = model.params["unit"]["T_a"]["value"]
    assert float(r["T_j_degC"]) >= T_a, "T_j must be >= T_a"


def test_rds_on_tempco_increases_loss(model):
    """Higher junction temperature must increase MOSFET conduction loss (positive tempco)."""
    import json
    from model import BuckBoostConverterF1b
    # Manually set different T_a to force different T_j
    p_cold = json.loads(json.dumps(model.params))
    p_hot = json.loads(json.dumps(model.params))
    p_cold["unit"]["T_a"]["value"] = 25.0
    p_hot["unit"]["T_a"]["value"] = 75.0
    m_cold = BuckBoostConverterF1b(p_cold)
    m_hot = BuckBoostConverterF1b(p_hot)
    loss_cold = float(m_cold.mosfet_conduction_loss(24.0, 12.0, 5.0))
    loss_hot = float(m_hot.mosfet_conduction_loss(24.0, 12.0, 5.0))
    assert loss_hot > loss_cold, f"Hot loss {loss_hot:.4f}W must exceed cold {loss_cold:.4f}W"


def test_switching_loss_scales_with_frequency(model):
    """Switching losses scale linearly with f_sw."""
    import json
    from model import BuckBoostConverterF1b
    p1 = json.loads(json.dumps(model.params))
    p2 = json.loads(json.dumps(model.params))
    p1["unit"]["f_sw"]["value"] = 50000.0
    p2["unit"]["f_sw"]["value"] = 100000.0
    m1 = BuckBoostConverterF1b(p1)
    m2 = BuckBoostConverterF1b(p2)
    psw1 = float(m1.switching_loss(24.0, 12.0, 5.0))
    psw2 = float(m2.switching_loss(24.0, 12.0, 5.0))
    assert abs(psw2 / psw1 - 2.0) < 1e-6, f"Expected ratio 2.0, got {psw2/psw1:.6f}"


def test_vectorized(model):
    """Vectorized computation produces correct shapes."""
    v_in = np.array([12.0, 18.0, 24.0, 36.0])
    r = model.predict({"v_in": v_in, "v_out_target": 12.0, "i_load": 5.0})
    assert r["efficiency"].shape == (4,)
    assert np.all(r["efficiency"] > 0)


def test_benchmark(model):
    v_in = np.random.uniform(10, 40, 1000)
    i_load = np.random.uniform(0.1, 10.0, 1000)
    start = time.perf_counter()
    model.predict({"v_in": v_in, "v_out_target": 12.0, "i_load": i_load})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 2.0
