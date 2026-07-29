"""EC158 -- Boost Converter -- F1b Detailed Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    expected = ["duty_cycle", "v_out", "i_in", "efficiency", "p_loss_w",
                "p_mosfet_cond_w", "p_diode_cond_w", "p_switching_w", "p_inductor_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC158"
    assert info["fidelity"] == "F1b"


def test_efficiency_less_than_100(model):
    """Efficiency must be < 100% when load > 0."""
    i_range = np.linspace(0.5, 10.0, 50)
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": i_range})
    assert np.all(r["efficiency"] < 1.0), "Efficiency must be < 100%"
    assert np.all(r["efficiency"] > 0.0), "Efficiency must be > 0%"


def test_conduction_losses_increase_with_current(model):
    """MOSFET and inductor conduction losses scale with I^2 (via I_in^2)."""
    r1 = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 2.0})
    r2 = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 4.0})
    # I_in = I_out * V_out / V_in; doubling I_out doubles I_in
    # MOSFET cond = I_in^2 * D * R_ds_on => ratio ~4
    ratio_mos = float(r2["p_mosfet_cond_w"]) / float(r1["p_mosfet_cond_w"])
    assert 3.5 < ratio_mos < 4.5, f"MOSFET cond ratio {ratio_mos:.3f}, expected ~4"
    # Inductor: I_in^2 * R_L => ratio ~4
    ratio_ind = float(r2["p_inductor_w"]) / float(r1["p_inductor_w"])
    assert 3.5 < ratio_ind < 4.5, f"Inductor loss ratio {ratio_ind:.3f}, expected ~4"


def test_switching_losses_increase_with_frequency(model):
    """Switching losses must increase linearly with frequency."""
    import json
    params_1 = json.loads(json.dumps(model.params))
    params_2 = json.loads(json.dumps(model.params))
    params_1["unit"]["f_sw"]["value"] = 100000.0
    params_2["unit"]["f_sw"]["value"] = 200000.0
    from model import BoostConverterF1b
    m1 = BoostConverterF1b(params_1)
    m2 = BoostConverterF1b(params_2)
    p_sw1 = float(m1.switching_loss(12.0, 48.0, 5.0))
    p_sw2 = float(m2.switching_loss(12.0, 48.0, 5.0))
    ratio = p_sw2 / p_sw1
    assert abs(ratio - 2.0) < 1e-6, f"Switching loss ratio {ratio:.6f}, expected 2.0"


def test_loss_breakdown_sums_to_total(model):
    """Sum of all loss components must equal total losses."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    component_sum = (float(r["p_mosfet_cond_w"]) + float(r["p_diode_cond_w"]) +
                     float(r["p_switching_w"]) + float(r["p_inductor_w"]))
    total = float(r["p_loss_w"])
    assert abs(component_sum - total) < 1e-9, \
        f"Component sum {component_sum:.6f} != total {total:.6f}"


def test_zero_load_zero_losses(model):
    """Zero load current should produce zero losses."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 0.0})
    assert float(r["p_loss_w"]) == 0.0, "Losses must be zero at zero load"
    assert float(r["p_mosfet_cond_w"]) == 0.0
    assert float(r["p_diode_cond_w"]) == 0.0
    assert float(r["p_switching_w"]) == 0.0
    assert float(r["p_inductor_w"]) == 0.0


def test_duty_cycle_correct(model):
    """D = 1 - V_in / V_out for ideal boost."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    D = float(r["duty_cycle"])
    expected = 1.0 - 12.0 / 48.0
    assert abs(D - expected) < 1e-9, f"D={D:.6f}, expected {expected:.6f}"


def test_input_current_conservation(model):
    """I_in = I_out * V_out / V_in (power conservation)."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    expected_i_in = 5.0 * 48.0 / 12.0
    assert abs(float(r["i_in"]) - expected_i_in) < 1e-6


def test_vectorized(model):
    """Vectorized computation works correctly."""
    v_in = np.array([10.0, 12.0, 15.0, 20.0])
    r = model.predict({"v_in": v_in, "v_out_target": 48.0, "i_load": 5.0})
    expected_D = 1.0 - v_in / 48.0
    np.testing.assert_allclose(r["duty_cycle"], expected_D, rtol=1e-9)


def test_benchmark(model):
    v_in = np.random.uniform(8, 30, 1000)
    i_load = np.random.uniform(0.5, 10.0, 1000)
    start = time.perf_counter()
    model.predict({"v_in": v_in, "v_out_target": 48.0, "i_load": i_load})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
