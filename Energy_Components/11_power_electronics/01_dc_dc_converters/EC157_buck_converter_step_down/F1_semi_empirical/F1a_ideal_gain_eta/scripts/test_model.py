"""EC157 — Buck Converter — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 10.0})
    for k in ["duty_cycle", "v_out", "efficiency", "p_loss_w", "p_conduction_w", "p_switching_w"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC157"
    assert info["fidelity"] == "F1a"


def test_duty_cycle_correct(model):
    """D = V_out / V_in for ideal buck."""
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    D = float(r["duty_cycle"])
    assert abs(D - 12.0 / 48.0) < 1e-9, f"D={D:.6f}, expected {12/48:.6f}"


def test_v_out_equals_d_times_vin(model):
    """V_out = D * V_in (ideal)."""
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    v_out = float(r["v_out"])
    assert abs(v_out - 12.0) < 1e-9, f"V_out={v_out:.6f}, expected 12.0"


def test_efficiency_less_than_one(model):
    """Efficiency < 1 always (losses exist)."""
    i_range = np.linspace(0.5, 15.0, 50)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})
    assert np.all(r["efficiency"] < 1.0)


def test_efficiency_positive(model):
    """Efficiency > 0."""
    i_range = np.linspace(0.5, 15.0, 50)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})
    assert np.all(r["efficiency"] > 0.0)


def test_losses_positive(model):
    """Total losses > 0."""
    i_range = np.linspace(0.5, 15.0, 50)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})
    assert np.all(r["p_loss_w"] > 0.0)


def test_losses_sum_to_total(model):
    """p_conduction + p_switching = p_loss."""
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 10.0})
    total = float(r["p_conduction_w"]) + float(r["p_switching_w"])
    assert abs(total - float(r["p_loss_w"])) < 1e-9


def test_efficiency_monotone_with_load(model):
    """
    With this converter's parameters (Rds_on=10mΩ, R_L=50mΩ, t_sw=30ns),
    conduction losses (I²R) dominate and efficiency decreases monotonically
    with increasing load current. This is the correct physics for low-loss
    synchronous buck converters.
    """
    i_range = np.linspace(0.5, 15.0, 20)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})
    # Efficiency must decrease as load increases (conduction loss dominated)
    assert np.all(np.diff(r["efficiency"]) < 0), \
        "Efficiency should decrease monotonically with load (I²R dominated converter)"


def test_conduction_losses_scale_with_i_squared(model):
    """Conduction losses scale roughly as I^2."""
    r1 = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    r2 = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 10.0})
    ratio = float(r2["p_conduction_w"]) / float(r1["p_conduction_w"])
    # Should be ~4 (2^2) for pure I^2 losses
    assert 3.5 < ratio < 4.5, f"Conduction loss ratio {ratio:.3f}, expected ~4"


def test_switching_losses_independent_of_current_at_fixed_Vin(model):
    """Switching losses scale linearly with I_out (P_sw = 0.5*V_in*I*...*f)."""
    r1 = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    r2 = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 10.0})
    ratio = float(r2["p_switching_w"]) / float(r1["p_switching_w"])
    assert abs(ratio - 2.0) < 1e-6, f"Switching loss ratio {ratio:.6f}, expected 2.0 (linear in I)"


def test_duty_cycle_array(model):
    """Vectorized: array of v_in inputs."""
    v_in = np.array([24.0, 36.0, 48.0, 60.0])
    r = model.predict({"v_in": v_in, "v_out_target": 12.0, "i_load": 5.0})
    expected_D = 12.0 / v_in
    np.testing.assert_allclose(r["duty_cycle"], expected_D, rtol=1e-9)


def test_benchmark(model):
    v_in = np.random.uniform(24, 72, 1000)
    i_load = np.random.uniform(0.5, 15.0, 1000)
    start = time.perf_counter()
    model.predict({"v_in": v_in, "v_out_target": 12.0, "i_load": i_load})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
