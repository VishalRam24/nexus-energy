"""EC168 -- MPPT Controller -- F1b Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"irradiance": 1000.0, "p_mpp_available": 8000.0, "dG_dt": 0.0})
    expected = ["eta_static", "eta_dynamic", "eta_converter", "eta_total",
                "p_out_w", "p_loss_w",
                "p_oscillation_loss_w", "p_dynamic_loss_w", "p_converter_loss_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC168"
    assert info["fidelity"] == "F1b"


def test_efficiency_less_than_100(model):
    """Total efficiency must be < 100% at any irradiance > 0."""
    G_range = np.linspace(100, 1200, 50)
    P_range = 10000.0 * G_range / 1000.0
    r = model.predict({"irradiance": G_range, "p_mpp_available": P_range, "dG_dt": 0.0})
    assert np.all(r["eta_total"] < 1.0), "Total efficiency must be < 100%"
    assert np.all(r["eta_total"] > 0.0), "Total efficiency must be > 0%"


def test_conduction_losses_increase_with_current(model):
    """Converter losses (proportional to power) increase with irradiance/power."""
    r1 = model.predict({"irradiance": 500.0, "p_mpp_available": 5000.0, "dG_dt": 0.0})
    r2 = model.predict({"irradiance": 1000.0, "p_mpp_available": 10000.0, "dG_dt": 0.0})
    assert float(r2["p_converter_loss_w"]) > float(r1["p_converter_loss_w"])


def test_switching_losses_increase_with_frequency(model):
    """
    The MPPT perturbation frequency (1/T_mppt) affects dynamic losses.
    Faster perturbation with large steps increases oscillation losses.
    Here we verify that reducing T_mppt (higher frequency) does not violate physics.
    """
    import json
    from model import MPPTF1b
    # With the same V_step, the oscillation loss is independent of T_mppt
    # But dynamic tracking improves with smaller T_mppt (responds faster)
    params_slow = json.loads(json.dumps(model.params))
    params_fast = json.loads(json.dumps(model.params))
    params_slow["unit"]["T_mppt"]["value"] = 0.100  # 100 ms
    params_fast["unit"]["T_mppt"]["value"] = 0.025   # 25 ms
    m_slow = MPPTF1b(params_slow)
    m_fast = MPPTF1b(params_fast)
    # During a ramp, faster tracking = less dynamic loss
    eta_slow = float(m_slow.dynamic_tracking_efficiency(800.0, 200.0))
    eta_fast = float(m_fast.dynamic_tracking_efficiency(800.0, 200.0))
    assert eta_fast > eta_slow, "Faster tracking should have better dynamic efficiency"


def test_loss_breakdown_sums_to_total(model):
    """Sum of loss components must approximately equal total losses."""
    r = model.predict({"irradiance": 800.0, "p_mpp_available": 8000.0, "dG_dt": 50.0})
    component_sum = (float(r["p_oscillation_loss_w"]) + float(r["p_dynamic_loss_w"]) +
                     float(r["p_converter_loss_w"]))
    total = float(r["p_loss_w"])
    # Allow small relative tolerance due to multiplicative chain
    assert abs(component_sum - total) < 0.1, \
        f"Component sum {component_sum:.3f} != total {total:.3f}"


def test_zero_load_zero_losses(model):
    """Zero irradiance / zero power should produce zero losses."""
    r = model.predict({"irradiance": 0.0, "p_mpp_available": 0.0, "dG_dt": 0.0})
    assert float(r["p_loss_w"]) == 0.0, "Losses must be zero when no power available"
    assert float(r["p_out_w"]) == 0.0, "Output must be zero when no power available"
    assert float(r["eta_total"]) == 0.0


def test_dynamic_loss_during_transient(model):
    """Dynamic tracking efficiency should drop during fast irradiance changes."""
    r_steady = model.predict({"irradiance": 800.0, "p_mpp_available": 8000.0, "dG_dt": 0.0})
    r_ramp = model.predict({"irradiance": 800.0, "p_mpp_available": 8000.0, "dG_dt": 300.0})
    assert float(r_ramp["eta_dynamic"]) < float(r_steady["eta_dynamic"])
    assert float(r_ramp["eta_total"]) < float(r_steady["eta_total"])


def test_static_efficiency_high_at_stc(model):
    """At STC (G=1000), static tracking efficiency should be close to nominal."""
    r = model.predict({"irradiance": 1000.0, "p_mpp_available": 10000.0, "dG_dt": 0.0})
    eta_s = float(r["eta_static"])
    assert eta_s > 0.98, f"Static efficiency {eta_s:.4f} too low at STC"
    assert eta_s <= 0.995, f"Static efficiency {eta_s:.4f} exceeds nominal"


def test_vectorized(model):
    """Vectorized computation over arrays."""
    G = np.array([200, 400, 600, 800, 1000], dtype=float)
    P = 10.0 * G  # 10 W per W/m2
    r = model.predict({"irradiance": G, "p_mpp_available": P, "dG_dt": 0.0})
    assert r["eta_total"].shape == (5,)
    assert np.all(r["eta_total"] > 0)
    assert np.all(r["eta_total"] < 1)


def test_benchmark(model):
    G = np.random.uniform(100, 1200, 1000)
    P = 10.0 * G
    dG = np.random.uniform(-100, 100, 1000)
    start = time.perf_counter()
    model.predict({"irradiance": G, "p_mpp_available": P, "dG_dt": dG})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
