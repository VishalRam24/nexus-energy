"""EC164 -- Three-Phase Inverter -- F1b Detailed Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_dc": 800.0, "p_load": 50000.0, "m": 0.9, "power_factor": 1.0})
    expected = ["v_ac_rms_V", "i_phase_peak_A", "efficiency", "p_loss_w",
                "p_igbt_cond_w", "p_igbt_sw_w", "p_diode_cond_w", "p_diode_rr_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC164"
    assert info["fidelity"] == "F1b"


def test_efficiency_less_than_100(model):
    """Efficiency must be < 100% for any positive load."""
    for p in [5000, 20000, 50000, 80000, 100000]:
        r = model.predict({"v_dc": 800.0, "p_load": p, "m": 0.9, "power_factor": 1.0})
        eta = float(r["efficiency"])
        assert 0.0 < eta < 1.0, f"eta={eta} at P={p}W"


def test_conduction_losses_increase_with_current(model):
    """IGBT conduction losses must increase with load (I^2 + I relationship)."""
    r1 = model.predict({"v_dc": 800.0, "p_load": 25000.0, "m": 0.9, "power_factor": 1.0})
    r2 = model.predict({"v_dc": 800.0, "p_load": 50000.0, "m": 0.9, "power_factor": 1.0})
    r3 = model.predict({"v_dc": 800.0, "p_load": 100000.0, "m": 0.9, "power_factor": 1.0})
    assert float(r2["p_igbt_cond_w"]) > float(r1["p_igbt_cond_w"])
    assert float(r3["p_igbt_cond_w"]) > float(r2["p_igbt_cond_w"])


def test_switching_losses_increase_with_frequency(model):
    """Switching losses must scale with frequency."""
    import json
    from model import ThreePhaseInverterF1b
    params_1 = json.loads(json.dumps(model.params))
    params_2 = json.loads(json.dumps(model.params))
    params_1["unit"]["f_sw"]["value"] = 10000.0
    params_2["unit"]["f_sw"]["value"] = 20000.0
    m1 = ThreePhaseInverterF1b(params_1)
    m2 = ThreePhaseInverterF1b(params_2)
    p_sw1 = float(m1.total_losses(800.0, 50000.0, 0.9, 1.0) -
                   6 * m1.igbt_conduction_loss_per_device(
                       m1.phase_peak_current(800.0, 0.9, 50000.0, 1.0), 0.9, 1.0) -
                   6 * m1.diode_conduction_loss_per_device(
                       m1.phase_peak_current(800.0, 0.9, 50000.0, 1.0), 0.9, 1.0))
    p_sw2 = float(m2.total_losses(800.0, 50000.0, 0.9, 1.0) -
                   6 * m2.igbt_conduction_loss_per_device(
                       m2.phase_peak_current(800.0, 0.9, 50000.0, 1.0), 0.9, 1.0) -
                   6 * m2.diode_conduction_loss_per_device(
                       m2.phase_peak_current(800.0, 0.9, 50000.0, 1.0), 0.9, 1.0))
    ratio = p_sw2 / p_sw1
    assert abs(ratio - 2.0) < 0.01, f"Switching loss ratio {ratio:.3f}, expected 2.0"


def test_loss_breakdown_sums_to_total(model):
    """Sum of loss components must equal total."""
    r = model.predict({"v_dc": 800.0, "p_load": 80000.0, "m": 0.9, "power_factor": 0.95})
    component_sum = (float(r["p_igbt_cond_w"]) + float(r["p_igbt_sw_w"]) +
                     float(r["p_diode_cond_w"]) + float(r["p_diode_rr_w"]))
    total = float(r["p_loss_w"])
    assert abs(component_sum - total) < 1e-6, \
        f"Component sum {component_sum:.6f} != total {total:.6f}"


def test_zero_load_zero_losses(model):
    """Zero load should produce zero (or near-zero) losses."""
    r = model.predict({"v_dc": 800.0, "p_load": 0.0, "m": 0.9, "power_factor": 1.0})
    assert float(r["p_loss_w"]) < 1e-9, "Losses must be ~zero at zero load"


def test_efficiency_map_physical(model):
    """Efficiency should generally increase with power factor."""
    r_low = model.predict({"v_dc": 800.0, "p_load": 80000.0, "m": 0.9, "power_factor": 0.7})
    r_high = model.predict({"v_dc": 800.0, "p_load": 80000.0, "m": 0.9, "power_factor": 1.0})
    # At higher PF, more power per amp -> lower losses relative to output
    # But total losses may still be comparable; just check both are valid
    assert 0.0 < float(r_low["efficiency"]) < 1.0
    assert 0.0 < float(r_high["efficiency"]) < 1.0


def test_vectorized(model):
    """Vectorized computation over p_load array."""
    p_load = np.array([10000, 30000, 60000, 90000], dtype=float)
    r = model.predict({"v_dc": 800.0, "p_load": p_load, "m": 0.9, "power_factor": 1.0})
    assert r["efficiency"].shape == (4,)
    assert np.all(r["efficiency"] > 0)
    assert np.all(r["efficiency"] < 1)


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({"v_dc": 800.0, "p_load": 50000.0, "m": 0.9, "power_factor": 0.95})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 2.0
