"""EC165 -- Multilevel Inverter -- F1b Detailed Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_dc": 700.0, "p_load": 40000.0, "m": 0.9})
    expected = ["efficiency", "p_loss_w", "t_j_degc",
                "p_outer_igbt_cond_w", "p_inner_igbt_cond_w",
                "p_outer_igbt_sw_w", "p_inner_igbt_sw_w",
                "p_clamp_diode_cond_w", "p_clamp_diode_rr_w"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC165"
    assert info["fidelity"] == "F1b"


def test_efficiency_physical_bounds(model):
    """Efficiency must be in (0, 1) for non-zero load."""
    r = model.predict({"v_dc": 700.0, "p_load": 40000.0, "m": 0.9})
    eta = float(r["efficiency"])
    assert 0.8 < eta < 1.0, f"Efficiency {eta:.4f} outside physical range"


def test_all_losses_positive(model):
    """All loss components must be non-negative."""
    r = model.predict({"v_dc": 700.0, "p_load": 40000.0, "m": 0.9})
    for k in ["p_outer_igbt_cond_w", "p_inner_igbt_cond_w",
              "p_outer_igbt_sw_w", "p_inner_igbt_sw_w",
              "p_clamp_diode_cond_w", "p_clamp_diode_rr_w"]:
        assert float(r[k]) >= 0.0, f"{k} is negative"


def test_loss_breakdown_sums_to_total(model):
    """Sum of all loss components must equal total losses."""
    r = model.predict({"v_dc": 700.0, "p_load": 40000.0, "m": 0.9})
    component_sum = (float(r["p_outer_igbt_cond_w"]) + float(r["p_inner_igbt_cond_w"]) +
                     float(r["p_outer_igbt_sw_w"]) + float(r["p_inner_igbt_sw_w"]) +
                     float(r["p_clamp_diode_cond_w"]) + float(r["p_clamp_diode_rr_w"]))
    total = float(r["p_loss_w"])
    assert abs(component_sum - total) < 1e-6, \
        f"Component sum {component_sum:.4f} != total {total:.4f}"


def test_losses_increase_with_power(model):
    """Total losses increase monotonically with output power."""
    powers = np.linspace(5000, 50000, 20)
    losses = []
    for p in powers:
        r = model.predict({"v_dc": 700.0, "p_load": float(p), "m": 0.9})
        losses.append(float(r["p_loss_w"]))
    diffs = np.diff(losses)
    assert np.all(diffs > 0), "Losses must increase monotonically with power"


def test_switching_loss_scales_with_frequency(model):
    """Switching losses scale linearly with switching frequency."""
    import json
    p1 = json.loads(json.dumps(model.params))
    p2 = json.loads(json.dumps(model.params))
    p1["unit"]["f_sw"]["value"] = 2000.0
    p2["unit"]["f_sw"]["value"] = 4000.0
    from model import MultilevelInverterF1b
    m1 = MultilevelInverterF1b(p1)
    m2 = MultilevelInverterF1b(p2)
    sw1 = float(m1.outer_igbt_switching_per_phase(100.0, 700.0))
    sw2 = float(m2.outer_igbt_switching_per_phase(100.0, 700.0))
    ratio = sw2 / sw1
    assert abs(ratio - 2.0) < 1e-6, f"Switching loss ratio {ratio:.6f}, expected 2.0"


def test_inner_switching_half_voltage(model):
    """Inner IGBTs switch at V_dc/2; switching loss should be ~half of outer at same I."""
    # At same current stress, inner loss = outer loss * (V_dc/2) / V_dc = 0.5
    # (energy also smaller for lower-rated devices, so check inner < outer)
    r = model.predict({"v_dc": 700.0, "p_load": 40000.0, "m": 0.9})
    assert float(r["p_inner_igbt_sw_w"]) < float(r["p_outer_igbt_sw_w"]), \
        "Inner IGBT switching loss must be less than outer (lower voltage stress)"


def test_junction_temperature_above_ambient(model):
    """Junction temperature must be above ambient at rated load."""
    T_a = model.params["unit"]["T_a"]["value"]
    r = model.predict({"v_dc": 700.0, "p_load": 40000.0, "m": 0.9})
    assert float(r["t_j_degc"]) > T_a, "T_j must exceed ambient under load"


def test_zero_load_zero_losses(model):
    """Zero output power yields zero losses."""
    r = model.predict({"v_dc": 700.0, "p_load": 0.0, "m": 0.9})
    assert abs(float(r["p_loss_w"])) < 1e-9, "Zero load must give zero losses"


def test_vectorized(model):
    """Vectorized input works correctly."""
    p_loads = np.array([10000.0, 20000.0, 30000.0, 40000.0, 50000.0])
    r = model.predict({"v_dc": 700.0, "p_load": p_loads, "m": 0.9})
    assert len(r["efficiency"]) == 5


def test_benchmark(model):
    p = np.random.uniform(5000, 50000, 1000)
    start = time.perf_counter()
    model.predict({"v_dc": 700.0, "p_load": p, "m": 0.9})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
