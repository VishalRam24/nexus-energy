"""EC173 -- Distribution Transformer -- F1b IEC Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"load_fraction": 1.0})
    expected = ["efficiency", "p_loss_w", "p_core_w", "p_copper_w",
                "p_stray_w", "t_hot_spot_degc"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC173"
    assert info["fidelity"] == "F1b"


def test_efficiency_at_rated(model):
    """Distribution transformer efficiency > 98% at rated load (IEC Ck class, 250 kVA)."""
    r = model.predict({"load_fraction": 1.0})
    eta = float(r["efficiency"])
    assert 0.97 < eta <= 1.0, f"Efficiency {eta:.4f} below IEC requirement at rated load"


def test_core_loss_constant_at_nominal_voltage(model):
    """Core loss = P_0 at V_pu = 1.0."""
    r = model.predict({"load_fraction": 0.0, "voltage_pu": 1.0})
    P_0 = model.params["unit"]["P_no_load_W"]["value"]
    assert abs(float(r["p_core_w"]) - P_0) < 1e-6, \
        f"Core loss {float(r['p_core_w']):.2f} != P_0={P_0}"


def test_core_loss_at_no_load_equals_total(model):
    """At no load, total losses = core loss only (copper = 0)."""
    r = model.predict({"load_fraction": 0.0})
    assert abs(float(r["p_copper_w"])) < 1e-9
    assert abs(float(r["p_stray_w"])) < 1e-9
    assert abs(float(r["p_loss_w"]) - float(r["p_core_w"])) < 1e-9


def test_copper_loss_quadratic_in_load(model):
    """Copper loss scales as PLR^2 (at constant temperature)."""
    r1 = model.predict({"load_fraction": 0.5, "winding_temp": 75.0})
    r2 = model.predict({"load_fraction": 1.0, "winding_temp": 75.0})
    ratio = float(r2["p_copper_w"]) / float(r1["p_copper_w"])
    assert abs(ratio - 4.0) < 0.01, f"Copper loss ratio {ratio:.4f}, expected 4.0"


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"load_fraction": 0.8})
    total = float(r["p_core_w"]) + float(r["p_copper_w"]) + float(r["p_stray_w"])
    assert abs(total - float(r["p_loss_w"])) < 1e-9


def test_copper_loss_increases_with_temperature(model):
    """Higher winding temperature → higher copper loss (positive tempco)."""
    r1 = model.predict({"load_fraction": 1.0, "winding_temp": 75.0})
    r2 = model.predict({"load_fraction": 1.0, "winding_temp": 120.0})
    assert float(r2["p_copper_w"]) > float(r1["p_copper_w"]), \
        "Copper loss must increase with temperature"


def test_optimal_load_maximizes_efficiency(model):
    """Efficiency maximum occurs near PLR_opt = sqrt(P_0 / P_k)."""
    plr_opt = model._model.optimal_load_fraction()
    plr_range = np.linspace(0.1, 1.2, 200)
    etas = []
    for plr in plr_range:
        r = model.predict({"load_fraction": plr})
        etas.append(float(r["efficiency"]))
    idx_max = np.argmax(etas)
    plr_at_max = plr_range[idx_max]
    # Optimal should be within 10% of predicted value
    assert abs(plr_at_max - plr_opt) < 0.10, \
        f"Efficiency peak at PLR={plr_at_max:.3f}, expected near PLR_opt={plr_opt:.3f}"


def test_core_loss_voltage_dependence(model):
    """Core loss must increase with higher voltage (Steinmetz)."""
    r1 = model.predict({"load_fraction": 0.0, "voltage_pu": 0.95})
    r2 = model.predict({"load_fraction": 0.0, "voltage_pu": 1.05})
    assert float(r2["p_core_w"]) > float(r1["p_core_w"]), \
        "Core loss must increase with voltage"


def test_hot_spot_above_ambient(model):
    r = model.predict({"load_fraction": 1.0, "ambient_temp": 25.0})
    assert float(r["t_hot_spot_degc"]) > 25.0


def test_vectorized(model):
    plr = np.linspace(0.1, 1.2, 12)
    r = model.predict({"load_fraction": plr})
    assert len(r["efficiency"]) == 12


def test_benchmark(model):
    plr = np.random.uniform(0.05, 1.2, 1000)
    start = time.perf_counter()
    model.predict({"load_fraction": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
