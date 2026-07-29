"""EC181 — Transmission Line — F1a Pi-Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.5, "Q_load_pu": 0.2})
    for k in ["V_r_pu", "delta_r_rad", "I_series_pu", "P_loss_pu",
              "Q_loss_pu", "P_s_pu", "Q_s_pu", "efficiency", "voltage_drop_pu"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC181"
    assert info["fidelity"] == "F1a"


def test_voltage_drop_positive_for_inductive_load(model):
    """V_r < V_s for positive P and lagging Q (inductive load)."""
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.8, "Q_load_pu": 0.4})
    assert float(r["V_r_pu"]) < 1.0, f"V_r={float(r['V_r_pu']):.4f} should be < V_s=1.0"


def test_P_loss_positive(model):
    """Active power loss must be positive for any real load."""
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.5, "Q_load_pu": 0.2})
    assert float(r["P_loss_pu"]) > 0.0, "P_loss must be > 0"


def test_P_loss_scales_with_current_squared(model):
    """Doubling current roughly quadruples I^2*R loss."""
    r1 = model.predict({"V_s_pu": 1.05, "delta_s_rad": 0.0,
                        "P_load_pu": 0.3, "Q_load_pu": 0.14})
    r2 = model.predict({"V_s_pu": 1.05, "delta_s_rad": 0.0,
                        "P_load_pu": 0.6, "Q_load_pu": 0.28})
    ratio = float(r2["P_loss_pu"]) / float(r1["P_loss_pu"])
    # I doubles → P_loss ≈ 4x (approximate — V_r changes slightly)
    assert 3.0 < ratio < 5.0, f"Loss ratio {ratio:.2f}, expected ~4"


def test_no_load_has_minimal_loss(model):
    """Zero load → near-zero series current → near-zero I^2*R loss."""
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.0, "Q_load_pu": 0.0})
    # Only shunt charging current flows → very small P_loss
    assert float(r["P_loss_pu"]) < 1e-4, f"No-load P_loss={float(r['P_loss_pu']):.6f}"


def test_efficiency_below_one(model):
    """Efficiency must be < 1 for loaded line."""
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.6, "Q_load_pu": 0.3})
    assert float(r["efficiency"]) < 1.0


def test_efficiency_positive(model):
    """Efficiency must be > 0."""
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.6, "Q_load_pu": 0.3})
    assert float(r["efficiency"]) > 0.0


def test_longer_line_more_loss(model):
    """Longer line → more resistance → higher P_loss."""
    r100 = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                          "P_load_pu": 0.6, "Q_load_pu": 0.3, "length_km": 100.0})
    r400 = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                          "P_load_pu": 0.6, "Q_load_pu": 0.3, "length_km": 400.0})
    assert float(r400["P_loss_pu"]) > float(r100["P_loss_pu"]), \
        "Longer line must have higher losses"


def test_voltage_drop_increases_with_load(model):
    """Higher load → larger voltage drop."""
    r_low = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                           "P_load_pu": 0.2, "Q_load_pu": 0.1})
    r_high = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                            "P_load_pu": 1.0, "Q_load_pu": 0.5})
    assert float(r_high["voltage_drop_pu"]) > float(r_low["voltage_drop_pu"]), \
        "Higher load must produce larger voltage drop"


def test_power_balance(model):
    """P_s = P_load + P_loss (within numerical tolerance)."""
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.6, "Q_load_pu": 0.2})
    P_balance = float(r["P_s_pu"]) - float(r["P_loss_pu"]) - 0.6
    # Shunt losses (I^2*R of charging current) are small but non-zero; allow 1% tolerance
    assert abs(P_balance) < 0.01, f"Power balance error: {P_balance:.6f} pu"


def test_vectorized_input(model):
    """Vectorized inputs return arrays of correct shape."""
    P = np.linspace(0.1, 1.0, 20)
    Q = P * 0.3
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": P, "Q_load_pu": Q})
    assert r["V_r_pu"].shape == (20,)
    assert r["P_loss_pu"].shape == (20,)


def test_benchmark(model):
    P = np.random.uniform(0.1, 1.2, 1000)
    Q = P * 0.3
    start = time.perf_counter()
    model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0, "P_load_pu": P, "Q_load_pu": Q})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
