"""EC122 — Pumped Hydro Storage — F1b Head Variation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 50.0, "mode": "discharge"})
    for k in ["power_kw", "effective_head_m", "friction_loss_m",
              "efficiency", "round_trip_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC122"
    assert info["fidelity"] == "F1b"


# --- Head Variation ---

def test_head_at_soc_0_equals_hmin(model):
    r = model.predict({"SOC": 0.0, "flow_rate_m3s": 50.0})
    assert abs(float(r["effective_head_m"]) - 480.0) < 0.01


def test_head_at_soc_1_equals_hmax(model):
    r = model.predict({"SOC": 1.0, "flow_rate_m3s": 50.0})
    assert abs(float(r["effective_head_m"]) - 500.0) < 0.01


def test_head_monotonically_increases_with_soc(model):
    soc = np.linspace(0, 1, 20)
    r = model.predict({"SOC": soc, "flow_rate_m3s": 50.0})
    assert np.all(np.diff(r["effective_head_m"]) > 0)


# --- Friction Losses ---

def test_friction_loss_zero_at_zero_flow(model):
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 0.0})
    assert float(r["friction_loss_m"]) == 0.0


def test_friction_loss_positive(model):
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 50.0})
    assert float(r["friction_loss_m"]) > 0.0


def test_friction_loss_increases_with_flow(model):
    """Friction ~ Q^2, so must be monotonically increasing with |Q|."""
    Q_arr = np.array([10, 25, 50, 75, 100])
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": Q_arr})
    assert np.all(np.diff(r["friction_loss_m"]) > 0)


def test_friction_loss_quadratic(model):
    """h_f proportional to Q^2: doubling flow -> 4x friction."""
    r1 = model.predict({"SOC": 0.5, "flow_rate_m3s": 10.0})
    r2 = model.predict({"SOC": 0.5, "flow_rate_m3s": 20.0})
    ratio = float(r2["friction_loss_m"]) / float(r1["friction_loss_m"])
    assert abs(ratio - 4.0) < 0.01, f"Friction ratio should be 4.0, got {ratio:.3f}"


# --- Power & Efficiency ---

def test_generation_power_positive(model):
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 50.0, "mode": "discharge"})
    assert float(r["power_kw"]) > 0


def test_pumping_power_positive(model):
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 50.0, "mode": "charge"})
    assert float(r["power_kw"]) > 0


def test_pumping_power_greater_than_generation(model):
    """For same flow and head, pumping requires more power than generation produces."""
    r_gen = model.predict({"SOC": 0.5, "flow_rate_m3s": 50.0, "mode": "discharge"})
    r_pump = model.predict({"SOC": 0.5, "flow_rate_m3s": 50.0, "mode": "charge"})
    assert float(r_pump["power_kw"]) > float(r_gen["power_kw"])


def test_efficiency_bounded(model):
    soc = np.random.uniform(0, 1, 100)
    Q = np.random.uniform(5, 80, 100)
    r = model.predict({"SOC": soc, "flow_rate_m3s": Q, "mode": "discharge"})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_rte_less_than_generation_efficiency(model):
    """Round-trip efficiency < one-way generation efficiency."""
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 50.0, "mode": "discharge"})
    rte = float(r["round_trip_efficiency"])
    eta_gen = float(r["efficiency"])
    assert rte < eta_gen


def test_rte_reasonable_range(model):
    """RTE for PHS typically 70-85%."""
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 25.0, "mode": "discharge"})
    rte = float(r["round_trip_efficiency"])
    assert 0.60 <= rte <= 0.90, f"RTE={rte:.4f} outside expected [0.60, 0.90]"


def test_efficiency_decreases_with_flow(model):
    """Higher flow -> more friction -> lower efficiency."""
    Q_arr = np.array([10, 25, 50, 75, 100])
    etas = []
    for Q in Q_arr:
        r = model.predict({"SOC": 0.5, "flow_rate_m3s": float(Q), "mode": "discharge"})
        etas.append(float(r["efficiency"]))
    for i in range(len(etas) - 1):
        assert etas[i] >= etas[i + 1], (
            f"eta should decrease with flow: eta(Q={Q_arr[i]})={etas[i]:.4f} "
            f"< eta(Q={Q_arr[i+1]})={etas[i+1]:.4f}"
        )


def test_generation_power_increases_with_soc(model):
    """Higher SOC -> higher head -> more power."""
    soc = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    r = model.predict({"SOC": soc, "flow_rate_m3s": 50.0, "mode": "discharge"})
    assert np.all(np.diff(r["power_kw"]) > 0)


# --- Edge Cases ---

def test_zero_flow(model):
    r = model.predict({"SOC": 0.5, "flow_rate_m3s": 0.0, "mode": "discharge"})
    assert float(r["power_kw"]) == 0.0


def test_vectorized(model):
    soc = np.linspace(0, 1, 50)
    Q = np.linspace(5, 80, 50)
    r = model.predict({"SOC": soc, "flow_rate_m3s": Q, "mode": "discharge"})
    assert len(r["power_kw"]) == 50


def test_benchmark(model):
    soc = np.random.uniform(0, 1, 1000)
    Q = np.random.uniform(5, 80, 1000)
    start = time.perf_counter()
    model.predict({"SOC": soc, "flow_rate_m3s": Q, "mode": "discharge"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
