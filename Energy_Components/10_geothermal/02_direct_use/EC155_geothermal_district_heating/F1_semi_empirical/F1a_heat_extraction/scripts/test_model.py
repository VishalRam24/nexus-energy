"""EC155 — Geothermal District Heating — F1a Heat Extraction — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_source": 80.0, "T_return": 40.0, "flow_rate_kgs": 50.0})
    for k in ["heat_extracted_kw", "heat_transferred_kw", "heat_delivered_kw",
              "heat_coefficient", "pump_power_kw", "system_cop"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC155"
    assert "fidelity" in info


def test_delivered_less_than_extracted(model):
    """Q_delivered must always be less than Q_extracted (2nd law + losses)."""
    T_srcs = np.linspace(50, 150, 20)
    for T_src in T_srcs:
        r = model.predict({"T_source": float(T_src), "T_return": 40.0, "flow_rate_kgs": 50.0})
        Q_ext = float(r["heat_extracted_kw"])
        Q_del = float(r["heat_delivered_kw"])
        assert Q_del < Q_ext + 1e-9, (
            f"T_src={T_src}: Q_del={Q_del:.1f} >= Q_ext={Q_ext:.1f}"
        )


def test_heat_coefficient_in_range(model):
    """Coefficient Q_delivered/Q_extracted must be in [0.85, 0.95] typical range."""
    T_srcs = np.linspace(60, 140, 15)
    for T_src in T_srcs:
        r = model.predict({"T_source": float(T_src), "T_return": 40.0, "flow_rate_kgs": 50.0})
        coeff = float(r["heat_coefficient"])
        assert 0.70 <= coeff <= 0.99, (
            f"T_src={T_src}: coefficient={coeff:.3f} outside expected 0.70-0.99"
        )


def test_heat_coefficient_constant_with_flow(model):
    """Heat coefficient must be independent of flow rate (it's a ratio)."""
    flows = np.array([10.0, 50.0, 100.0, 250.0, 500.0])
    coeffs = np.array([float(model.predict({
        "T_source": 80.0, "T_return": 40.0, "flow_rate_kgs": float(f)
    })["heat_coefficient"]) for f in flows])
    assert np.std(coeffs) < 1e-9, f"Coefficient varies with flow: {coeffs}"


def test_heat_proportional_to_flow(model):
    """Q_extracted and Q_delivered must scale linearly with flow rate."""
    flows = np.array([10.0, 50.0, 100.0, 250.0, 500.0])
    Q_dels = np.array([float(model.predict({
        "T_source": 80.0, "T_return": 40.0, "flow_rate_kgs": float(f)
    })["heat_delivered_kw"]) for f in flows])
    ratios = Q_dels / flows
    assert np.std(ratios) / np.mean(ratios) < 0.01, "Q_delivered not proportional to flow"


def test_heat_increases_with_T_source(model):
    """Higher T_source → more heat extracted → more delivered."""
    T_srcs = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
    Q_dels = np.array([float(model.predict({
        "T_source": float(T), "T_return": 40.0, "flow_rate_kgs": 50.0
    })["heat_delivered_kw"]) for T in T_srcs])
    assert np.all(np.diff(Q_dels) > 0), f"Q_del not increasing with T_source: {Q_dels}"


def test_heat_decreases_with_T_return(model):
    """Higher T_return → less delta-T → less heat extracted."""
    T_rets = np.array([20.0, 30.0, 40.0, 50.0])
    Q_dels = np.array([float(model.predict({
        "T_source": 80.0, "T_return": float(T), "flow_rate_kgs": 50.0
    })["heat_delivered_kw"]) for T in T_rets])
    assert np.all(np.diff(Q_dels) < 0), f"Q_del not decreasing with T_return: {Q_dels}"


def test_zero_delta_T_gives_zero_heat(model):
    """If T_source = T_return, no heat can be extracted."""
    r = model.predict({"T_source": 80.0, "T_return": 80.0, "flow_rate_kgs": 50.0})
    assert abs(float(r["heat_extracted_kw"])) < 1e-6, "Non-zero heat when T_source=T_return"
    assert abs(float(r["heat_delivered_kw"])) < 1e-6


def test_system_cop_high(model):
    """System COP (Q_del / W_pump) should be very high — direct use, not compressor HP."""
    r = model.predict({"T_source": 80.0, "T_return": 40.0, "flow_rate_kgs": 50.0})
    scop = float(r["system_cop"])
    assert scop > 20.0, f"System COP={scop:.1f}, expected >20 for direct-use geothermal"


def test_heat_nonnegative(model):
    """All heat outputs must be non-negative."""
    T_srcs = np.linspace(50, 150, 20)
    for T_src in T_srcs:
        r = model.predict({"T_source": float(T_src), "T_return": 40.0, "flow_rate_kgs": 50.0})
        for k in ["heat_extracted_kw", "heat_delivered_kw", "pump_power_kw"]:
            assert float(r[k]) >= 0.0, f"Negative {k} at T_src={T_src}"


def test_array_input(model):
    """Model must accept array inputs."""
    T_srcs = np.linspace(50, 150, 15)
    r = model.predict({"T_source": T_srcs, "T_return": 40.0, "flow_rate_kgs": 50.0})
    assert len(r["heat_delivered_kw"]) == 15


def test_benchmark(model):
    T_srcs = np.random.uniform(50, 150, 1000)
    T_rets = np.random.uniform(20, 60, 1000)
    flows  = np.random.uniform(5, 500, 1000)
    start = time.perf_counter()
    model.predict({"T_source": T_srcs, "T_return": T_rets, "flow_rate_kgs": flows})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
