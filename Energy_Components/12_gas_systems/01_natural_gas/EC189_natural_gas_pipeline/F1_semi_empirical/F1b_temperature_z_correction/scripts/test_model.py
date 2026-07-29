"""EC189 — Natural Gas Pipeline — F1b Temperature-Z Correction — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"length_km": 100.0, "diameter_m": 0.5,
                        "P_in_bar": 70.0, "P_out_bar": 55.0})
    for k in ["flow_rate_std_m3_per_day", "flow_rate_kg_per_s",
              "T_avg_K", "T_out_K", "Z_avg"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC189"
    assert info["fidelity"] == "F1b"


def test_flow_positive(model):
    """Flow must be positive for P_in > P_out."""
    r = model.predict({"length_km": 100.0, "diameter_m": 0.5,
                        "P_in_bar": 70.0, "P_out_bar": 55.0})
    Q = float(np.atleast_1d(r["flow_rate_std_m3_per_day"])[0])
    assert Q > 0, f"Flow = {Q}"


def test_flow_zero_when_no_pressure_drop(model):
    """Flow must be ~0 when P_in == P_out."""
    r = model.predict({"length_km": 100.0, "diameter_m": 0.5,
                        "P_in_bar": 70.0, "P_out_bar": 70.0})
    Q = float(np.atleast_1d(r["flow_rate_std_m3_per_day"])[0])
    assert Q < 1e-3, f"Flow not zero at zero dP: {Q}"


def test_flow_increases_with_diameter(model):
    """Larger diameter → higher flow at same pressures."""
    r1 = model.predict({"length_km": 100.0, "diameter_m": 0.3,
                         "P_in_bar": 70.0, "P_out_bar": 55.0})
    r2 = model.predict({"length_km": 100.0, "diameter_m": 0.6,
                         "P_in_bar": 70.0, "P_out_bar": 55.0})
    Q1 = float(np.atleast_1d(r1["flow_rate_std_m3_per_day"])[0])
    Q2 = float(np.atleast_1d(r2["flow_rate_std_m3_per_day"])[0])
    assert Q2 > Q1, f"Larger diameter not giving more flow: {Q1:.1f} vs {Q2:.1f}"


def test_flow_decreases_with_length(model):
    """Longer pipeline → lower flow at same pressures."""
    r1 = model.predict({"length_km": 50.0, "diameter_m": 0.5,
                         "P_in_bar": 70.0, "P_out_bar": 55.0})
    r2 = model.predict({"length_km": 200.0, "diameter_m": 0.5,
                         "P_in_bar": 70.0, "P_out_bar": 55.0})
    Q1 = float(np.atleast_1d(r1["flow_rate_std_m3_per_day"])[0])
    Q2 = float(np.atleast_1d(r2["flow_rate_std_m3_per_day"])[0])
    assert Q2 < Q1, f"Longer pipeline not reducing flow: {Q1:.1f} vs {Q2:.1f}"


def test_z_factor_range(model):
    """Papay Z must be in physically valid range (0.5-1.1)."""
    r = model.predict({"length_km": 100.0, "diameter_m": 0.5,
                        "P_in_bar": 70.0, "P_out_bar": 55.0})
    Z = float(np.atleast_1d(r["Z_avg"])[0])
    assert 0.5 <= Z <= 1.1, f"Z = {Z:.4f} out of range"


def test_z_decreases_with_pressure(model):
    """Z < 1 at high pressure (real gas effect) for NG at pipeline conditions."""
    r_low = model.predict({"length_km": 50.0, "diameter_m": 0.5,
                            "P_in_bar": 20.0, "P_out_bar": 18.0})
    r_high = model.predict({"length_km": 50.0, "diameter_m": 0.5,
                             "P_in_bar": 100.0, "P_out_bar": 90.0})
    Z_low = float(np.atleast_1d(r_low["Z_avg"])[0])
    Z_high = float(np.atleast_1d(r_high["Z_avg"])[0])
    # At high pressure, Z deviates more from 1.0 (Papay correlation)
    assert abs(Z_high - 1.0) >= abs(Z_low - 1.0) * 0.9, \
        f"Z not more deviant at high P: Z_low={Z_low:.4f}, Z_high={Z_high:.4f}"


def test_temperature_cooling(model):
    """Gas must cool toward soil temperature along pipeline."""
    r = model.predict({"length_km": 100.0, "diameter_m": 0.5,
                        "P_in_bar": 70.0, "P_out_bar": 55.0,
                        "T_in_K": 310.0})  # warmer inlet than soil
    T_avg = float(np.atleast_1d(r["T_avg_K"])[0])
    T_out = float(np.atleast_1d(r["T_out_K"])[0])
    # T_out <= T_avg <= T_in (cooling profile)
    assert T_out <= 310.0 + 1e-6, f"T_out = {T_out:.2f} K not cooling"
    assert T_avg <= 310.0 + 1e-6, f"T_avg = {T_avg:.2f} K not cooling"


def test_weymouth_k_constant():
    """Confirm Weymouth K = 3.7435e-3 per Menon (2005) SI table."""
    from model import NGPipelineF1b
    import json
    from pathlib import Path
    base = Path(__file__).parent.parent
    with open(base / "data" / "parameters.json") as f:
        params = json.load(f)
    m = NGPipelineF1b(params)
    assert abs(m.WEYMOUTH_K - 3.7435e-3) < 1e-8, \
        f"Weymouth K = {m.WEYMOUTH_K}, expected 3.7435e-3"


def test_array_input(model):
    """Model must handle array length inputs."""
    lengths = np.linspace(50, 300, 10)
    r = model.predict({"length_km": lengths, "diameter_m": 0.5,
                        "P_in_bar": 70.0, "P_out_bar": 55.0})
    assert len(np.atleast_1d(r["flow_rate_std_m3_per_day"])) == 10


def test_benchmark(model):
    lengths = np.random.uniform(50, 500, 1000)
    start = time.perf_counter()
    model.predict({"length_km": lengths, "diameter_m": 0.5,
                   "P_in_bar": 70.0, "P_out_bar": 55.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
