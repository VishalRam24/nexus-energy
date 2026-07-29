"""EC191 — Gas Compressor Station — F1b Part-Load + Inlet Temperature — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0})
    for k in ["specific_work_kJ_per_kg", "sec_kwh_per_kg", "shaft_power_kw",
              "discharge_temperature_K", "polytropic_efficiency", "overall_efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC191"
    assert info["fidelity"] == "F1b"


def test_efficiency_at_design(model):
    """At PLR=1.0, polytropic efficiency should be design value ~0.82."""
    r = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                        "PLR": 1.0})
    eta = float(np.atleast_1d(r["polytropic_efficiency"])[0])
    assert 0.75 < eta < 0.92, f"Polytropic efficiency at design = {eta:.4f}"


def test_efficiency_decreases_at_partload(model):
    """Polytropic efficiency must decrease at part-load."""
    r_full = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                             "PLR": 1.0})
    r_half = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                             "PLR": 0.5})
    eta_full = float(np.atleast_1d(r_full["polytropic_efficiency"])[0])
    eta_half = float(np.atleast_1d(r_half["polytropic_efficiency"])[0])
    assert eta_half < eta_full, f"Efficiency not lower at part-load: {eta_full:.4f} vs {eta_half:.4f}"


def test_work_increases_with_pressure_ratio(model):
    """Higher pressure ratio → more specific work."""
    r1 = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 60.0})
    r2 = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 100.0})
    w1 = float(np.atleast_1d(r1["specific_work_kJ_per_kg"])[0])
    w2 = float(np.atleast_1d(r2["specific_work_kJ_per_kg"])[0])
    assert w2 > w1, f"Work not increasing with PR: {w1:.1f} vs {w2:.1f}"


def test_work_increases_with_inlet_temperature(model):
    """Hotter inlet → more compression work (proportional to T1)."""
    r_cold = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                             "T_in_K": 260.0})
    r_hot = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                            "T_in_K": 320.0})
    w_cold = float(np.atleast_1d(r_cold["specific_work_kJ_per_kg"])[0])
    w_hot = float(np.atleast_1d(r_hot["specific_work_kJ_per_kg"])[0])
    assert w_hot > w_cold, f"Work not increasing with T_in: {w_cold:.2f} vs {w_hot:.2f}"


def test_discharge_temperature_above_inlet(model):
    """Discharge temperature must exceed inlet temperature."""
    r = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                        "T_in_K": 288.15, "PLR": 1.0})
    T2 = float(np.atleast_1d(r["discharge_temperature_K"])[0])
    assert T2 > 288.15, f"Discharge T = {T2:.2f} K not above inlet 288.15 K"


def test_power_positive(model):
    """Shaft power must be positive."""
    r = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0})
    P = float(np.atleast_1d(r["shaft_power_kw"])[0])
    assert P > 0, f"Power = {P:.1f} kW"


def test_overall_efficiency_bounded(model):
    """Overall efficiency must be in (0, 1)."""
    for plr in [0.3, 0.5, 1.0]:
        r = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                            "PLR": plr})
        eta = float(np.atleast_1d(r["overall_efficiency"])[0])
        assert 0.0 < eta < 1.0, f"Overall efficiency = {eta:.4f} at PLR={plr}"


def test_array_input(model):
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                        "PLR": PLRs})
    assert len(np.atleast_1d(r["polytropic_efficiency"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0, "PLR": PLRs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
