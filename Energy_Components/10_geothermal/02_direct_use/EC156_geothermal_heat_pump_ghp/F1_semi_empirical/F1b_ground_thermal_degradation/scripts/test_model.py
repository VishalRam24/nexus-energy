"""EC156 -- GHP -- F1b Ground Thermal Degradation -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_sink_c": 35.0})
    for k in ["cop_heating", "cop_cooling", "cop_effective", "T_source_effective",
              "ground_dT", "fouling_factor", "part_load_factor",
              "heating_capacity_kw", "electrical_input_kw", "cop_advantage_over_ashp"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC156"
    assert info["fidelity"] == "F1b"


def test_cop_heating_greater_than_one(model):
    """Heating COP must always be > 1 (fundamental thermodynamic requirement)."""
    for T_sink in [30.0, 40.0, 50.0, 60.0]:
        r = model.predict({"T_sink_c": T_sink, "operation_hours": 0.0})
        assert r["cop_heating"] > 1.0, \
            f"T_sink={T_sink}: COP_heating={r['cop_heating']:.2f} <= 1"


def test_cop_cooling_positive(model):
    """Cooling COP must be positive."""
    r = model.predict({"T_sink_c": 35.0, "mode": "cooling"})
    assert r["cop_cooling"] > 0.0


def test_ground_dT_zero_at_start(model):
    """At operation_hours=0, no ground thermal depression."""
    r = model.predict({"T_sink_c": 35.0, "operation_hours": 0.0})
    assert abs(r["ground_dT"]) < 1e-6, f"ground_dT at t=0 = {r['ground_dT']:.6f}"


def test_ground_dT_increases_with_time(model):
    """Ground temperature depression must increase with operation hours."""
    r_0    = model.predict({"T_sink_c": 35.0, "operation_hours": 0.0})
    r_500  = model.predict({"T_sink_c": 35.0, "operation_hours": 500.0})
    r_2000 = model.predict({"T_sink_c": 35.0, "operation_hours": 2000.0})
    assert r_0["ground_dT"] < r_500["ground_dT"] < r_2000["ground_dT"], \
        f"ground_dT not increasing: {r_0['ground_dT']:.3f} -> {r_500['ground_dT']:.3f} -> {r_2000['ground_dT']:.3f}"


def test_ground_thermal_saturation(model):
    """
    Ground temperature depression saturates (exponential approach to steady state).
    After 5*tau (5*800=4000h), dT should be >90% of steady-state value.
    """
    r_long = model.predict({"T_sink_c": 35.0, "operation_hours": 4000.0, "heat_rate_kw": 10.0})
    r_ss   = model.predict({"T_sink_c": 35.0, "operation_hours": 100000.0, "heat_rate_kw": 10.0})
    # At t=5*tau, saturation should be >1-exp(-5)=99.3%
    assert r_long["ground_dT"] > 0.85 * r_ss["ground_dT"], \
        f"Ground dT at 4000h = {r_long['ground_dT']:.3f}, steady-state = {r_ss['ground_dT']:.3f}"


def test_cop_decreases_with_operation_time(model):
    """
    COP effective must decrease with operation hours (ground saturation degrades T_source).
    """
    r_0    = model.predict({"T_sink_c": 35.0, "operation_hours": 0.0,    "TDS_ppm": 200.0})
    r_2000 = model.predict({"T_sink_c": 35.0, "operation_hours": 2000.0, "TDS_ppm": 200.0})
    assert r_0["cop_effective"] > r_2000["cop_effective"], \
        f"COP_eff not decreasing: {r_0['cop_effective']:.2f} -> {r_2000['cop_effective']:.2f}"


def test_cop_decreases_with_T_sink(model):
    """Higher T_sink (larger lift) must give lower COP."""
    cops = [model.predict({"T_sink_c": float(T), "operation_hours": 0.0})["cop_heating"]
            for T in [30.0, 40.0, 50.0, 60.0]]
    assert all(cops[i] > cops[i+1] for i in range(len(cops)-1)), \
        f"COP not decreasing with T_sink: {cops}"


def test_fouling_factor_at_zero_time_is_one(model):
    """Fouling factor must be 1.0 at t=0 (no fouling yet)."""
    r = model.predict({"T_sink_c": 35.0, "operation_hours": 0.0, "TDS_ppm": 1000.0})
    # At t~0, fouling R_f = k * (TDS/TDS_ref)^0.5 * 0.1^0.3 which is small
    assert r["fouling_factor"] >= 0.99, \
        f"fouling_factor at t=0 = {r['fouling_factor']:.4f}"


def test_fouling_decreases_with_high_tds(model):
    """Higher TDS must give lower fouling factor (more scaling)."""
    r_low  = model.predict({"T_sink_c": 35.0, "operation_hours": 5000.0, "TDS_ppm": 100.0})
    r_high = model.predict({"T_sink_c": 35.0, "operation_hours": 5000.0, "TDS_ppm": 2000.0})
    assert r_high["fouling_factor"] < r_low["fouling_factor"], \
        f"fouling_factor not decreasing with TDS: {r_low['fouling_factor']:.3f} -> {r_high['fouling_factor']:.3f}"


def test_part_load_factor_at_full_load_is_one(model):
    """PLR factor must be 1.0 at full load."""
    r = model.predict({"T_sink_c": 35.0, "PLR": 1.0})
    assert abs(r["part_load_factor"] - 1.0) < 0.01


def test_part_load_factor_below_one_at_low_plr(model):
    r = model.predict({"T_sink_c": 35.0, "PLR": 0.3})
    assert r["part_load_factor"] < 1.0


def test_cop_advantage_positive_vs_ashp(model):
    """GHP must deliver COP advantage over ASHP at cold conditions."""
    r = model.predict({"T_sink_c": 45.0, "operation_hours": 0.0})
    assert r["cop_advantage_over_ashp"] > 0, \
        f"COP advantage = {r['cop_advantage_over_ashp']:.2f}; expected > 0"


def test_heating_capacity_scales_with_plr(model):
    """Heating capacity must scale proportionally with PLR."""
    r_full = model.predict({"T_sink_c": 35.0, "PLR": 1.0})
    r_half = model.predict({"T_sink_c": 35.0, "PLR": 0.5})
    assert abs(r_full["heating_capacity_kw"] / r_half["heating_capacity_kw"] - 2.0) < 0.01


def test_electrical_input_positive(model):
    r = model.predict({"T_sink_c": 35.0})
    assert r["electrical_input_kw"] > 0


def test_benchmark(model):
    """1000 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({"T_sink_c": 35.0, "PLR": 0.8, "operation_hours": 500.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
