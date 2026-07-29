"""EC204 — Calcium Looping — F1b Part-Load + Sorbent Degradation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0, "n_cycles": 0})
    for k in ["co2_captured_kg_h", "calcination_duty_gj_ton", "electrical_kwh_ton",
              "sorbent_activity_pct", "total_energy_penalty_pct"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC204"
    assert info["fidelity"] == "F1b"


def test_calcination_duty_at_design(model):
    """At PLR=1.0, fresh sorbent, CR=0.90: duty should be ~3.2 GJ/tCO2."""
    r = model.predict({"PLR": 1.0, "n_cycles": 0, "capture_rate": 0.90})
    q = float(np.atleast_1d(r["calcination_duty_gj_ton"])[0])
    assert 2.5 < q < 5.0, f"Calcination duty at design = {q:.2f}"


def test_calcination_increases_at_part_load(model):
    """Calcination duty increases at lower PLR."""
    PLRs = [0.3, 0.5, 0.7, 1.0]
    qs = [float(np.atleast_1d(model.predict({"PLR": p, "n_cycles": 0})["calcination_duty_gj_ton"])[0])
          for p in PLRs]
    assert qs[0] > qs[-1], f"Duty not higher at part-load: {qs}"


def test_calcination_increases_with_cycles(model):
    """Calcination duty increases with cycle count (degraded sorbent)."""
    cycles = [0, 10000, 50000, 100000]
    qs = [float(np.atleast_1d(model.predict({"PLR": 1.0, "n_cycles": c})["calcination_duty_gj_ton"])[0])
          for c in cycles]
    assert all(qs[i] <= qs[i + 1] + 1e-6 for i in range(len(qs) - 1)), \
        f"Duty not monotone with cycles: {qs}"


def test_activity_100_pct_at_start(model):
    """Fresh sorbent: activity should be 100%."""
    r = model.predict({"PLR": 1.0, "n_cycles": 0})
    act = float(np.atleast_1d(r["sorbent_activity_pct"])[0])
    assert abs(act - 100.0) < 1e-6, f"Activity at N=0 = {act:.2f}%"


def test_activity_declines_with_cycles(model):
    """Sorbent activity must decrease with number of cycles."""
    cycles = [0, 1000, 10000, 50000]
    acts = [float(np.atleast_1d(model.predict({"PLR": 1.0, "n_cycles": c})["sorbent_activity_pct"])[0])
            for c in cycles]
    assert all(acts[i] >= acts[i + 1] - 1e-6 for i in range(len(acts) - 1)), \
        f"Activity not declining: {acts}"


def test_activity_residual_floor(model):
    """At very large N, activity approaches residual (~9.375% = X_INF/X0)."""
    r = model.predict({"PLR": 1.0, "n_cycles": 1e6})
    act = float(np.atleast_1d(r["sorbent_activity_pct"])[0])
    # With makeup, floor is above X_INF/X0 due to fresh CaO addition
    assert act > 5.0, f"Activity floor too low: {act:.2f}%"


def test_co2_captured_positive(model):
    r = model.predict({"flue_gas_flow_mol_s": 100.0, "co2_concentration": 0.12,
                        "capture_rate": 0.90, "PLR": 0.5, "n_cycles": 0})
    co2 = float(np.atleast_1d(r["co2_captured_kg_h"])[0])
    assert co2 > 0


def test_co2_scales_with_plr(model):
    """CO2 captured scales with PLR."""
    r1 = model.predict({"PLR": 0.5, "n_cycles": 0})
    r2 = model.predict({"PLR": 1.0, "n_cycles": 0})
    co2_1 = float(np.atleast_1d(r1["co2_captured_kg_h"])[0])
    co2_2 = float(np.atleast_1d(r2["co2_captured_kg_h"])[0])
    assert co2_2 > co2_1


def test_electrical_positive(model):
    r = model.predict({"PLR": 0.5, "n_cycles": 0})
    e = float(np.atleast_1d(r["electrical_kwh_ton"])[0])
    assert e > 0


def test_energy_penalty_bounded(model):
    """Energy penalty 8-55%."""
    r = model.predict({"PLR": 0.5, "n_cycles": 20000})
    pen = float(np.atleast_1d(r["total_energy_penalty_pct"])[0])
    assert 8.0 <= pen <= 55.0, f"Energy penalty = {pen:.1f}%"


def test_array_input(model):
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"PLR": PLRs, "n_cycles": 0})
    assert len(np.atleast_1d(r["calcination_duty_gj_ton"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLRs, "n_cycles": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
