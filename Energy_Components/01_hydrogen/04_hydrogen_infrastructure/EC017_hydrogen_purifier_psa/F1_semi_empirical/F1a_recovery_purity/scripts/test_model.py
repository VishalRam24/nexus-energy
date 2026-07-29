"""EC017 — Hydrogen Purifier (PSA) — F1a Recovery-Purity — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def _base_inputs(**overrides):
    inp = {
        "feed_flow_kg_s": 0.1,
        "feed_h2_fraction": 0.75,
        "feed_pressure_bar": 20.0,
        "target_purity": 0.9999,
    }
    inp.update(overrides)
    return inp


def test_predict_keys(model):
    r = model.predict(_base_inputs())
    for k in ["recovery", "product_flow_kg_s", "tail_gas_flow_kg_s",
              "specific_energy_kWh_per_kg", "electric_power_kW",
              "pressure_ratio", "h2_yield_kg_s"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC017"
    assert info["fidelity"] == "F1a"


def test_recovery_strictly_below_one(model):
    """
    Recovery must be < 1 (thermodynamic limit: some H2 lost to purge stream).
    Sircar & Golden (2000): practical PSA recovery 70-90%.
    """
    for P in [10, 20, 40, 60, 80]:
        r = model.predict(_base_inputs(feed_pressure_bar=P))
        eta = float(r["recovery"])
        assert eta < 1.0, f"Recovery must be < 1.0 at P={P} bar, got {eta:.4f}"


def test_recovery_strictly_above_zero(model):
    """Recovery must be > 0 for any valid operating condition."""
    r = model.predict(_base_inputs())
    assert float(r["recovery"]) > 0.0


def test_recovery_increases_with_feed_pressure(model):
    """
    Higher feed pressure → better adsorption selectivity → higher recovery.
    Sircar & Golden (2000) Sep. Sci. Technol. 35(5): recovery improves with P_H/P_L ratio.
    """
    P_arr = np.array([10.0, 20.0, 30.0, 50.0, 70.0])
    recoveries = []
    for P in P_arr:
        r = model.predict(_base_inputs(feed_pressure_bar=P))
        recoveries.append(float(r["recovery"]))
    assert np.all(np.diff(recoveries) > 0), \
        "Recovery must increase monotonically with feed pressure"


def test_recovery_increases_with_feed_h2_fraction(model):
    """
    Richer feed (higher y_H2) → easier separation → higher recovery at fixed conditions.
    Sircar & Golden (2000): recovery depends on feed composition.
    """
    y_arr = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    recoveries = []
    for y in y_arr:
        r = model.predict(_base_inputs(feed_h2_fraction=y))
        recoveries.append(float(r["recovery"]))
    assert np.all(np.diff(recoveries) > 0), \
        "Recovery must increase with feed H2 fraction"


def test_recovery_in_nominal_range(model):
    """
    Nominal PSA recovery 70–90% at reference conditions.
    Sircar & Golden (2000): industrial H2 PSA recovery typically 70-90%.
    """
    r = model.predict(_base_inputs())
    eta = float(r["recovery"])
    assert 0.65 < eta < 0.95, \
        f"Nominal recovery should be 70-90%, got {eta*100:.1f}%"


def test_mass_balance(model):
    """
    Mass balance: F_product + F_tail = F_feed (within 1% relative).
    """
    F_feed = 0.1  # kg/s
    r = model.predict(_base_inputs(feed_flow_kg_s=F_feed))
    F_out = float(r["product_flow_kg_s"]) + float(r["tail_gas_flow_kg_s"])
    assert abs(F_out - F_feed) / F_feed < 0.01, \
        f"Mass balance violation: F_in={F_feed:.4f}, F_out={F_out:.4f} kg/s"


def test_tail_gas_positive(model):
    """Tail gas flow must be positive (some feed is always lost to purge)."""
    r = model.predict(_base_inputs())
    assert float(r["tail_gas_flow_kg_s"]) > 0.0


def test_product_flow_less_than_feed(model):
    """Product flow must be less than feed flow (mass balance constraint)."""
    r = model.predict(_base_inputs())
    assert float(r["product_flow_kg_s"]) < float(_base_inputs()["feed_flow_kg_s"])


def test_h2_yield_consistent_with_recovery(model):
    """
    H2 yield = feed_flow * y_H2 * recovery.
    Verify: h2_yield_kg_s ≈ F_feed * y_H2 * recovery (within 1% due to purity correction).
    """
    F = 0.1
    y = 0.75
    r = model.predict(_base_inputs(feed_flow_kg_s=F, feed_h2_fraction=y))
    eta = float(r["recovery"])
    expected_approx = F * y * eta
    actual = float(r["h2_yield_kg_s"])
    # product purity (99.99%) causes <0.01% deviation
    assert abs(actual - expected_approx) / expected_approx < 0.01, \
        f"H2 yield inconsistent: expected ~{expected_approx:.5f}, got {actual:.5f} kg/s"


def test_specific_energy_in_physical_range(model):
    """
    PSA specific energy 1–3 kWh/kg H2 per DOE targets.
    DOE Hydrogen Program Roadmap: 1-3 kWh/kg H2 for PSA purification.
    """
    for P in [10, 20, 40, 60]:
        r = model.predict(_base_inputs(feed_pressure_bar=P))
        W = float(r["specific_energy_kWh_per_kg"])
        assert 0.5 < W < 5.0, \
            f"Specific energy at P={P} bar should be 0.5-5 kWh/kg, got {W:.2f} kWh/kg"


def test_specific_energy_decreases_with_pressure(model):
    """
    Higher feed pressure → less vacuum-swing energy → lower specific energy.
    Sircar & Golden (2000): energy scales inversely with pressure ratio.
    # RATIONALE: At higher feed pressure, the ΔP across the bed is larger relative to purge,
    # reducing the blower/compressor energy needed per kg H2 recovered. DOE H2 Program data.
    """
    P_arr = np.array([5.0, 10.0, 20.0, 40.0, 60.0])
    W_arr = []
    for P in P_arr:
        r = model.predict(_base_inputs(feed_pressure_bar=P))
        W_arr.append(float(r["specific_energy_kWh_per_kg"]))
    assert np.all(np.diff(W_arr) <= 0), \
        "Specific energy must not increase with increasing feed pressure"


def test_electric_power_positive(model):
    r = model.predict(_base_inputs())
    assert float(r["electric_power_kW"]) > 0.0


def test_pressure_ratio_above_one(model):
    """Feed pressure must be above purge pressure; ratio must be > 1."""
    r = model.predict(_base_inputs(feed_pressure_bar=20.0))
    assert float(r["pressure_ratio"]) > 1.0


def test_higher_purity_reduces_recovery(model):
    """
    Demanding higher purity sacrifices recovery (PSA trade-off).
    Sircar & Golden (2000): purity-recovery trade-off is fundamental to PSA design.
    """
    r_low = model.predict(_base_inputs(target_purity=0.99))
    r_high = model.predict(_base_inputs(target_purity=0.99999))
    assert float(r_low["recovery"]) >= float(r_high["recovery"]), \
        "Higher target purity should give equal or lower recovery"


def test_vectorized_pressure(model):
    """Model must handle array pressure inputs."""
    P = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    r = model.predict({
        "feed_flow_kg_s": 0.1,
        "feed_h2_fraction": 0.75,
        "feed_pressure_bar": P,
        "target_purity": 0.9999,
    })
    assert r["recovery"].shape == P.shape
    assert r["product_flow_kg_s"].shape == P.shape


def test_benchmark(model):
    F = np.random.uniform(0.001, 1.0, 1000)
    P = np.random.uniform(5.0, 80.0, 1000)
    y = np.random.uniform(0.3, 0.99, 1000)
    start = time.perf_counter()
    model.predict({"feed_flow_kg_s": F, "feed_h2_fraction": y,
                   "feed_pressure_bar": P, "target_purity": 0.9999})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
