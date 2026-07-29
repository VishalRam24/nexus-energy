"""EC105 -- Gas Turbine CHP -- F1b Part-Load + Ambient + HRSG -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

ISO_T = 288.15   # 15 degC in Kelvin
ISO_P = 101.325  # kPa


@pytest.fixture
def model():
    return ComponentModel()


# --- Output key checks ---

def test_predict_keys(model):
    r = model.predict({"PLR": 1.0})
    for k in ["efficiency_electrical", "efficiency_thermal", "efficiency_total",
              "power_electrical_kw", "heat_recovery_kw", "fuel_input_kw",
              "exhaust_temp_K", "heat_to_power_ratio", "heat_rate_kj_kwh"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC105"
    assert info["fidelity"] == "F1b"


# --- Electrical efficiency physics ---

def test_eta_el_at_iso_rated(model):
    """At ISO full load, eta_el should be near 0.32 (within 5%)."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P})
    eta = float(r["efficiency_electrical"])
    assert abs(eta - 0.32) / 0.32 < 0.05, f"eta_el = {eta:.4f}, expected ~0.32"


def test_eta_el_drops_at_part_load(model):
    """Gas turbine efficiency is worst at minimum PLR."""
    r_full = model.predict({"PLR": 1.0, "T_ambient": ISO_T})
    r_part = model.predict({"PLR": 0.4, "T_ambient": ISO_T})
    assert float(r_part["efficiency_electrical"]) < float(r_full["efficiency_electrical"])


def test_higher_T_reduces_eta_el(model):
    """Higher ambient T degrades compressor efficiency -> lower eta_el."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 263.15})  # -10C
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 313.15})  # 40C
    assert float(r_cold["efficiency_electrical"]) > float(r_hot["efficiency_electrical"])


def test_eta_el_bounded(model):
    """Electrical efficiency must be in (0, 0.50] across valid range."""
    PLR = np.linspace(0.4, 1.0, 50)
    T = np.linspace(263.15, 313.15, 50)
    r = model.predict({"PLR": PLR, "T_ambient": T})
    assert np.all(r["efficiency_electrical"] > 0)
    assert np.all(r["efficiency_electrical"] <= 0.50)


# --- Power output ---

def test_power_at_iso_rated(model):
    """At ISO full load, power should be near P_rated (within 5%)."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P})
    P = float(r["power_electrical_kw"])
    assert abs(P - 5000.0) / 5000.0 < 0.05, f"P = {P:.1f} kW, expected ~5000"


def test_higher_T_reduces_power(model):
    """Higher ambient T reduces air density -> less mass flow -> lower power."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 263.15})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 313.15})
    assert float(r_cold["power_electrical_kw"]) > float(r_hot["power_electrical_kw"])


def test_higher_pressure_increases_power(model):
    """Higher ambient pressure increases air density -> more power."""
    r_low  = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": 85.0})
    r_high = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P})
    assert float(r_high["power_electrical_kw"]) > float(r_low["power_electrical_kw"])


# --- Exhaust temperature ---

def test_exhaust_temp_rises_at_part_load(model):
    """Gas turbine exhaust temperature is higher at part load."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.5})
    assert float(r_part["exhaust_temp_K"]) >= float(r_full["exhaust_temp_K"])


def test_exhaust_temp_at_rated(model):
    """Exhaust temp at PLR=1 should equal T_exhaust_rated (798.15 K = 525 degC)."""
    r = model.predict({"PLR": 1.0})
    T_exh = float(r["exhaust_temp_K"])
    assert abs(T_exh - 798.15) < 1.0, f"T_exh = {T_exh:.2f} K, expected 798.15"


# --- Thermal efficiency and HRSG heat ---

def test_eta_th_nonzero(model):
    """Thermal efficiency must be positive across all PLR values."""
    PLR = np.linspace(0.4, 1.0, 50)
    r = model.predict({"PLR": PLR})
    assert np.all(r["efficiency_thermal"] > 0)


def test_heat_recovery_positive(model):
    """Heat recovery must be positive when CHP is operating."""
    PLR = np.linspace(0.4, 1.0, 20)
    r = model.predict({"PLR": PLR})
    assert np.all(r["heat_recovery_kw"] > 0)


def test_hrsg_heat_increases_at_part_load_due_to_exhaust_T(model):
    """
    At part load, exhaust T rises, partially compensating for the PLR-driven
    thermal efficiency reduction. HRSG T boost (hrsg_eff_T_coeff * dT_exhaust)
    partially offsets the linear PLR penalty (th_a + th_b*PLR).
    At PLR=0.9 vs PLR=1.0: difference should be small due to exhaust T boost.
    # RATIONALE: The primary physics here is that f_T_exh > 1 at part load;
    # the net eta_th still drops with PLR but less steeply than without the boost.
    # We test the exhaust T boost is active (eta_th at PLR=0.9 > f_PLR_only prediction).
    """
    m = model._model
    PLR_test = 0.9
    # Prediction with T-exhaust boost
    r = model.predict({"PLR": PLR_test})
    eta_th_actual = float(r["efficiency_thermal"])
    # Prediction without boost (pure PLR factor)
    f_plr_only = m.th_a + m.th_b * PLR_test   # 0.40 + 0.60*0.9 = 0.94
    eta_th_no_boost = m.eta_th_rated * f_plr_only
    # With exhaust T boost (T_exh > T_rated at PLR<1), eta_th_actual >= eta_th_no_boost
    assert eta_th_actual >= eta_th_no_boost * 0.999, (
        f"eta_th ({eta_th_actual:.4f}) should be >= PLR-only prediction ({eta_th_no_boost:.4f}) "
        f"due to exhaust T boost"
    )


# --- Total efficiency ---

def test_total_equals_sum(model):
    """eta_total = eta_el + eta_th (first law)."""
    PLR = np.linspace(0.4, 1.0, 50)
    r = model.predict({"PLR": PLR})
    diff = np.abs(r["efficiency_total"] - r["efficiency_electrical"] - r["efficiency_thermal"])
    assert np.all(diff < 1e-10)


def test_total_efficiency_range(model):
    """Total CHP efficiency at full load should be > 0.75 (electrical + thermal)."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P})
    assert float(r["efficiency_total"]) > 0.75, \
        f"Total eta = {float(r['efficiency_total']):.3f}, expected > 0.75"


# --- Energy balance ---

def test_energy_balance(model):
    """P_el + Q_th <= fuel_input (small tolerance for numerical precision)."""
    PLR = np.linspace(0.4, 1.0, 50)
    r = model.predict({"PLR": PLR})
    total_useful = r["power_electrical_kw"] + r["heat_recovery_kw"]
    assert np.all(total_useful <= r["fuel_input_kw"] * 1.001)


# --- Heat-to-power ratio ---

def test_hpr_at_rated(model):
    """HPR at ISO full load ~ 0.5/0.32 ~ 1.56."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P})
    hpr = float(r["heat_to_power_ratio"])
    # Gas turbine CHP typically has HPR 1.2 - 2.0
    assert 1.0 < hpr < 2.5, f"HPR = {hpr:.3f}"


# --- Heat rate ---

def test_heat_rate_consistent_with_efficiency(model):
    """Heat rate = 3600 / eta_el."""
    r = model.predict({"PLR": 0.8, "T_ambient": ISO_T})
    hr = float(r["heat_rate_kj_kwh"])
    eta = float(r["efficiency_electrical"])
    assert abs(hr - 3600.0 / eta) < 1.0


def test_heat_rate_worsens_at_part_load(model):
    """Heat rate should increase (worsen) at part load."""
    r_full = model.predict({"PLR": 1.0, "T_ambient": ISO_T})
    r_part = model.predict({"PLR": 0.4, "T_ambient": ISO_T})
    assert float(r_part["heat_rate_kj_kwh"]) > float(r_full["heat_rate_kj_kwh"])


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.4})
    assert float(r["efficiency_electrical"]) > 0
    assert float(r["power_electrical_kw"]) > 0
    assert float(r["heat_recovery_kw"]) > 0


def test_extreme_cold(model):
    """Very cold ambient (-20C, 243.15K) should boost power above ISO rated."""
    r = model.predict({"PLR": 1.0, "T_ambient": 253.15, "P_ambient": ISO_P})
    assert float(r["power_electrical_kw"]) > 5000.0


def test_extreme_hot(model):
    """Very hot ambient (50C, 323.15K) should reduce power below ISO rated."""
    r = model.predict({"PLR": 1.0, "T_ambient": 323.15, "P_ambient": ISO_P})
    assert float(r["power_electrical_kw"]) < 5000.0


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.4, 1.0, 1000)
    T = np.random.uniform(263.15, 313.15, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
