"""
EC015 -- Chemical H2 Storage (LOHC / Ammonia) -- F1b Thermal+Part-load -- Test Suite
Run: python -m pytest test_model.py -v
"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Output structure ---

def test_predict_lohc_dehydro_returns_keys(model):
    r = model.predict({"carrier": "lohc", "direction": "dehydrogenation", "h2_mass_kg": 1.0})
    for k in ["thermal_energy_MJ", "specific_energy_MJ_per_kg", "efficiency",
              "carrier_mass_kg", "roundtrip_efficiency"]:
        assert k in r


def test_predict_nh3_crack_returns_keys(model):
    r = model.predict({"carrier": "ammonia", "direction": "cracking", "h2_mass_kg": 1.0})
    for k in ["thermal_energy_MJ", "specific_energy_MJ_per_kg", "efficiency",
              "carrier_mass_kg", "roundtrip_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC015"
    assert info["fidelity"] == "F1b"


# --- Dehydrogenation: positive heat demand ---

def test_lohc_dehydro_positive_energy(model):
    """Dehydrogenation requires heat (endothermic) -> positive Q."""
    r = model.predict({"carrier": "lohc", "direction": "dehydrogenation", "h2_mass_kg": 1.0})
    assert float(r["thermal_energy_MJ"]) > 0, "Dehydrogenation must require heat"


def test_nh3_cracking_positive_energy(model):
    """Cracking requires heat (endothermic) -> positive Q."""
    r = model.predict({"carrier": "ammonia", "direction": "cracking", "h2_mass_kg": 1.0})
    assert float(r["thermal_energy_MJ"]) > 0, "Cracking must require heat"


# --- Hydrogenation/synthesis: negative heat (exothermic) ---

def test_lohc_hydro_negative_energy(model):
    """Hydrogenation is exothermic -> negative Q (heat released)."""
    r = model.predict({"carrier": "lohc", "direction": "hydrogenation", "h2_mass_kg": 1.0})
    assert float(r["thermal_energy_MJ"]) < 0, "Hydrogenation must release heat"


def test_nh3_synthesis_negative_energy(model):
    """Synthesis is exothermic -> negative Q."""
    r = model.predict({"carrier": "ammonia", "direction": "synthesis", "h2_mass_kg": 1.0})
    assert float(r["thermal_energy_MJ"]) < 0, "NH3 synthesis must release heat"


# --- Temperature effect: higher T -> higher efficiency (Arrhenius) ---

def test_lohc_efficiency_increases_with_temperature(model):
    """Higher reactor T -> better conversion -> higher efficiency."""
    eta_lo = float(model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                                   "h2_mass_kg": 1.0, "temperature_K": 523.15})["efficiency"])
    eta_hi = float(model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                                   "h2_mass_kg": 1.0, "temperature_K": 623.15})["efficiency"])
    assert eta_hi > eta_lo, f"Efficiency at 623K ({eta_hi:.3f}) must exceed 523K ({eta_lo:.3f})"


def test_lohc_heat_decreases_with_temperature(model):
    """Higher efficiency at higher T -> less heat needed per kg H2."""
    q_lo = float(model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                                 "h2_mass_kg": 1.0, "temperature_K": 523.15})["thermal_energy_MJ"])
    q_hi = float(model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                                 "h2_mass_kg": 1.0, "temperature_K": 623.15})["thermal_energy_MJ"])
    assert q_hi < q_lo, "Higher T should reduce specific heat demand"


# --- Part-load: lower flow rate -> lower efficiency ---

def test_lohc_efficiency_decreases_at_part_load(model):
    """Part-load operation reduces efficiency relative to full load."""
    T = 573.15
    eta_full = float(model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                                     "h2_mass_kg": 1.0, "temperature_K": T,
                                     "flow_rate_kg_s": 0.01})["efficiency"])
    eta_part = float(model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                                     "h2_mass_kg": 1.0, "temperature_K": T,
                                     "flow_rate_kg_s": 0.003})["efficiency"])
    assert eta_full >= eta_part, f"Full-load eta={eta_full:.3f} must >= part-load eta={eta_part:.3f}"


# --- Round-trip efficiency bounded ---

def test_roundtrip_efficiency_in_range(model):
    """Round-trip efficiency must be in (0, 1)."""
    for carrier in ["lohc", "ammonia"]:
        r = model.predict({"carrier": carrier, "direction": "dehydrogenation" if carrier == "lohc" else "cracking",
                           "h2_mass_kg": 1.0, "temperature_K": 573.15 if carrier == "lohc" else 773.15})
        rt = float(r["roundtrip_efficiency"])
        assert 0.0 < rt < 1.0, f"{carrier} round-trip efficiency {rt:.3f} must be in (0, 1)"


# --- Carrier mass positive ---

def test_carrier_mass_positive(model):
    """Carrier mass must be positive for both carriers."""
    for carrier in ["lohc", "ammonia"]:
        r = model.predict({"carrier": carrier, "h2_mass_kg": 1.0})
        assert float(r["carrier_mass_kg"]) > 0, f"{carrier} carrier mass must be positive"


# --- NH3 has higher gravimetric capacity than LOHC ---

def test_nh3_carrier_mass_less_than_lohc(model):
    """NH3 has higher H2 capacity (17.6 wt%) vs LOHC (6.2 wt%) -> less carrier mass per kg H2."""
    r_lohc = model.predict({"carrier": "lohc", "h2_mass_kg": 1.0})
    r_nh3  = model.predict({"carrier": "ammonia", "h2_mass_kg": 1.0})
    assert float(r_nh3["carrier_mass_kg"]) < float(r_lohc["carrier_mass_kg"]), \
        "NH3 carrier mass must be less than LOHC for same H2 (higher gravimetric capacity)"


# --- Array inputs ---

def test_array_h2_mass(model):
    """Model must handle array of H2 masses."""
    m = np.array([0.5, 1.0, 2.0, 5.0])
    r = model.predict({"carrier": "lohc", "direction": "dehydrogenation", "h2_mass_kg": m})
    assert r["thermal_energy_MJ"].shape == (4,)
    # Energy must scale with mass (linear)
    assert np.all(np.diff(r["thermal_energy_MJ"]) > 0), "Q must increase with H2 mass"


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    """1000 predictions should complete in < 1s."""
    m_arr = np.random.uniform(0.1, 10.0, 1000)
    T_arr = np.random.uniform(520.0, 650.0, 1000)
    start = time.perf_counter()
    model.predict({"carrier": "lohc", "direction": "dehydrogenation",
                   "h2_mass_kg": m_arr, "temperature_K": T_arr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
