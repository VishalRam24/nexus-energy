"""EC080 -- PCM Storage -- F1b Enthalpy Method -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "T_htf_in_degC": 80.0, "flow_rate_kg_s": 1.0,
        "mode": "charge", "duration_s": 600.0, "T_pcm_init": 40.0,
    })
    for k in ["T_pcm_degC", "phase_fraction", "energy_stored_kwh",
              "thermal_power_kw", "T_outlet_degC"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC080"
    assert info["fidelity"] == "F1b"


def test_phase_fraction_zero_below_solidus(model):
    """Phase fraction must be 0 below solidus (T_pc - dT = 56 degC)."""
    r = model.predict({
        "T_htf_in_degC": 40.0, "flow_rate_kg_s": 0.0,
        "mode": "idle", "duration_s": 1.0, "T_pcm_init": 40.0,
    })
    assert r["phase_fraction"] == pytest.approx(0.0, abs=1e-6)


def test_phase_fraction_one_above_liquidus(model):
    """Phase fraction must be 1 above liquidus (T_pc + dT = 60 degC)."""
    r = model.predict({
        "T_htf_in_degC": 80.0, "flow_rate_kg_s": 0.0,
        "mode": "idle", "duration_s": 1.0, "T_pcm_init": 70.0,
    })
    assert r["phase_fraction"] == pytest.approx(1.0, abs=1e-6)


def test_phase_fraction_half_at_tpc(model):
    """Phase fraction at T_pc = 58 degC should be 0.5."""
    r = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0,
        "duration_s": 1.0, "T_pcm_init": 58.0,
    })
    assert r["phase_fraction"] == pytest.approx(0.5, abs=0.05)


def test_charging_increases_temperature(model):
    """Charging with hot HTF must increase PCM temperature."""
    r = model.predict({
        "T_htf_in_degC": 80.0, "flow_rate_kg_s": 1.0,
        "mode": "charge", "duration_s": 3600.0, "T_pcm_init": 40.0,
    })
    assert r["T_pcm_degC"] > 40.0, "PCM must heat up during charging"


def test_discharging_decreases_temperature(model):
    """Discharging with cold HTF must decrease PCM temperature."""
    r = model.predict({
        "T_htf_in_degC": 30.0, "flow_rate_kg_s": 1.0,
        "mode": "discharge", "duration_s": 3600.0, "T_pcm_init": 70.0,
    })
    assert r["T_pcm_degC"] < 70.0, "PCM must cool during discharging"


def test_energy_stored_nonnegative(model):
    """Stored energy must always be non-negative."""
    r = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0,
        "duration_s": 1.0, "T_pcm_init": 30.0,
    })
    assert r["energy_stored_kwh"] >= 0.0


def test_energy_increases_with_charge(model):
    """Energy stored must increase after charging."""
    r_before = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0,
        "duration_s": 1.0, "T_pcm_init": 40.0,
    })
    r_after = model.predict({
        "T_htf_in_degC": 80.0, "flow_rate_kg_s": 1.0,
        "mode": "charge", "duration_s": 3600.0, "T_pcm_init": 40.0,
    })
    assert r_after["energy_stored_kwh"] > r_before["energy_stored_kwh"]


def test_enthalpy_monotonic(model):
    """Enthalpy must be monotonically increasing with temperature."""
    m = model._model
    T_range = np.linspace(20, 90, 500)
    h = m.enthalpy(T_range)
    assert np.all(np.diff(h) >= 0), "Enthalpy must be monotonically increasing"


def test_enthalpy_inversion_roundtrip(model):
    """Converting T -> h -> T must recover the original temperature."""
    m = model._model
    T_test = np.array([30.0, 45.0, 57.0, 58.0, 59.0, 65.0, 80.0])
    h = m.enthalpy(T_test)
    T_recovered = m.temperature_from_enthalpy(h)
    assert np.allclose(T_test, T_recovered, atol=0.1)


def test_latent_heat_magnitude(model):
    """Latent heat = m * L = 5000 * 180 kJ = 900 MJ = 250 kWh."""
    m = model._model
    h_solidus = float(m.enthalpy(np.array([m.T_solidus]))[0])
    h_liquidus = float(m.enthalpy(np.array([m.T_liquidus]))[0])
    dh = h_liquidus - h_solidus
    E_latent_kwh = m.mass * dh / 3.6e6
    assert 240 < E_latent_kwh < 260, f"Latent heat = {E_latent_kwh:.1f} kWh, expected ~250"


def test_thermal_power_positive_during_charge(model):
    """Thermal power should be positive when charging."""
    r = model.predict({
        "T_htf_in_degC": 80.0, "flow_rate_kg_s": 1.0,
        "mode": "charge", "duration_s": 600.0, "T_pcm_init": 40.0,
    })
    assert r["thermal_power_kw"] > 0.0


def test_outlet_temp_between_inlet_and_pcm(model):
    """HTF outlet temp should be between inlet and PCM temp (charge)."""
    r = model.predict({
        "T_htf_in_degC": 80.0, "flow_rate_kg_s": 1.0,
        "mode": "charge", "duration_s": 600.0, "T_pcm_init": 40.0,
    })
    assert r["T_outlet_degC"] < 80.0, "Outlet must be cooler than inlet during charge"
    assert r["T_outlet_degC"] > 40.0, "Outlet must be warmer than initial PCM"


def test_benchmark(model):
    """10 predictions must complete in < 5 seconds."""
    start = time.perf_counter()
    for _ in range(10):
        model.predict({
            "T_htf_in_degC": 80.0, "flow_rate_kg_s": 1.0,
            "mode": "charge", "duration_s": 3600.0, "T_pcm_init": 40.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 10 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
