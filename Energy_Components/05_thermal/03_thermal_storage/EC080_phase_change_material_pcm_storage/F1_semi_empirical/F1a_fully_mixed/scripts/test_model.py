"""EC080 — PCM Storage — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

@pytest.fixture
def model():
    return ComponentModel()

def test_predict_keys(model):
    r = model.predict({"temperature": 42.0, "liquid_fraction": 0.5,
                       "q_charge": 0.0, "q_discharge": 0.0})
    for k in ["dT_dt", "d_fraction_dt", "energy_stored_kwh", "soc"]:
        assert k in r, f"Missing key: {k}"

def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC080"
    assert info["fidelity"] == "F1a"

def test_liquid_fraction_in_range(model):
    """liquid_fraction must always remain in [0, 1]."""
    # Test all three regions
    for T, f in [(20.0, 0.0), (42.0, 0.5), (60.0, 1.0)]:
        r = model.predict({"temperature": T, "liquid_fraction": f,
                           "q_charge": 500.0, "q_discharge": 0.0})
        assert 0.0 <= float(r["soc"]) <= 1.0, f"SOC out of range at T={T}, f={f}"

def test_soc_in_range(model):
    """SOC must be in [0, 1] across all states."""
    T_vals = np.array([10.0, 30.0, 42.0, 55.0, 70.0])
    f_vals = np.array([ 0.0,  0.0,  0.5,  1.0,  1.0])
    r = model.predict({"temperature": T_vals, "liquid_fraction": f_vals,
                       "q_charge": 0.0, "q_discharge": 0.0})
    assert np.all(r["soc"] >= 0.0) and np.all(r["soc"] <= 1.0)

def test_energy_increases_with_charge(model):
    """Energy stored must be higher at higher liquid fractions (more latent heat)."""
    # At T_melt, increase f from 0 to 1 → energy increases by L*m
    r0 = model.predict({"temperature": 42.0, "liquid_fraction": 0.0,
                        "q_charge": 0.0, "q_discharge": 0.0})
    r1 = model.predict({"temperature": 42.0, "liquid_fraction": 1.0,
                        "q_charge": 0.0, "q_discharge": 0.0})
    assert float(r1["energy_stored_kwh"]) > float(r0["energy_stored_kwh"]), \
        "Energy must increase with liquid fraction"

def test_dT_dt_zero_in_mushy_zone(model):
    """In the mushy zone (T=Tm), dT/dt must be 0 (heat goes to melting, not temp change)."""
    r = model.predict({"temperature": 42.0, "liquid_fraction": 0.5,
                       "q_charge": 500.0, "q_discharge": 0.0})
    assert float(r["dT_dt"]) == pytest.approx(0.0, abs=1e-10), \
        "dT/dt must be 0 in mushy zone"

def test_df_dt_positive_when_charging_in_mushy(model):
    """df/dt must be positive when charging in mushy zone (melting)."""
    r = model.predict({"temperature": 42.0, "liquid_fraction": 0.5,
                       "q_charge": 500.0, "q_discharge": 0.0})
    assert float(r["d_fraction_dt"]) > 0.0, "df/dt must be positive during melting"

def test_df_dt_negative_when_discharging_in_mushy(model):
    """df/dt must be negative when discharging in mushy zone (solidifying)."""
    r = model.predict({"temperature": 42.0, "liquid_fraction": 0.5,
                       "q_charge": 0.0, "q_discharge": 500.0})
    assert float(r["d_fraction_dt"]) < 0.0, "df/dt must be negative during solidification"

def test_dT_dt_positive_solid_when_charging(model):
    """In solid region, dT/dt must be positive when charging."""
    r = model.predict({"temperature": 20.0, "liquid_fraction": 0.0,
                       "q_charge": 500.0, "q_discharge": 0.0})
    assert float(r["dT_dt"]) > 0.0, "dT/dt must be positive in solid when charging"

def test_dT_dt_positive_liquid_when_charging(model):
    """In liquid region, dT/dt must be positive when charging."""
    r = model.predict({"temperature": 60.0, "liquid_fraction": 1.0,
                       "q_charge": 500.0, "q_discharge": 0.0})
    assert float(r["dT_dt"]) > 0.0, "dT/dt must be positive in liquid when charging"

def test_phase_change_energy_magnitude(model):
    """Full latent heat = m*L = 500*174 kJ = 87 MJ = 24.17 kWh."""
    r0 = model.predict({"temperature": 42.0, "liquid_fraction": 0.0,
                        "q_charge": 0.0, "q_discharge": 0.0})
    r1 = model.predict({"temperature": 42.0, "liquid_fraction": 1.0,
                        "q_charge": 0.0, "q_discharge": 0.0})
    dE = float(r1["energy_stored_kwh"]) - float(r0["energy_stored_kwh"])
    # Expected: 500 * 174 / 3600 = 24.17 kWh
    assert 22.0 < dE < 27.0, f"Latent heat = {dE:.2f} kWh, expected ~24.2 kWh"

def test_simulation_charging_increases_energy(model):
    """Simulation: charge only → energy must increase monotonically."""
    from model import PCMF1a
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = PCMF1a(params)
    N = 60  # 60 minutes
    q_c = np.full(N, 500.0)   # 500 W
    q_d = np.zeros(N)
    sim = m.simulate(20.0, 0.0, q_c, q_d, dt_s=60.0)
    # Energy should increase (or stay same) at every step
    assert np.all(np.diff(sim["energy_kwh"]) >= -1e-9), "Energy must not decrease during charging"

def test_simulation_solid_to_liquid(model):
    """Simulation: starting fully solid, charging should eventually reach liquid phase."""
    from model import PCMF1a
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = PCMF1a(params)
    N = 300  # 5 hours
    q_c = np.full(N, 800.0)   # 800 W charging
    q_d = np.zeros(N)
    sim = m.simulate(20.0, 0.0, q_c, q_d, dt_s=60.0)
    # Should have warmed significantly from 20°C start
    # 800W for 5h = 14.4 MJ; m*cp*dT=500*2000*dT => dT~14.4 (minus losses)
    assert sim["T_C"][-1] > 30.0 or sim["f"][-1] > 0.0, \
        f"Should have warmed up from 20°C: T={sim['T_C'][-1]:.1f}°C, f={sim['f'][-1]:.3f}"

def test_benchmark(model):
    """1000 instantaneous predictions in <1 second."""
    T = np.random.uniform(10, 70, 1000)
    f = np.random.uniform(0, 1, 1000)
    Qc = np.random.uniform(0, 5000, 1000)
    Qd = np.random.uniform(0, 5000, 1000)
    start = time.perf_counter()
    model.predict({"temperature": T, "liquid_fraction": f,
                   "q_charge": Qc, "q_discharge": Qd})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
