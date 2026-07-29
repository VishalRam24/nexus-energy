"""EC202 — DAC Liquid Solvent — F1b Part-Load + Solvent Degradation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"plr": 1.0, "operating_hours": 0})
    for k in ["capture_rate_kgh", "calciner_duty_gj_ton", "electrical_kwh_ton",
              "liquefaction_energy_gj_ton", "total_specific_energy_gj_ton",
              "solvent_capacity_pct"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC202"
    assert info["fidelity"] == "F1b"


def test_calciner_duty_at_design(model):
    """At PLR=1, fresh solvent, T=900C: calciner duty ~8 GJ/tCO2 (Keith 2018: 7-9 GJ/tCO2)."""
    r = model.predict({"plr": 1.0, "operating_hours": 0, "T_calciner_degC": 900.0})
    q = float(np.atleast_1d(r["calciner_duty_gj_ton"])[0])
    # Literature: Keith 2018 reports 7.0-9.0 GJ/tCO2 for calciner
    assert 5.0 < q < 12.0, f"Calciner duty at design = {q:.2f} GJ/tCO2"


def test_calciner_duty_increases_at_lower_temperature(model):
    """Lower calciner temperature should increase specific calciner duty (incomplete regeneration)."""
    r_low  = model.predict({"plr": 1.0, "operating_hours": 0, "T_calciner_degC": 800.0})
    r_high = model.predict({"plr": 1.0, "operating_hours": 0, "T_calciner_degC": 900.0})
    q_low  = float(np.atleast_1d(r_low["calciner_duty_gj_ton"])[0])
    q_high = float(np.atleast_1d(r_high["calciner_duty_gj_ton"])[0])
    assert q_low > q_high, f"Lower T gave lower duty: {q_low:.2f} vs {q_high:.2f}"


def test_calciner_duty_increases_at_part_load(model):
    """Reboiler/calciner specific duty should increase at part-load (fixed infrastructure)."""
    PLRs = [0.3, 0.5, 0.7, 1.0]
    qs = []
    for plr in PLRs:
        r = model.predict({"plr": plr, "operating_hours": 0})
        qs.append(float(np.atleast_1d(r["calciner_duty_gj_ton"])[0]))
    assert qs[0] > qs[-1], f"Calciner duty not higher at part-load: {qs}"


def test_capacity_factor_at_zero_hours(model):
    """At t=0: solvent capacity = 100%."""
    r = model.predict({"plr": 1.0, "operating_hours": 0})
    cap = float(np.atleast_1d(r["solvent_capacity_pct"])[0])
    assert abs(cap - 100.0) < 1e-3, f"Capacity at t=0 = {cap:.2f}%"


def test_capacity_declines_with_hours(model):
    """KOH capacity should decrease with operating time."""
    hours_list = [0, 8760, 43800, 87600]
    caps = []
    for h in hours_list:
        r = model.predict({"plr": 1.0, "operating_hours": h})
        caps.append(float(np.atleast_1d(r["solvent_capacity_pct"])[0]))
    assert all(caps[i] >= caps[i + 1] for i in range(len(caps) - 1)), \
        f"Capacity not declining: {caps}"


def test_capacity_5pct_loss_per_year(model):
    """After 1 year (8760h): capacity = exp(-0.05)*100 ~ 95.1%."""
    r = model.predict({"plr": 1.0, "operating_hours": 8760})
    cap = float(np.atleast_1d(r["solvent_capacity_pct"])[0])
    expected = np.exp(-0.05) * 100.0
    assert abs(cap - expected) < 1.0, f"1-yr capacity = {cap:.2f}%, expected ~{expected:.2f}%"


def test_liquefaction_energy_scales_with_pressure(model):
    """Liquefaction energy should increase with injection pressure."""
    r_low  = model.predict({"plr": 1.0, "operating_hours": 0, "injection_pressure_bar": 10.0})
    r_high = model.predict({"plr": 1.0, "operating_hours": 0, "injection_pressure_bar": 200.0})
    e_low  = float(np.atleast_1d(r_low["liquefaction_energy_gj_ton"])[0])
    e_high = float(np.atleast_1d(r_high["liquefaction_energy_gj_ton"])[0])
    assert e_high > e_low, f"E_liq not increasing with pressure: {e_low:.3f} vs {e_high:.3f}"


def test_total_energy_reasonable(model):
    """Total specific energy at design: ~8-12 GJ/tCO2 (Keith 2018: ~8.8 GJ/tCO2 total)."""
    r = model.predict({"plr": 1.0, "operating_hours": 0, "T_calciner_degC": 900.0,
                       "injection_pressure_bar": 100.0})
    e = float(np.atleast_1d(r["total_specific_energy_gj_ton"])[0])
    # Keith 2018 total: ~8.8 GJ/tCO2; allow generous range for parametric variation
    assert 6.0 < e < 18.0, f"Total energy = {e:.2f} GJ/tCO2"


def test_capture_rate_positive(model):
    """CO2 capture rate must be positive at design conditions."""
    r = model.predict({"air_flow_m3h": 3.6e6, "plr": 1.0, "operating_hours": 0})
    co2 = float(np.atleast_1d(r["capture_rate_kgh"])[0])
    assert co2 > 0, f"Capture rate = {co2:.2f} kg/h"


def test_capture_rate_scales_with_plr(model):
    """CO2 capture rate (sorbent-limited regime) should increase with PLR.
    At very low air flow (sorbent-limited), capture scales with n_contactors * PLR.
    At normal air flow, capture is air-flow-limited (sorbent capacity >> air-side CO2).
    Test the sorbent-limited regime using high air flow exceeding sorbent capacity.
    RATIONALE: Carbon Engineering DAC: at design, air-side is the bottleneck (capture efficiency
    of 75% of available CO2). Sorbent capacity far exceeds air-side CO2 flux. Testing PLR effect
    requires explicitly checking sorbent-limited conditions (very high air flow).
    """
    # At extremely high air flow, sorbent capacity becomes the bottleneck
    # k_cap=25 mol/(m2*h) * A=1000 m2 * n=40 * M_CO2/1000 = 44000 kg/h max sorbent at PLR=1
    # Set air flow so air-side > sorbent at PLR=1 but close enough to see PLR effect
    r_low  = model.predict({"air_flow_m3h": 1e9, "plr": 0.5, "operating_hours": 0})
    r_high = model.predict({"air_flow_m3h": 1e9, "plr": 1.0, "operating_hours": 0})
    co2_low  = float(np.atleast_1d(r_low["capture_rate_kgh"])[0])
    co2_high = float(np.atleast_1d(r_high["capture_rate_kgh"])[0])
    assert co2_high > co2_low, f"Capture not increasing with PLR (sorbent-limited): {co2_low:.1f} vs {co2_high:.1f}"


def test_capture_rate_declines_with_degradation(model):
    """CO2 capture rate should decline as KOH is poisoned (sorbent-limited regime).
    RATIONALE: Degradation reduces KOH capacity. In sorbent-limited regime,
    capture = min(air-side, sorbent) = sorbent, which declines with degradation.
    At normal air flow, air-side limits capture, so degradation only matters
    when sorbent capacity falls below air-side CO2 rate. For 10-yr period
    (exp(-0.05*10) = 0.6), sorbent falls to 0.6*44000 = 26400 kg/h, still >> 2074.
    We test at high air flow where degradation effect is visible.
    """
    r_new = model.predict({"air_flow_m3h": 1e9, "plr": 1.0, "operating_hours": 0})
    r_old = model.predict({"air_flow_m3h": 1e9, "plr": 1.0, "operating_hours": 87600})
    co2_new = float(np.atleast_1d(r_new["capture_rate_kgh"])[0])
    co2_old = float(np.atleast_1d(r_old["capture_rate_kgh"])[0])
    assert co2_old < co2_new, f"Capture not declining with degradation: {co2_new:.1f} vs {co2_old:.1f}"


def test_electrical_positive(model):
    r = model.predict({"plr": 0.7, "operating_hours": 0})
    e = float(np.atleast_1d(r["electrical_kwh_ton"])[0])
    assert e > 0


def test_array_input(model):
    """Model should handle array PLR inputs."""
    plrs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"plr": plrs, "operating_hours": 0})
    assert len(np.atleast_1d(r["calciner_duty_gj_ton"])) == 10


def test_benchmark(model):
    plrs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"plr": plrs, "operating_hours": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
