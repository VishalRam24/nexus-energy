"""EC015 — Chemical H2 Storage (LOHC/Ammonia) — F1a Conversion Energy — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys_lohc(model):
    r = model.predict({"h2_mass_kg": 1.0, "mode": "lohc", "direction": "dehydrogenation"})
    for k in ["carrier_mass_kg", "thermal_energy_MJ", "specific_energy_MJ_per_kg_H2",
              "reactor_temperature_K", "roundtrip_efficiency", "gravimetric_capacity_wt_pct"]:
        assert k in r


def test_predict_keys_nh3(model):
    r = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia", "direction": "cracking"})
    for k in ["carrier_mass_kg", "thermal_energy_MJ", "specific_energy_MJ_per_kg_H2",
              "reactor_temperature_K", "roundtrip_efficiency", "gravimetric_capacity_wt_pct"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC015"
    assert info["fidelity"] == "F1a"


def test_dehydrogenation_endothermic_lohc(model):
    """
    Dehydrogenation must be endothermic (positive Q): reaction is driven by heat input.
    Preuster et al. (2017) Acc. Chem. Res.: ΔH_dehydro = +65 kJ/mol H2 > 0.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "lohc", "direction": "dehydrogenation"})
    assert float(r["thermal_energy_MJ"]) > 0, \
        "Dehydrogenation must require heat input (endothermic)"


def test_hydrogenation_exothermic_lohc(model):
    """
    Hydrogenation must be exothermic (negative Q): heat released.
    Preuster et al. (2017): ΔH_hydro = -65 kJ/mol H2 < 0.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "lohc", "direction": "hydrogenation"})
    assert float(r["thermal_energy_MJ"]) < 0, \
        "Hydrogenation must release heat (exothermic)"


def test_cracking_endothermic_nh3(model):
    """
    NH3 cracking is endothermic. Lamb et al. (2019): ΔH_crack = +46 kJ/mol H2 > 0.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia", "direction": "cracking"})
    assert float(r["thermal_energy_MJ"]) > 0, \
        "NH3 cracking must require heat input (endothermic)"


def test_synthesis_exothermic_nh3(model):
    """
    NH3 synthesis is exothermic (Haber-Bosch). Lamb et al. (2019): ΔH_synth = -46 kJ/mol H2.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia", "direction": "synthesis"})
    assert float(r["thermal_energy_MJ"]) < 0, \
        "NH3 synthesis must release heat (exothermic)"


def test_thermal_energy_scales_linearly_with_h2_mass(model):
    """Energy demand is proportional to H2 mass (semi-empirical linear scaling)."""
    m1, m2 = 1.0, 10.0
    r1 = model.predict({"h2_mass_kg": m1, "mode": "lohc", "direction": "dehydrogenation"})
    r2 = model.predict({"h2_mass_kg": m2, "mode": "lohc", "direction": "dehydrogenation"})
    ratio = float(r2["thermal_energy_MJ"]) / float(r1["thermal_energy_MJ"])
    assert abs(ratio - (m2 / m1)) < 1e-6, f"Energy should scale 10x with 10x H2 mass, got {ratio:.4f}"


def test_carrier_mass_scales_linearly(model):
    """Carrier mass is proportional to H2 mass."""
    m1, m2 = 1.0, 5.0
    r1 = model.predict({"h2_mass_kg": m1, "mode": "lohc"})
    r2 = model.predict({"h2_mass_kg": m2, "mode": "lohc"})
    ratio = float(r2["carrier_mass_kg"]) / float(r1["carrier_mass_kg"])
    assert abs(ratio - (m2 / m1)) < 1e-6


def test_lohc_gravimetric_capacity_range(model):
    """
    LOHC (DBT) gravimetric H2 capacity should be ~6.2 wt%.
    Niermann et al. (2021) Energy Environ. Sci.: DBT/H18-DBT = 6.2 wt%.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "lohc"})
    cap = float(r["gravimetric_capacity_wt_pct"])
    assert 5.0 < cap < 8.0, \
        f"LOHC gravimetric capacity should be ~6.2 wt%, got {cap:.2f} wt%"


def test_nh3_gravimetric_capacity_range(model):
    """
    NH3 gravimetric H2 capacity ~17.6 wt%.
    Lamb et al. (2019): NH3 contains 17.6 wt% H2 by mass.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia"})
    cap = float(r["gravimetric_capacity_wt_pct"])
    assert 15.0 < cap < 20.0, \
        f"NH3 gravimetric capacity should be ~17.6 wt%, got {cap:.2f} wt%"


def test_nh3_higher_gravimetric_than_lohc(model):
    """
    NH3 has higher gravimetric H2 density than LOHC.
    NH3: 17.6 wt% vs LOHC (DBT): 6.2 wt%. Niermann et al. 2021.
    """
    r_lohc = model.predict({"h2_mass_kg": 1.0, "mode": "lohc"})
    r_nh3 = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia"})
    assert float(r_nh3["gravimetric_capacity_wt_pct"]) > \
           float(r_lohc["gravimetric_capacity_wt_pct"]), \
        "NH3 must have higher gravimetric capacity than LOHC"


def test_lohc_specific_dehydro_energy_range(model):
    """
    LOHC dehydrogenation specific energy: ~65 kJ/mol / 0.002016 kg/mol / eta ≈ 38 MJ/kg_H2.
    Preuster et al. (2017): 65 kJ/mol H2, eta~0.85 => ~38.2 MJ/kg_H2.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "lohc", "direction": "dehydrogenation"})
    spec = float(r["specific_energy_MJ_per_kg_H2"])
    # 65000 / 0.002016 / 0.85 / 1e6 = ~37.9 MJ/kg_H2
    assert 30.0 < spec < 50.0, \
        f"LOHC dehydrogenation specific energy should be ~38 MJ/kg_H2, got {spec:.2f}"


def test_nh3_cracking_specific_energy_range(model):
    """
    NH3 cracking specific energy: ~46 kJ/mol / 0.002016 kg/mol / 0.9 ≈ 25.4 MJ/kg_H2.
    Lamb et al. (2019): 46 kJ/mol H2.
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia", "direction": "cracking"})
    spec = float(r["specific_energy_MJ_per_kg_H2"])
    # 46000 / 0.002016 / 0.9 / 1e6 = ~25.3 MJ/kg_H2
    assert 18.0 < spec < 35.0, \
        f"NH3 cracking specific energy should be ~25 MJ/kg_H2, got {spec:.2f}"


def test_lohc_reactor_temperature_reasonable(model):
    """
    LOHC dehydrogenation temperature ~300°C (573 K). Preuster et al. (2017).
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "lohc", "direction": "dehydrogenation"})
    T = float(r["reactor_temperature_K"])
    assert 500 < T < 700, f"LOHC dehydrogenation T should be ~573 K, got {T:.0f} K"


def test_nh3_reactor_temperature_reasonable(model):
    """
    NH3 cracking temperature ~500°C (773 K). Lamb et al. (2019).
    """
    r = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia", "direction": "cracking"})
    T = float(r["reactor_temperature_K"])
    assert 600 < T < 900, f"NH3 cracking T should be ~773 K, got {T:.0f} K"


def test_roundtrip_efficiency_below_unity(model):
    """Round-trip efficiency must be < 1 (thermodynamic losses)."""
    for mode in ["lohc", "ammonia"]:
        r = model.predict({"h2_mass_kg": 1.0, "mode": mode})
        eta = float(r["roundtrip_efficiency"])
        assert 0 < eta < 1.0, f"{mode} round-trip efficiency must be 0 < η < 1, got {eta:.4f}"


def test_vectorized_h2_mass(model):
    """Model must handle array inputs correctly."""
    m = np.array([0.5, 1.0, 5.0, 10.0, 100.0])
    r = model.predict({"h2_mass_kg": m, "mode": "lohc", "direction": "dehydrogenation"})
    assert r["carrier_mass_kg"].shape == m.shape
    assert r["thermal_energy_MJ"].shape == m.shape
    assert np.all(r["carrier_mass_kg"] > 0)
    assert np.all(r["thermal_energy_MJ"] > 0)


def test_benchmark(model):
    m = np.random.uniform(0.1, 1000, 1000)
    start = time.perf_counter()
    model.predict({"h2_mass_kg": m, "mode": "lohc", "direction": "dehydrogenation"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
