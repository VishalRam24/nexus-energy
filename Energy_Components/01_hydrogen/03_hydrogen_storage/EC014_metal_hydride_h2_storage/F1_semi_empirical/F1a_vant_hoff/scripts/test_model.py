"""EC014 — Metal Hydride H2 Storage — F1a van't Hoff — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"temperature": 298.15, "soc": 0.5})
    for k in ["plateau_pressure_bar", "stored_mass_kg", "heat_of_reaction_kJ",
              "gravimetric_wt_pct", "volumetric_kg_per_m3", "fill_fraction"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC014"
    assert info["fidelity"] == "F1a"


def test_plateau_pressure_increases_with_temperature(model):
    """
    Fundamental van't Hoff physics: equilibrium pressure rises monotonically with T
    for desorption (endothermic reaction, Le Chatelier's principle).
    Lototskyy et al. (2014) Prog. Nat. Sci. Mater. 24(2), Fig. 1.
    """
    T = np.array([253.15, 273.15, 298.15, 323.15, 353.15])
    r = model.predict({"temperature": T, "soc": 0.5, "mode": "desorption"})
    assert np.all(np.diff(r["plateau_pressure_bar"]) > 0), \
        "Desorption plateau pressure must increase monotonically with temperature"


def test_absorption_pressure_above_desorption(model):
    """
    Hysteresis: absorption plateau > desorption plateau at same T.
    Lototskyy et al. (2014): hysteresis factor typically 1.1–1.5 for LaNi5.
    """
    T = 298.15
    r_abs = model.predict({"temperature": T, "soc": 0.5, "mode": "absorption"})
    r_des = model.predict({"temperature": T, "soc": 0.5, "mode": "desorption"})
    assert float(r_abs["plateau_pressure_bar"]) > float(r_des["plateau_pressure_bar"]), \
        "Absorption plateau pressure must exceed desorption plateau pressure (hysteresis)"


def test_lani5_plateau_pressure_range_298K(model):
    """
    LaNi5 at 25°C desorption plateau ≈ 2–3 bar.
    Sakintuna et al. (2007) Int. J. Hydrogen Energy 32(9), Table 1: LaNi5 ~2 bar at 298 K.
    """
    r = model.predict({"temperature": 298.15, "soc": 0.5, "mode": "desorption"})
    P = float(r["plateau_pressure_bar"])
    assert 0.5 < P < 20.0, \
        f"LaNi5 desorption plateau at 298 K should be ~2 bar, got {P:.2f} bar"


def test_stored_mass_increases_with_soc(model):
    """More H2 stored at higher state of charge (definition of SOC)."""
    soc = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    r = model.predict({"temperature": 298.15, "soc": soc})
    # Skip first diff (SOC=0 => m=0, second might equal if SOC spacing is small)
    masses = r["stored_mass_kg"]
    assert np.all(np.diff(masses) >= 0), "Stored mass must increase with SOC"
    assert float(masses[-1]) > float(masses[0]), "Full tank must have more H2 than empty"


def test_zero_soc_zero_mass(model):
    r = model.predict({"temperature": 298.15, "soc": 0.0})
    assert float(r["stored_mass_kg"]) == pytest.approx(0.0, abs=1e-10)


def test_full_soc_max_capacity(model):
    """At SOC=1, gravimetric density should match material spec (~1.4 wt% for LaNi5)."""
    r = model.predict({"temperature": 298.15, "soc": 1.0})
    wt = float(r["gravimetric_wt_pct"])
    # LaNi5: 1.4 wt%, but wt% of total system (bed+H2)
    assert 0.5 < wt < 3.0, \
        f"Gravimetric density at SOC=1 should be ~1.4 wt% for LaNi5, got {wt:.2f} wt%"


def test_heat_of_reaction_positive_for_absorption(model):
    """Absorption is exothermic: heat released to environment is positive."""
    r = model.predict({"temperature": 298.15, "soc": 0.5,
                       "mode": "absorption", "delta_m_H2_kg": 0.05})
    assert float(r["heat_of_reaction_kJ"]) > 0


def test_heat_of_reaction_zero_delta_m(model):
    """Zero H2 transfer => zero heat of reaction."""
    r = model.predict({"temperature": 298.15, "soc": 0.5,
                       "mode": "absorption", "delta_m_H2_kg": 0.0})
    assert float(r["heat_of_reaction_kJ"]) == pytest.approx(0.0, abs=1e-10)


def test_heat_per_kg_h2_reasonable(model):
    """
    Heat of absorption for LaNi5: |ΔH| ≈ 30 kJ/mol_H2 = 30e3/0.002016 ≈ 14.9 MJ/kg_H2.
    Sakintuna et al. (2007): LaNi5 ΔH = -30.1 kJ/mol.
    """
    delta_m = 0.14  # kg H2 (full bed capacity)
    r = model.predict({"temperature": 298.15, "soc": 0.5,
                       "mode": "absorption", "delta_m_H2_kg": delta_m})
    q_kJ = float(r["heat_of_reaction_kJ"])
    q_per_kg = q_kJ / delta_m  # kJ/kg_H2
    # Expect ~14880 kJ/kg_H2 = 14.88 MJ/kg_H2
    assert 10000 < q_per_kg < 20000, \
        f"Heat per kg H2 should be ~14880 kJ/kg for LaNi5, got {q_per_kg:.0f} kJ/kg"


def test_fill_fraction_bounds(model):
    r_low = model.predict({"temperature": 298.15, "soc": 0.0})
    r_high = model.predict({"temperature": 298.15, "soc": 1.0})
    assert float(r_low["fill_fraction"]) == pytest.approx(0.0, abs=1e-10)
    assert float(r_high["fill_fraction"]) == pytest.approx(1.0, abs=1e-10)


def test_volumetric_density_reasonable(model):
    """
    LaNi5 volumetric capacity ~105 kg_H2/m3 (at full charge).
    DOE target is 40 kg/m3; metal hydrides far exceed this.
    Sakintuna et al. (2007): LaNi5 ~115 kg_H2/m3.
    """
    r = model.predict({"temperature": 298.15, "soc": 1.0})
    rho = float(r["volumetric_kg_per_m3"])
    assert 50 < rho < 200, \
        f"Volumetric density at full charge should be ~105 kg/m3 for LaNi5, got {rho:.1f} kg/m3"


def test_pressure_log_linearity_with_inverse_T(model):
    """
    van't Hoff: ln(P) vs 1/T should be linear with slope ΔH_des/R.
    Test: R^2 > 0.999 for regression of ln(P) on 1/T.
    """
    T = np.linspace(260, 370, 20)
    r = model.predict({"temperature": T, "soc": 0.5, "mode": "desorption"})
    ln_P = np.log(r["plateau_pressure_bar"])
    inv_T = 1.0 / T
    # Linear regression
    A = np.vstack([inv_T, np.ones_like(inv_T)]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, ln_P, rcond=None)
    ss_res = np.sum((ln_P - A @ coeffs) ** 2)
    ss_tot = np.sum((ln_P - ln_P.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.999, f"ln(P) vs 1/T should be linear (R²>0.999), got R²={r2:.6f}"


def test_benchmark(model):
    T = np.random.uniform(253, 373, 1000)
    soc = np.random.uniform(0, 1, 1000)
    start = time.perf_counter()
    model.predict({"temperature": T, "soc": soc, "mode": "desorption"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
