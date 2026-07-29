"""EC012 — Compressed Gas H2 Storage — F1b Real-Gas — Test Suite"""
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


# --- Interface ---

def test_predict_storage_keys(model):
    r = model.predict({"P_bar": 350.0, "T_K": 300.0})
    for k in ["stored_mass_kg", "energy_stored_MJ", "fill_fraction", "Z",
              "gravimetric_density_wt_pct", "volumetric_density_kg_per_m3"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC012"
    assert info["fidelity"] == "F1b"


# --- Real-Gas Compressibility: Z > 1 for H2 at high pressure ---

def test_Z_equals_1_at_low_pressure(model):
    """At very low pressure, Z → 1 for any real gas."""
    r = model.predict({"P_bar": 1.0, "T_K": 300.0})
    Z = float(r["Z"])
    assert abs(Z - 1.0) < 0.02, f"Z at 1 bar should be ~1, got {Z:.4f}"


def test_Z_greater_than_1_at_high_pressure(model):
    """H2 is a quantum gas: Z > 1 at all T > 150 K for P > 10 bar."""
    for P in [100.0, 200.0, 350.0, 700.0]:
        r = model.predict({"P_bar": P, "T_K": 300.0})
        Z = float(r["Z"])
        assert Z > 1.0, f"Z should be > 1 at P={P} bar for H2, got Z={Z:.4f}"


def test_Z_increases_with_pressure(model):
    """Z monotonically increases with pressure for H2 at 300 K (quantum gas behaviour)."""
    pressures = np.array([10.0, 100.0, 200.0, 350.0, 700.0])
    Zs = [float(model.predict({"P_bar": P, "T_K": 300.0})["Z"]) for P in pressures]
    assert all(Zs[i] < Zs[i + 1] for i in range(len(Zs) - 1)), \
        f"Z not monotonically increasing: {Zs}"


def test_Z_700bar_magnitude(model):
    """At 700 bar and 300 K, Z ≈ 1.3–1.5 for H2 (Leachman 2009 / NIST data)."""
    r = model.predict({"P_bar": 700.0, "T_K": 300.0})
    Z = float(r["Z"])
    assert 1.2 <= Z <= 1.6, f"Z at 700 bar / 300 K should be 1.2–1.6, got {Z:.4f}"


def test_Z_temperature_dependence(model):
    """Lower T → higher Z for H2 at high pressure (virial A2 negative → increases B at low T)."""
    # At 700 bar: colder tank → Z higher (more deviation from ideal)
    Z_cold = float(model.predict({"P_bar": 700.0, "T_K": 240.0})["Z"])
    Z_warm = float(model.predict({"P_bar": 700.0, "T_K": 360.0})["Z"])
    assert Z_cold > Z_warm, (
        f"At high P, colder H2 should have higher Z (quantum effect): "
        f"Z(240K)={Z_cold:.4f}, Z(360K)={Z_warm:.4f}"
    )


# --- Real-gas correction: less mass than ideal gas predicts ---

def test_real_gas_stores_less_than_ideal(model):
    """
    Real-gas model should store LESS mass than ideal gas (Z > 1 means lower density).
    Compare against ideal-gas formula m_ideal = P*V/(R*T).
    """
    R_H2 = 8.314 / 0.002016   # J/(kg·K)
    P_bar = 700.0
    T_K = 300.0
    P_Pa = P_bar * 1e5
    V = model._model.V_tank

    m_ideal = P_Pa * V / (R_H2 * T_K)
    r = model.predict({"P_bar": P_bar, "T_K": T_K})
    m_real = float(r["stored_mass_kg"])

    assert m_real < m_ideal, (
        f"Real-gas mass ({m_real:.3f} kg) should be < ideal-gas ({m_ideal:.3f} kg) at high P"
    )


def test_stored_mass_positive(model):
    """Stored mass must be positive for all valid pressures."""
    pressures = np.array([20.0, 100.0, 350.0, 700.0])
    for P in pressures:
        r = model.predict({"P_bar": P, "T_K": 300.0})
        assert float(r["stored_mass_kg"]) > 0, f"Mass should be positive at P={P} bar"


def test_stored_mass_increases_with_pressure(model):
    """More pressure → more stored mass."""
    pressures = np.array([20.0, 100.0, 200.0, 350.0, 700.0])
    masses = [float(model.predict({"P_bar": P, "T_K": 300.0})["stored_mass_kg"])
              for P in pressures]
    assert all(masses[i] < masses[i + 1] for i in range(len(masses) - 1)), \
        f"Mass not increasing with pressure: {masses}"


def test_stored_mass_decreases_with_temperature(model):
    """Hotter gas → lower density → less stored mass at fixed pressure."""
    temps = [240.0, 280.0, 300.0, 320.0, 360.0]
    masses = [float(model.predict({"P_bar": 350.0, "T_K": T})["stored_mass_kg"])
              for T in temps]
    assert all(masses[i] > masses[i + 1] for i in range(len(masses) - 1)), \
        f"Mass should decrease with temperature: {masses}"


# --- T_amb coupling ---

def test_usable_mass_increases_with_cold_ambient(model):
    """Colder ambient → more H2 stored (higher density)."""
    T_cold = 253.15   # -20 C
    T_warm = 313.15   # +40 C
    m_cold = float(model._model.usable_mass_vs_Tamb(T_cold))
    m_warm = float(model._model.usable_mass_vs_Tamb(T_warm))
    assert m_cold > m_warm, (
        f"Colder T_amb should give more usable mass: "
        f"m(−20°C)={m_cold:.3f} > m(+40°C)={m_warm:.3f}"
    )


# --- Heat of compression ---

def test_temperature_rises_during_fill(model):
    """Filling the tank should raise temperature above T_amb."""
    r = model.predict({"mode": "fill", "P1_bar": 20.0, "P2_bar": 700.0, "T_amb_K": 298.15})
    dT = float(r["dT_K"])
    assert dT > 0, f"Temperature should rise during fill, got dT={dT:.2f} K"


def test_temperature_rise_larger_fill_larger_dT(model):
    """Filling from lower initial pressure (more mass added) → larger dT."""
    r_small = model.predict({"mode": "fill", "P1_bar": 500.0, "P2_bar": 700.0, "T_amb_K": 298.15})
    r_large = model.predict({"mode": "fill", "P1_bar": 20.0, "P2_bar": 700.0, "T_amb_K": 298.15})
    assert float(r_large["dT_K"]) > float(r_small["dT_K"]), \
        "Larger fill should cause larger temperature rise"


def test_tau_positive(model):
    """Thermal equilibration time constant must be positive."""
    r = model.predict({"mode": "fill", "P1_bar": 20.0, "P2_bar": 700.0})
    tau = float(r["tau_s"])
    assert tau > 0, f"tau should be positive, got {tau}"


def test_cooling_returns_to_ambient(model):
    """After long enough time, tank temperature returns to T_amb."""
    m = model._model
    T_post = 360.0
    T_amb = 298.15
    tau = m.thermal_equilibration_time()
    T_final = float(m.tank_temperature_cooling(T_post, T_amb, 10 * tau))
    assert abs(T_final - T_amb) < 0.1, \
        f"After 10x tau, tank should be near T_amb: T={T_final:.2f} K"


# --- Compression work ---

def test_compression_work_positive(model):
    r = model.predict({"mode": "compression", "P1_bar": 30.0, "P2_bar": 700.0})
    assert float(r["compression_work_kJ_per_kg"]) > 0


def test_compression_work_real_gas_higher_than_ideal(model):
    """
    Real-gas Z>1 correction increases compression work vs ideal-gas formula.
    w_real = Z*w_ideal (approximately), so w_real > w_ideal at high pressure.
    """
    # Compute ideal-gas reference manually (Z1=1)
    m = model._model
    k = m.gamma
    R = m.R_H2
    T1 = m.T_inlet_default
    P1, P2 = 30.0, 700.0
    ratio = (P2 / P1) ** ((k - 1.0) / k)
    w_ideal = (k / (k - 1.0)) * R * T1 * (ratio - 1.0) / m.eta_s / 1000.0

    r = model.predict({"mode": "compression", "P1_bar": P1, "P2_bar": P2})
    w_real = float(r["compression_work_kJ_per_kg"])

    assert w_real > w_ideal, (
        f"Real-gas compression work ({w_real:.1f}) should exceed ideal ({w_ideal:.1f}) kJ/kg"
    )


def test_compression_work_increases_with_pressure_ratio(model):
    """Higher final pressure → more compression work."""
    P_targets = [100.0, 200.0, 350.0, 700.0]
    works = [float(model.predict({"mode": "compression", "P1_bar": 30.0, "P2_bar": P})
                   ["compression_work_kJ_per_kg"]) for P in P_targets]
    assert all(works[i] < works[i + 1] for i in range(len(works) - 1)), \
        f"Work not increasing with target pressure: {works}"


# --- Fill fraction and energy ---

def test_fill_fraction_at_Pmax(model):
    """Fill fraction at P_max should be 1.0."""
    m = model._model
    r = model.predict({"P_bar": m.P_max, "T_K": 300.0})
    ff = float(r["fill_fraction"])
    assert abs(ff - 1.0) < 1e-9, f"Fill fraction at P_max should be 1.0, got {ff}"


def test_fill_fraction_below_Pmax(model):
    """Fill fraction below P_max should be < 1."""
    r = model.predict({"P_bar": 350.0, "T_K": 300.0})
    assert float(r["fill_fraction"]) < 1.0


def test_energy_stored_positive(model):
    """Stored energy must be positive."""
    r = model.predict({"P_bar": 350.0, "T_K": 300.0})
    assert float(r["energy_stored_MJ"]) > 0


# --- Vectorized interface ---

def test_vectorized_storage(model):
    """Model accepts numpy array inputs."""
    P = np.array([20.0, 100.0, 350.0, 700.0])
    T = np.full(4, 300.0)
    r = model.predict({"P_bar": P, "T_K": T})
    assert len(r["stored_mass_kg"]) == 4
    assert len(r["Z"]) == 4


# --- Benchmark ---

def test_benchmark_1000(model):
    """1000 storage predictions in < 200 ms."""
    P = np.random.uniform(20, 700, 1000)
    T = np.random.uniform(240, 360, 1000)
    start = time.perf_counter()
    model.predict({"P_bar": P, "T_K": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 0.2
