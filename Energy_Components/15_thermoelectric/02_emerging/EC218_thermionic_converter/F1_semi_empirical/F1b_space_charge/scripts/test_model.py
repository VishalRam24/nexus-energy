"""EC218 — Thermionic Converter — F1b Space-Charge — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

k_B = 1.380649e-23
q_e = 1.602176634e-19


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_emitter_K": 1700.0, "T_collector_K": 900.0})
    for k in ["phi_e_eV", "phi_c_eV", "J_emitter_Am2", "J_net_Am2",
              "V_terminal_V", "power_w", "efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC218"
    assert info["fidelity"] == "F1b"


def test_emission_positive(model):
    """Emitter emission must be positive at operating temperature."""
    r = model.predict({"T_emitter_K": 1700.0, "T_collector_K": 900.0})
    J_e = float(np.atleast_1d(r["J_emitter_Am2"])[0])
    assert J_e > 0


def test_power_positive(model):
    """Power output must be positive when T_emitter >> T_collector."""
    r = model.predict({"T_emitter_K": 1700.0, "T_collector_K": 900.0})
    P = float(np.atleast_1d(r["power_w"])[0])
    assert P > 0


def test_efficiency_below_carnot(model):
    """Efficiency must be below Carnot limit."""
    T_e, T_c = 1700.0, 900.0
    r = model.predict({"T_emitter_K": T_e, "T_collector_K": T_c})
    eta = float(np.atleast_1d(r["efficiency"])[0])
    eta_carnot = 1.0 - T_c / T_e
    assert eta <= eta_carnot + 0.01, f"eta={eta:.4f} > Carnot={eta_carnot:.4f}"


def test_efficiency_positive(model):
    r = model.predict({"T_emitter_K": 1700.0, "T_collector_K": 900.0})
    eta = float(np.atleast_1d(r["efficiency"])[0])
    assert eta >= 0.0


def test_power_increases_with_emitter_temperature(model):
    """Higher emitter temperature -> more emission -> more power."""
    Ps = []
    for T_e in [1400.0, 1600.0, 1800.0, 2000.0]:
        r = model.predict({"T_emitter_K": T_e, "T_collector_K": 900.0})
        Ps.append(float(np.atleast_1d(r["power_w"])[0]))
    assert all(Ps[i] <= Ps[i + 1] for i in range(len(Ps) - 1)), \
        f"Power not increasing with T_emitter: {Ps}"


def test_space_charge_reduces_current_vs_f1a(model):
    """F1b space-charge correction means J < pure Richardson-Dushman."""
    T_e, T_c = 1700.0, 900.0
    r = model.predict({"T_emitter_K": T_e, "T_collector_K": T_c})
    J_sc = float(np.atleast_1d(r["J_emitter_Am2"])[0])
    # Pure RD without space-charge (f_sc=1)
    phi_e = float(np.atleast_1d(r["phi_e_eV"])[0])
    A_r = model._model.A_r
    J_rd = A_r * T_e**2 * np.exp(-phi_e * q_e / (k_B * T_e))
    f_sc = model._model.f_sc
    assert abs(J_sc - J_rd * f_sc) < 1e-3 * J_rd, \
        f"Space-charge factor not applied correctly"


def test_work_function_decreases_with_T(model):
    """Emitter work function decreases with temperature (dphi/dT < 0)."""
    phi_1600 = float(np.atleast_1d(
        model.predict({"T_emitter_K": 1600.0, "T_collector_K": 900.0})["phi_e_eV"]
    )[0])
    phi_1800 = float(np.atleast_1d(
        model.predict({"T_emitter_K": 1800.0, "T_collector_K": 900.0})["phi_e_eV"]
    )[0])
    assert phi_1800 < phi_1600, "Work function should decrease with T"


def test_terminal_voltage_less_than_open_circuit(model):
    """Lead resistance should reduce terminal voltage below open-circuit."""
    r = model.predict({"T_emitter_K": 1700.0, "T_collector_K": 900.0})
    V_open = float(np.atleast_1d(r["V_open_V"])[0])
    V_term = float(np.atleast_1d(r["V_terminal_V"])[0])
    assert V_term <= V_open + 1e-10, f"V_terminal={V_term:.4f} > V_open={V_open:.4f}"


def test_efficiency_increases_with_emitter_T(model):
    """Higher emitter T should give higher efficiency (more Peltier, less back-emission)."""
    etas = []
    for T_e in [1400.0, 1600.0, 1800.0, 2000.0]:
        r = model.predict({"T_emitter_K": T_e, "T_collector_K": 900.0})
        etas.append(float(np.atleast_1d(r["efficiency"])[0]))
    assert all(etas[i] <= etas[i + 1] for i in range(len(etas) - 1)), \
        f"Efficiency not increasing with T_emitter: {etas}"


def test_benchmark(model):
    start = time.perf_counter()
    for T_e in np.random.uniform(1300, 2000, 100):
        model.predict({"T_emitter_K": float(T_e), "T_collector_K": 900.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 5.0
