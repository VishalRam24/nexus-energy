"""
EC081 -- Thermochemical Energy Storage (CaO/Ca(OH)2) -- F2a Reaction Kinetics
Test suite: physics sanity (energy conservation, bounds, equilibrium gating,
loss-free hold), edge cases, predict() interface, benchmark timing.

Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ThermochemicalStorageF2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    cm = ComponentModel()
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_equilibrium_temperature():
    print("\n[Test 1] van't Hoff equilibrium temperature physical & P-dependent")
    m, _ = make_model()
    Teq = m.T_eq()  # at P_ref
    assert_true(600.0 < Teq < 900.0,
                f"T_eq={Teq-273.15:.1f} C in CaO/Ca(OH)2 range (~450-510 C)")
    # Lower water vapour pressure lowers equilibrium temperature
    Teq_low = m.T_eq(P_h2o=10000.0)
    assert_true(Teq_low < Teq,
                f"Lower P_h2o lowers T_eq: {Teq_low-273.15:.1f} < {Teq-273.15:.1f} C")


def test_stored_energy_linear_in_extent():
    print("\n[Test 2] Stored energy = reaction enthalpy x extent (linear in X)")
    m, _ = make_model()
    assert_true(abs(m.stored_energy(0.0)) < 1e-6, "E_stored(X=0) = 0")
    assert_true(abs(m.stored_energy(1.0) - m.E_max) < 1e-6, "E_stored(X=1) = E_max")
    e_half = m.stored_energy(0.5)
    assert_true(abs(e_half - 0.5 * m.E_max) < 1e-6,
                f"E_stored(0.5)={e_half/3.6e6:.2f} kWh = half of E_max")


def test_conversion_bounds():
    print("\n[Test 3] Conversion X stays in [0, 1] over full charge & discharge")
    m, _ = make_model()
    rc = m.simulate("charge", 0.0, 723.15, 873.15, dt=30.0, duration_s=10800.0)
    rd = m.simulate("discharge", 1.0, 663.15, 600.0, dt=30.0, duration_s=10800.0)
    assert_true(rc["X"].min() >= -1e-9 and rc["X"].max() <= 1.0 + 1e-9,
                f"charge X in [0,1] (max={rc['X'].max():.4f})")
    assert_true(rd["X"].min() >= -1e-9 and rd["X"].max() <= 1.0 + 1e-9,
                f"discharge X in [0,1] (min={rd['X'].min():.4f})")


def test_charge_increases_discharge_decreases():
    print("\n[Test 4] Charge raises SOC; discharge lowers SOC")
    m, _ = make_model()
    rc = m.simulate("charge", 0.0, 723.15, 873.15, dt=30.0, duration_s=10800.0)
    rd = m.simulate("discharge", 1.0, 663.15, 600.0, dt=30.0, duration_s=10800.0)
    assert_true(rc["SOC"][-1] > rc["SOC"][0] + 0.05,
                f"charge SOC {rc['SOC'][0]:.3f} -> {rc['SOC'][-1]:.3f}")
    assert_true(rd["SOC"][-1] < rd["SOC"][0] - 0.05,
                f"discharge SOC {rd['SOC'][0]:.3f} -> {rd['SOC'][-1]:.3f}")


def test_loss_free_hold():
    print("\n[Test 5] Halted reaction -> loss-free long-term storage (X const)")
    m, _ = make_model()
    # 30 days at fixed SOC with reaction halted
    r = m.simulate("hold", X0=0.65, T0=700.0, T_source=700.0,
                   dt=3600.0, duration_s=30 * 86400.0)
    dX = abs(r["X"][-1] - r["X"][0])
    assert_true(dX < 1e-9, f"SOC drift over 30 days = {dX:.2e} (loss-free)")
    assert_true(np.allclose(r["reaction_rate"], 0.0), "reaction rate identically 0")


def test_equilibrium_gates_reaction():
    print("\n[Test 6] Reaction direction gated by equilibrium temperature")
    m, _ = make_model()
    Teq = m.T_eq()
    # Charge (dehydration) needs T > T_eq: below eq -> rate 0
    r_below = m.reaction_rate(0.5, Teq - 30.0, "charge")
    r_above = m.reaction_rate(0.5, Teq + 30.0, "charge")
    assert_true(r_below == 0.0, "charge rate=0 below T_eq")
    assert_true(r_above > 0.0, f"charge rate>0 above T_eq ({r_above:.2e}/s)")
    # Discharge (hydration) needs T < T_eq
    d_below = m.reaction_rate(0.5, Teq - 30.0, "discharge")
    d_above = m.reaction_rate(0.5, Teq + 30.0, "discharge")
    assert_true(d_below < 0.0, f"discharge rate<0 below T_eq ({d_below:.2e}/s)")
    assert_true(d_above == 0.0, "discharge rate=0 above T_eq")


def test_arrhenius_temperature_dependence():
    print("\n[Test 7] Arrhenius: higher T (above eq) -> faster charge kinetics")
    m, _ = make_model()
    Teq = m.T_eq()
    # Stay in saturated driving-force regime (>= T_eq + window) to isolate
    # the exp(-Ea/RT) Arrhenius term.
    r1 = abs(m.reaction_rate(0.5, Teq + 60.0, "charge"))
    r2 = abs(m.reaction_rate(0.5, Teq + 120.0, "charge"))
    assert_true(r2 > r1, f"rate(T+120)={r2:.3e} > rate(T+60)={r1:.3e} /s")


def test_energy_conservation():
    print("\n[Test 8] Energy conservation: reaction heat balances enthalpy x dX")
    m, _ = make_model()
    r = m.simulate("discharge", 1.0, 663.15, 600.0, dt=10.0, duration_s=10800.0)
    # Integrate reaction heat over time; compare to enthalpy * delta-extent
    _trapz = getattr(np, "trapezoid", np.trapz)
    Q_int = _trapz(r["Q_rxn_W"], r["t"])            # J released by reaction
    dX = r["X"][0] - r["X"][-1]                       # extent consumed (>0)
    E_chem = m.n_mol * m.dH * dX                      # J of chemical energy
    rel_err = abs(Q_int - E_chem) / max(E_chem, 1.0)
    assert_true(rel_err < 0.02,
                f"int(Q_rxn)dt={Q_int/3.6e6:.2f} kWh vs dH*dX={E_chem/3.6e6:.2f} kWh "
                f"(rel err {rel_err*100:.2f}%)")


def test_discharge_releases_heat():
    print("\n[Test 9] Exothermic discharge releases heat (Q_rxn > 0)")
    m, _ = make_model()
    r = m.simulate("discharge", 1.0, 663.15, 600.0, dt=30.0, duration_s=7200.0)
    assert_true(r["Q_rxn_W"].max() > 0.0, "discharge Q_rxn peak > 0 (exothermic)")
    rc = m.simulate("charge", 0.0, 723.15, 873.15, dt=30.0, duration_s=7200.0)
    assert_true(rc["Q_rxn_W"].min() < 0.0, "charge Q_rxn < 0 (endothermic, absorbs heat)")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC081", "component_id == EC081")
    assert_true(cm.version == "1.0.0", "version == 1.0.0")
    r = cm.predict({"mode": "discharge", "duration_s": 1800.0, "dt": 60.0})
    for key in ["t", "X", "SOC", "temperature", "T_eq", "stored_energy_J",
                "reaction_rate", "Q_rxn_W", "E_max_J"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["X"]) == len(r["temperature"]),
                "Time-series arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 2h discharge sim at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate("discharge", 1.0, 663.15, 600.0, dt=10.0, duration_s=7200.0)
    elapsed = time.perf_counter() - t0
    print(f"  2h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_equilibrium_temperature,
        test_stored_energy_linear_in_extent,
        test_conversion_bounds,
        test_charge_increases_discharge_decreases,
        test_loss_free_hold,
        test_equilibrium_gates_reaction,
        test_arrhenius_temperature_dependence,
        test_energy_conservation,
        test_discharge_releases_heat,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC081 Thermochemical Storage F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
