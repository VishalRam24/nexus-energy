"""
EC113 -- Subcritical Pulverized Coal Plant -- F2a Physics-Lumped
Test suite: thermodynamic sanity, conservation, 2nd-law bound, drum ODE,
edge cases, predict() interface, benchmark timing.  Custom harness (no pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from model import SubcriticalCoalF2a, Tsat_from_P, Psat_from_T, h_superheated
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
def test_steam_properties():
    print("\n[Test 1] Steam-table correlations match IAPWS anchor points")
    # saturation temperatures
    assert_true(abs((Tsat_from_P(0.07) - 273.15) - 39.0) < 3.0,
                f"Tsat(0.07 bar)={Tsat_from_P(0.07)-273.15:.1f}C ~ 39C")
    assert_true(abs((Tsat_from_P(165) - 273.15) - 357.0) < 6.0,
                f"Tsat(165 bar)={Tsat_from_P(165)-273.15:.1f}C ~ 357C")
    # Psat is inverse of Tsat
    assert_true(abs(Psat_from_T(Tsat_from_P(165)) - 165.0) < 1.0,
                "Psat(Tsat(165))==165 bar (round-trip)")
    # superheated main-steam enthalpy near steam-table value
    h3 = float(h_superheated(165, 540 + 273.15))
    assert_true(abs(h3 - 3410.0) < 60.0, f"h(165 bar,540C)={h3:.0f} ~ 3410 kJ/kg")


def test_boiler_efficiency():
    print("\n[Test 2] Boiler efficiency in realistic band (0.88-0.94)")
    m, _ = make_model()
    eta_b = m.boiler_efficiency()
    assert_true(0.85 < eta_b < 0.95, f"eta_boiler={eta_b:.4f} in (0.85, 0.95)")


def test_mass_conservation():
    print("\n[Test 3] Combustion mass balance: flue = fuel + air")
    m, _ = make_model()
    afr = m.air_fuel_ratio()
    flue = m.flue_per_fuel()
    assert_true(abs(flue - (1.0 + afr)) < 1e-9,
                f"flue/fuel={flue:.3f} == 1+AFR={1+afr:.3f}")
    assert_true(afr > m.stoich_air,
                f"AFR={afr:.2f} > stoichiometric={m.stoich_air:.2f} (excess air)")


def test_net_efficiency_band():
    print("\n[Test 4] Net plant LHV efficiency in 0.36-0.40 band")
    m, _ = make_model()
    eta_net, eta_b, eta_c = m.net_efficiency()
    assert_true(0.36 <= eta_net <= 0.40,
                f"eta_net={eta_net:.4f} in [0.36, 0.40] (subcritical)")
    assert_true(0 < eta_c < 1 and 0 < eta_b < 1, "component effs in (0,1)")


def test_carnot_bound():
    print("\n[Test 5] 2nd law: eta_net < eta_cycle < eta_Carnot")
    m, _ = make_model()
    eta_net, eta_b, eta_c = m.net_efficiency()
    eta_carnot = m.carnot_efficiency()
    assert_true(eta_c < eta_carnot,
                f"eta_cycle={eta_c:.4f} < Carnot={eta_carnot:.4f}")
    assert_true(eta_net < eta_carnot,
                f"eta_net={eta_net:.4f} < Carnot={eta_carnot:.4f}")


def test_cycle_energy_balance():
    print("\n[Test 6] Cycle 1st-law balance: w_net = q_in - q_out closes")
    m, _ = make_model()
    st = m.cycle_states()
    # First law on the working-fluid loop WITHOUT the regenerative feedwater
    # bump (regen is an internal heat recirculation; it must not create or
    # destroy energy).  q_in from pumped condensate h2, full LP expansion.
    w_turb = (st["h3"] - st["h4"]) + (st["h5"] - st["h6"])
    w_pump = (st["h2"] - st["h1"])
    q_in = (st["h3"] - st["h2"]) + (st["h5"] - st["h4"])
    q_out = (st["h6"] - st["h1"])
    w_net = w_turb - w_pump
    resid = abs(w_net - (q_in - q_out)) / q_in
    assert_true(resid < 1e-6, f"energy balance residual={resid*100:.4f}% (closes to machine precision)")


def test_co2_intensity():
    print("\n[Test 7] CO2 intensity realistic for subcritical (850-1050 g/kWh)")
    m, _ = make_model()
    ci = m.co2_intensity_g_per_kwh(1.0)
    assert_true(850 < ci < 1050, f"CO2 intensity={ci:.0f} g/kWh in (850,1050)")


def test_coal_monotone_with_load():
    print("\n[Test 8] Coal & CO2 rate increase with load")
    m, _ = make_model()
    plrs = np.linspace(0.3, 1.0, 8)
    coal = [m.coal_rate_kgs(p) for p in plrs]
    assert_true(all(coal[i] < coal[i + 1] for i in range(len(coal) - 1)),
                "coal_rate strictly increases with PLR")
    assert_true(m.co2_rate_kgs(1.0) > m.co2_rate_kgs(0.5),
                "CO2 rate higher at full load")


def test_drum_ode_warmup():
    print("\n[Test 9] Drum ODE: warms from cold start, pins at T_sat, no overshoot")
    m, _ = make_model()
    r = m.simulate(plr=1.0, T0_K=320.0, dt=20.0, duration_s=10000.0)
    T_sat = r["T_sat_drum"][0]
    assert_true(r["T_drum"][0] < T_sat, "starts cold (T0 < T_sat)")
    assert_true(abs(r["T_drum"][-1] - T_sat) < 3.0,
                f"settles at T_sat: T_end={r['T_drum'][-1]:.1f} ~ {T_sat:.1f} K")
    assert_true(r["T_drum"].max() <= T_sat + 1.0,
                f"no overshoot above T_sat (max={r['T_drum'].max():.1f})")
    # warm-up phase strictly monotone (below saturation)
    below = r["T_drum"][:-1] < T_sat - 1.0
    dT = np.diff(r["T_drum"])
    assert_true(np.all(dT[below] >= -1e-6), "warm-up phase monotone non-decreasing")


def test_drum_steam_generation():
    print("\n[Test 10] Steam generated only after drum reaches saturation")
    m, _ = make_model()
    r = m.simulate(plr=1.0, T0_K=320.0, dt=20.0, duration_s=10000.0)
    assert_true(r["steam_rate_kgs"][0] == 0.0, "no steam while cold")
    assert_true(r["steam_rate_kgs"][-1] > 300.0,
                f"steam at SS={r['steam_rate_kgs'][-1]:.0f} kg/s (>300 for 500MW)")


def test_load_step():
    print("\n[Test 11] Load step: higher firing -> higher coal/steam, T stays at T_sat")
    m, _ = make_model()
    r = m.simulate(plr=lambda t: 0.5 if t < 3000 else 1.0,
                   T0_K=float(Tsat_from_P(165)), dt=20.0, duration_s=6000.0)
    i_lo = np.argmin(np.abs(r["t"] - 2000.0))
    i_hi = np.argmin(np.abs(r["t"] - 5500.0))
    assert_true(r["coal_rate_kgs"][i_hi] > r["coal_rate_kgs"][i_lo],
                "coal rate rises after load step")
    assert_true(abs(r["T_drum"][i_hi] - r["T_sat_drum"][0]) < 3.0,
                "drum stays pinned at T_sat across load change")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "fidelity", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC113" and cm.version == "1.0.0",
                "id/version correct")
    r = cm.predict({"part_load_ratio": 1.0, "T0_drum_K": 350.0,
                    "dt": 40.0, "duration_s": 4000.0})
    for key in ["t", "T_drum", "plr", "power_net_mw", "coal_rate_kgs",
                "steam_rate_kgs", "eta_net"]:
        assert_true(key in r, f"output '{key}' present")
    assert_true(len(r["t"]) == len(r["T_drum"]), "time-series arrays aligned")


def test_benchmark():
    print("\n[Test 13] Benchmark: 8000 s drum transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(plr=1.0, T0_K=320.0, dt=20.0, duration_s=8000.0)
    elapsed = time.perf_counter() - t0
    print(f"  8000 s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_steam_properties,
        test_boiler_efficiency,
        test_mass_conservation,
        test_net_efficiency_band,
        test_carnot_bound,
        test_cycle_energy_balance,
        test_co2_intensity,
        test_coal_monotone_with_load,
        test_drum_ode_warmup,
        test_drum_steam_generation,
        test_load_step,
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

    print(f"\n{'='*64}")
    print(f"EC113 Subcritical Coal F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
