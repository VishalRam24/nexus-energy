"""
EC211 -- Forward Osmosis (FO) -- F2a Physics-Lumped
Test suite: osmotic-flux physics sanity, concentration polarization,
mass conservation, no-applied-pressure check, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ForwardOsmosisF2a, LMH_TO_SI, BAR_TO_PA
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
def test_osmotic_pressure_vant_hoff():
    print("\n[Test 1] Osmotic pressure (van't Hoff) magnitude sane")
    m, _ = make_model()
    # 1 M NaCl (~1000 mol/m3) -> ~45-49 bar (textbook ~48 bar at 25C)
    pi = m.osmotic_pressure(1000.0) / BAR_TO_PA
    assert_true(40.0 < pi < 55.0, f"pi(1 M NaCl)={pi:.2f} bar in [40,55]")
    # monotonic in concentration
    assert_true(m.osmotic_pressure(2000.0) > m.osmotic_pressure(1000.0),
                "pi increases with concentration")


def test_flux_from_osmotic_driving_force():
    print("\n[Test 2] Water flux driven by osmotic difference, zero when none")
    m, _ = make_model()
    Jw = m.water_flux(1000.0, 17.1)
    assert_true(Jw > 0, f"Jw={Jw/LMH_TO_SI:.3f} LMH > 0 with draw>>feed")
    # equal osmotic pressures -> no driving force -> no flux
    Jw0 = m.water_flux(17.1, 17.1)
    assert_true(Jw0 == 0.0, "Jw=0 when pi_draw == pi_feed")
    # draw weaker than feed -> no forward flux
    Jw_rev = m.water_flux(10.0, 100.0)
    assert_true(Jw_rev == 0.0, "Jw=0 when pi_draw < pi_feed")


def test_no_applied_hydraulic_pressure():
    print("\n[Test 3] FO: flux ONLY from osmotic d_pi (no hydraulic term)")
    m, _ = make_model()
    # The implicit eq must satisfy Jw = A*(pi_draw_eff - pi_feed_eff) exactly,
    # i.e. NO additive applied-pressure (dP) term as in RO.
    c_draw = 1000.0
    Jw = m.water_flux(c_draw, 17.1)
    pi_draw_eff = m.osmotic_pressure(c_draw) * np.exp(-Jw * m.K_icp)
    pi_feed_eff = m.osmotic_pressure(17.1) * np.exp(Jw / m.k_ecp)
    residual = m.A_perm * (pi_draw_eff - pi_feed_eff) - Jw
    assert_true(abs(residual) < 1e-9,
                f"Jw = A*d_pi_eff with no dP term (residual={residual:.2e})")


def test_icp_reduces_flux():
    print("\n[Test 4] Concentration polarization REDUCES flux below ideal")
    m, _ = make_model()
    c_draw = 1000.0
    Jw_real = m.water_flux(c_draw, 17.1)
    Jw_ideal = m.A_perm * (m.osmotic_pressure(c_draw) - m.osmotic_pressure(17.1))
    assert_true(Jw_real < Jw_ideal,
                f"Jw_real={Jw_real/LMH_TO_SI:.2f} < Jw_ideal={Jw_ideal/LMH_TO_SI:.2f} LMH")
    # ICP is the dominant limiter: real flux is a small fraction of ideal
    assert_true(Jw_real < 0.6 * Jw_ideal,
                f"ICP-dominated: real/ideal={Jw_real/Jw_ideal:.2f} (<0.6)")


def test_flux_monotone_in_draw_conc():
    print("\n[Test 5] Flux increases monotonically with draw concentration")
    m, _ = make_model()
    cs = np.linspace(200.0, 3000.0, 30)
    Jw_prev = m.water_flux(cs[0], 17.1)
    for c in cs[1:]:
        Jw = m.water_flux(c, 17.1)
        assert_true(Jw >= Jw_prev - 1e-12,
                    f"Jw({c:.0f})={Jw/LMH_TO_SI:.3f} >= prev {Jw_prev/LMH_TO_SI:.3f} LMH")
        Jw_prev = Jw
    print("  All 29 pairs checked.")


def test_reverse_salt_flux():
    print("\n[Test 6] Reverse salt flux positive, draw->feed, scales with B")
    m, _ = make_model()
    Js = m.salt_flux(1000.0, 17.1)
    assert_true(Js > 0, f"Js={Js:.3e} mol/(m2.s) > 0 (salt leaks draw->feed)")
    # no concentration difference -> no reverse salt flux
    assert_true(m.salt_flux(17.1, 17.1) == 0.0, "Js=0 when c_draw==c_feed")


def test_draw_dilution_over_time():
    print("\n[Test 7] Draw solution dilutes: concentration drops, volume rises")
    m, _ = make_model()
    r = m.simulate(duration_s=7200.0, n_points=100)
    assert_true(r["c_draw_mol_m3"][-1] < r["c_draw_mol_m3"][0],
                f"c_draw {r['c_draw_mol_m3'][0]:.1f} -> {r['c_draw_mol_m3'][-1]:.1f} mol/m3")
    assert_true(r["V_draw_m3"][-1] > r["V_draw_m3"][0],
                f"V_draw {r['V_draw_m3'][0]:.4f} -> {r['V_draw_m3'][-1]:.4f} m3 (water in)")
    # flux declines as draw dilutes
    assert_true(r["Jw_LMH"][-1] < r["Jw_LMH"][0],
                f"Jw declines {r['Jw_LMH'][0]:.3f} -> {r['Jw_LMH'][-1]:.3f} LMH")


def test_mass_conservation():
    print("\n[Test 8] Mass conservation: water gained == integral of Qw")
    m, _ = make_model()
    r = m.simulate(duration_s=3600.0, n_points=400)
    # permeate (volume gained) should equal time-integral of Jw*Am
    Qw = r["Jw_m_s"] * m.Am
    integ = np.trapezoid(Qw, r["t"])
    gained = r["permeate_m3"][-1]
    rel_err = abs(integ - gained) / max(gained, 1e-12)
    assert_true(rel_err < 1e-2,
                f"V gained={gained*1000:.3f} L vs integral={integ*1000:.3f} L (err {rel_err*100:.2f}%)")
    # salt mass conserved: n_salt drop == integral of Js*Am
    n0 = r["c_draw_mol_m3"][0] * r["V_draw_m3"][0]
    nF = r["c_draw_mol_m3"][-1] * r["V_draw_m3"][-1]
    salt_lost = n0 - nF
    salt_integ = np.trapezoid(r["Js_mol_m2_s"] * m.Am, r["t"])
    rel_s = abs(salt_lost - salt_integ) / max(salt_lost, 1e-12)
    assert_true(rel_s < 5e-2,
                f"salt lost={salt_lost:.4f} vs integral={salt_integ:.4f} mol (err {rel_s*100:.2f}%)")


def test_flux_magnitude_realistic():
    print("\n[Test 9] FO flux in realistic LMH band (1-30 LMH)")
    m, _ = make_model()
    Jw_LMH = m.water_flux(1000.0, 17.1) / LMH_TO_SI
    assert_true(1.0 < Jw_LMH < 30.0,
                f"Jw={Jw_LMH:.2f} LMH within typical FO range [1,30]")


def test_regen_dominates_energy():
    print("\n[Test 10] Energy: draw regeneration dominates (FO step ~free)")
    _, cm = make_model()
    r = cm.predict({"duration_s": 3600.0, "n_points": 20})
    assert_true(r["SEC_regen_kWh_m3"] > 0.5,
                f"SEC_regen={r['SEC_regen_kWh_m3']:.2f} kWh/m3 is the dominant energy term")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface keys present")
    _, cm = make_model()
    r = cm.predict({"duration_s": 600.0, "n_points": 30})
    for key in ["t", "V_draw_m3", "c_draw_mol_m3", "Jw_LMH", "Js_gMH",
                "pi_draw_bar", "pi_feed_bar", "permeate_m3", "SEC_regen_kWh_m3"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["Jw_LMH"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC211" and info["version"] == "1.0.0",
                "get_info reports EC211 v1.0.0")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(duration_s=3600.0, n_points=200)
    elapsed = time.perf_counter() - t0
    print(f"  1 h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_osmotic_pressure_vant_hoff,
        test_flux_from_osmotic_driving_force,
        test_no_applied_hydraulic_pressure,
        test_icp_reduces_flux,
        test_flux_monotone_in_draw_conc,
        test_reverse_salt_flux,
        test_draw_dilution_over_time,
        test_mass_conservation,
        test_flux_magnitude_realistic,
        test_regen_dominates_energy,
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
    print(f"EC211 Forward Osmosis F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
