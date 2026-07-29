"""
EC170 -- Solid State Transformer (SST) -- F2a Cascaded Averaged Model
Test suite: cascaded energy conservation, eta in (0,1) as product of stages,
bidirectional flow, voltage transformation, DC-link ODE sanity, predict interface.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SST_F2a
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
def test_efficiency_in_unit_interval():
    print("\n[Test 1] Total efficiency strictly in (0,1) over full load range")
    m, _ = make_model()
    for p in np.linspace(100.0, 12000.0, 40):
        eta = float(m.total_efficiency(p))
        assert_true(0.0 < eta < 1.0, f"eta(P={p:.0f})={eta:.5f} in (0,1)")
    print("  40 load points checked.")


def test_eta_is_product_of_stages():
    print("\n[Test 2] eta_total <= product of stage efficiencies (cascade)")
    m, _ = make_model()
    for p in [2000.0, 6000.0, 10000.0]:
        er, ed, ei = m.stage_efficiencies(p)
        prod = float(er) * float(ed) * float(ei)
        eta = float(m.total_efficiency(p))
        assert_true(0.0 < float(er) < 1.0 and 0.0 < float(ed) < 1.0 and 0.0 < float(ei) < 1.0,
                    f"each stage in (0,1) at P={p:.0f}: {float(er):.4f},{float(ed):.4f},{float(ei):.4f}")
        # eta_total includes core loss so must be <= switching product
        assert_true(eta <= prod + 1e-9, f"eta_total={eta:.5f} <= product={prod:.5f}")


def test_cascaded_energy_conservation():
    print("\n[Test 3] Cascaded energy conservation: P_source = P_delivered + P_loss")
    m, _ = make_model()
    for p in [1000.0, 5000.0, 9000.0]:
        c = m.cascade(p)
        src = float(c["p_source_w"])
        deliv = float(c["p_delivered_mag_w"])
        loss = float(c["p_loss_w"])
        assert_true(abs(src - (deliv + loss)) < 1e-6,
                    f"P={p:.0f}: src={src:.3f} == deliv+loss={deliv + loss:.3f}")
        assert_true(loss > 0.0, f"P={p:.0f}: loss={loss:.2f} W > 0")


def test_cascade_chain_matches_product():
    print("\n[Test 4] Stage-by-stage chain product equals delivered (switching part)")
    m, _ = make_model()
    p = 8000.0
    c = m.cascade(p)
    # p3_out is the stage-cascaded switching delivery; should equal mag*er*ed*ei
    chain = float(c["p_stage3_out_w"])
    er, ed, ei = m.stage_efficiencies(p)
    expected = p * float(er) * float(ed) * float(ei)
    assert_true(abs(chain - expected) < 1e-6,
                f"chain={chain:.4f} == p*er*ed*ei={expected:.4f}")


def test_bidirectional():
    print("\n[Test 5] Bidirectional: forward & reverse both deliver eta*|P|")
    m, _ = make_model()
    cf = m.cascade(+6000.0)
    cr = m.cascade(-6000.0)
    assert_true(float(cf["direction"]) > 0, "forward direction = +1")
    assert_true(float(cr["direction"]) < 0, "reverse direction = -1")
    # magnitudes of efficiency/loss symmetric
    assert_true(abs(float(cf["eta_total"]) - float(cr["eta_total"])) < 1e-9,
                "same eta magnitude both directions")
    assert_true(float(cr["p_delivered_w"]) < 0, "reverse delivered power is negative (LV->MV)")
    assert_true(float(cf["p_delivered_w"]) > 0, "forward delivered power is positive")


def test_voltage_transformation():
    print("\n[Test 6] Voltage transformation MV-AC -> LV-AC (step-down)")
    m, _ = make_model()
    v_lv = float(m.voltage_transform(10000.0))
    assert_true(v_lv < 10000.0, f"V_lv={v_lv:.1f} < V_hv=10000 (step-down)")
    # ratio = (Vlv_dc/Vhv_dc)/n = (400/800)/2 = 0.25
    assert_true(abs(v_lv - 2500.0) < 1e-6, f"V_lv={v_lv:.1f} == 2500 (ratio 0.25)")
    # linearity
    assert_true(abs(float(m.voltage_transform(20000.0)) - 2.0 * v_lv) < 1e-6,
                "voltage transform is linear")


def test_efficiency_load_rolloff():
    print("\n[Test 7] Efficiency droops at very light load (core loss) and high load")
    m, _ = make_model()
    eta_light = float(m.total_efficiency(200.0))
    eta_mid = float(m.total_efficiency(5000.0))
    eta_full = float(m.total_efficiency(10000.0))
    assert_true(eta_mid > eta_light, f"mid {eta_mid:.4f} > light {eta_light:.4f} (core-loss droop)")
    assert_true(eta_mid > eta_full, f"mid {eta_mid:.4f} > full {eta_full:.4f} (load rolloff)")


def test_dclink_ode_settles():
    print("\n[Test 8] DC-link ODE: links stay near nominal & DAB tracks command")
    m, _ = make_model()
    r = m.simulate(8000.0, dt=0.0005, duration_s=0.1)
    assert_true(r["success"], "solve_ivp succeeded")
    assert_true(abs(r["p_dab_w"][-1] - 8000.0) < 50.0,
                f"DAB power tracked command: {r['p_dab_w'][-1]:.1f} ~ 8000 W")
    # DC links remain physically bounded (regulated buffers)
    assert_true(np.all(r["v_hv_dc"] > 0) and np.all(r["v_lv_dc"] > 0), "DC links stay positive")
    drift_hv = abs(r["v_hv_dc"][-1] - 800.0)
    assert_true(drift_hv < 200.0, f"HV link bounded near nominal (drift {drift_hv:.1f} V)")


def test_dclink_reverse_flow():
    print("\n[Test 9] DC-link ODE handles reverse power command")
    m, _ = make_model()
    r = m.simulate(-6000.0, dt=0.0005, duration_s=0.08)
    assert_true(r["success"], "reverse solve_ivp succeeded")
    assert_true(r["p_dab_w"][-1] < -1000.0, f"DAB power negative (reverse): {r['p_dab_w'][-1]:.1f} W")
    assert_true(np.all(np.asarray(r["efficiency"]) < 1.0), "efficiency < 1 throughout reverse run")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"p_command_w": 7000.0, "power_factor": 0.95,
                    "duration_s": 0.05, "dt": 0.001})
    for key in ["t", "v_hv_dc", "v_lv_dc", "p_dab_w", "p_delivered_w",
                "efficiency", "p_loss_w", "cascade_summary", "v_lv_ac_transformed_v"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_hv_dc"]), "time arrays same length")
    cs = r["cascade_summary"]
    assert_true(0.0 < cs["eta_total"] < 1.0, f"summary eta_total={cs['eta_total']:.4f} in (0,1)")


def test_power_factor_effect():
    print("\n[Test 11] Lower power factor reduces rectifier (and total) efficiency")
    m, _ = make_model()
    eta_pf1 = float(m.total_efficiency(8000.0, power_factor=1.0))
    eta_pf07 = float(m.total_efficiency(8000.0, power_factor=0.7))
    assert_true(eta_pf07 < eta_pf1, f"eta(pf=0.7)={eta_pf07:.4f} < eta(pf=1.0)={eta_pf1:.4f}")


def test_benchmark():
    print("\n[Test 12] Benchmark: 0.2 s DC-link ODE simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(8000.0, dt=0.001, duration_s=0.2)
    elapsed = time.perf_counter() - t0
    print(f"  0.2 s simulation in {elapsed * 1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_in_unit_interval,
        test_eta_is_product_of_stages,
        test_cascaded_energy_conservation,
        test_cascade_chain_matches_product,
        test_bidirectional,
        test_voltage_transformation,
        test_efficiency_load_rolloff,
        test_dclink_ode_settles,
        test_dclink_reverse_flow,
        test_predict_interface,
        test_power_factor_effect,
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

    print(f"\n{'=' * 60}")
    print(f"EC170 SST F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
