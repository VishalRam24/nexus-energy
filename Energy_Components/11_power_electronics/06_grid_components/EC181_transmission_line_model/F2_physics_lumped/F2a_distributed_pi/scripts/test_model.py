"""
EC181 — Transmission Line — F2a Distributed-Parameter / Cascaded-Pi
Test suite: physics sanity (energy conservation, losses ~ I^2, Ferranti, reciprocity),
ABCD exact-vs-pi limits, ODE dynamics, edge cases, predict() interface, benchmark.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TransmissionLineF2a
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
def test_abcd_reciprocity():
    print("\n[Test 1] ABCD reciprocity AD - BC = 1 (exact & nominal-pi)")
    m, _ = make_model()
    for length in [50.0, 200.0, 500.0]:
        A, B, C, D = m.abcd_exact(length)
        res = abs(A * D - B * C - 1.0)
        assert_true(res < 1e-9, f"exact L={length}km: |AD-BC-1|={res:.2e}")
        A, B, C, D = m.abcd_nominal_pi(length)
        res = abs(A * D - B * C - 1.0)
        assert_true(res < 1e-12, f"nominal-pi L={length}km: |AD-BC-1|={res:.2e}")


def test_abcd_symmetry():
    print("\n[Test 2] Symmetric line => A == D")
    m, _ = make_model()
    A, B, C, D = m.abcd_exact(300.0)
    assert_true(abs(A - D) < 1e-12, f"A=D ({abs(A-D):.2e}) for symmetric line")


def test_pi_converges_to_exact():
    print("\n[Test 3] Nominal-pi -> exact as line shortens (short-line limit)")
    m, _ = make_model()
    # short line: pi and exact must nearly coincide
    A_e, B_e, C_e, D_e = m.abcd_exact(20.0)
    A_p, B_p, C_p, D_p = m.abcd_nominal_pi(20.0)
    relB = abs(B_e - B_p) / abs(B_e)
    assert_true(relB < 1e-3, f"short line B match: rel diff={relB:.2e}")
    # long line: they should differ measurably (distributed effect matters)
    A_e2, B_e2, _, _ = m.abcd_exact(500.0)
    A_p2, B_p2, _, _ = m.abcd_nominal_pi(500.0)
    relB2 = abs(B_e2 - B_p2) / abs(B_e2)
    assert_true(relB2 > relB, f"long line shows larger pi error ({relB2:.2e} > {relB:.2e})")


def test_surge_and_sil():
    print("\n[Test 4] Surge impedance & SIL in physical range")
    m, _ = make_model()
    Zs = m.surge_impedance()
    sil = m.sil_MW()
    assert_true(200.0 < Zs < 450.0, f"surge Z={Zs:.1f} ohm in [200,450] (OH line)")
    assert_true(50.0 < sil < 400.0, f"SIL={sil:.1f} MW physical for 230 kV")
    # Z_char (with losses) close to surge impedance (losses small)
    _, Zc = m.gamma_zc()
    assert_true(abs(abs(Zc) - Zs) / Zs < 0.1, "Z_char ~ surge Z (low-loss line)")


def test_ferranti_light_load():
    print("\n[Test 5] Ferranti effect: |V_r| > |V_s| at no/light load")
    m, _ = make_model()
    f = m.ferranti_no_load(1.0, length_km=300.0)
    assert_true(f["ferranti"] and f["V_r_pu"] > 1.0,
                f"no-load V_r={f['V_r_pu']:.4f} > 1.0 (rise x{f['rise_factor']:.4f})")
    # Longer line => stronger Ferranti rise
    f_short = m.ferranti_no_load(1.0, length_km=100.0)
    f_long = m.ferranti_no_load(1.0, length_km=400.0)
    assert_true(f_long["rise_factor"] > f_short["rise_factor"],
                f"longer line larger rise ({f_long['rise_factor']:.4f} > {f_short['rise_factor']:.4f})")


def test_voltage_drop_under_load():
    print("\n[Test 6] Voltage drop increases with load; heavy load => V_r < V_s")
    m, _ = make_model()
    # Light load on a long line can still show a net rise (partial Ferranti from
    # line charging); the monotone trend with load is the robust physical check.
    # Sweep within the stable (pre-collapse) region P <= 2 pu and check monotone drop.
    drops = [m.solve_receiving(1.0, P, 0.4 * P, length_km=300.0)["voltage_drop_pu"]
             for P in [0.5, 1.0, 1.5, 2.0]]
    assert_true(all(b > a for a, b in zip(drops, drops[1:])),
                f"drop monotonically increases with load: {[f'{d:.3f}' for d in drops]}")
    r_heavy = m.solve_receiving(1.0, 1.5, 0.6, length_km=300.0)
    assert_true(r_heavy["V_r_pu"] < 1.0,
                f"heavy load: V_r={r_heavy['V_r_pu']:.4f} < V_s (inductive sag dominates)")


def test_losses_scale_with_current_squared():
    print("\n[Test 7] Series losses scale ~ I^2 (short line, shunt negligible)")
    m, _ = make_model()
    # On a SHORT line the shunt charging current is tiny, so the series current is
    # essentially the load current and P_loss = I^2 * R must scale cleanly.
    Lkm = 30.0
    r1 = m.solve_receiving(1.0, 0.5, 0.0, length_km=Lkm)
    r2 = m.solve_receiving(1.0, 1.0, 0.0, length_km=Lkm)
    ratio_I2 = (r2["I_s_pu"] / r1["I_s_pu"]) ** 2
    ratio_loss = r2["P_loss_pu"] / r1["P_loss_pu"]
    assert_true(abs(ratio_loss / ratio_I2 - 1.0) < 0.1,
                f"loss ratio {ratio_loss:.3f} ~ I^2 ratio {ratio_I2:.3f} (short line)")
    # Direct identity: P_loss = |I|^2 * R_total for the short line (I ~ I_s ~ I_r)
    z, _ = m.primary_constants()
    R_total = z.real * Lkm / m.Z_base
    I_eff2 = 0.5 * (r2["I_s_pu"] ** 2 + r2["I_r_pu"] ** 2)
    pred_loss = I_eff2 * R_total
    rel = abs(pred_loss - r2["P_loss_pu"]) / r2["P_loss_pu"]
    assert_true(rel < 0.05, f"P_loss = I^2*R identity holds (rel err {rel:.3f})")
    assert_true(r1["P_loss_pu"] > 0 and r2["P_loss_pu"] > 0, "losses positive")


def test_energy_conservation():
    print("\n[Test 8] Energy conservation: P_s = P_load + P_loss")
    m, _ = make_model()
    r = m.solve_receiving(1.0, 1.2, 0.4, length_km=300.0)
    residual = abs(r["P_s_pu"] - (1.2 + r["P_loss_pu"]))
    assert_true(residual < 1e-9, f"P_s - (P_load+P_loss) = {residual:.2e}")
    assert_true(0.0 < r["efficiency"] < 1.0, f"efficiency={r['efficiency']:.4f} in (0,1)")


def test_dynamic_charging_open_end():
    print("\n[Test 9] Cascaded-pi ODE: open-end charging transient settles")
    m, _ = make_model()
    sim = m.simulate(1.0, n_sections=6, length_km=300.0, duration_s=0.08, open_end=True)
    assert_true(sim["success"], "solve_ivp converged")
    # open-end receiving voltage swings to ~ Ferranti-amplified peak (> source peak)
    vr_peak = np.max(np.abs(sim["v_r"]))
    vs_peak = np.max(np.abs(sim["v_s"]))
    assert_true(vr_peak > vs_peak * 0.99,
                f"open-end v_r peak {vr_peak:.3f} >= v_s peak {vs_peak:.3f} (Ferranti dynamic)")


def test_dynamic_energy_balance():
    print("\n[Test 10] Dynamic loaded run: time-avg P_in = P_load + P_loss")
    m, _ = make_model()
    sim = m.simulate(1.0, P_load_pu=1.0, n_sections=6, length_km=200.0, duration_s=0.15)
    assert_true(sim["success"], "solve_ivp converged")
    # average over the last ~3 cycles (steady AC), skip startup
    t = sim["t"]
    mask = t > t[-1] - 3.0 / m.f_Hz
    p_in = np.mean(sim["p_in"][mask])
    p_load = np.mean(sim["p_load"][mask])
    p_loss = np.mean(sim["p_loss"][mask])
    # account for shunt-capacitor stored energy: in steady AC avg dE/dt -> 0,
    # so P_in ~= P_load + P_loss within numerical tolerance
    resid = abs(p_in - (p_load + p_loss)) / max(abs(p_in), 1e-6)
    assert_true(p_in > 0 and p_load > 0, f"P_in={p_in:.4f}, P_load={p_load:.4f} positive")
    assert_true(resid < 0.15, f"avg power balance residual {resid:.3f} < 0.15")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    r = cm.predict({"V_s_pu": 1.0, "P_load_pu": 1.0, "Q_load_pu": 0.3, "length_km": 250.0})
    for key in ["V_r_pu", "I_s_pu", "P_loss_pu", "efficiency", "ABCD",
                "SIL_MW", "ferranti", "reciprocity_residual"]:
        assert_true(key in r, f"Key '{key}' in predict() output")
    assert_true(r["reciprocity_residual"] < 1e-9, "predict reciprocity holds")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC181" and info["version"] == "1.0.0",
                "metadata id/version correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: dynamic 8-section, ~9 cycle sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, P_load_pu=1.0, n_sections=8, length_km=300.0, duration_s=0.15)
    elapsed = time.perf_counter() - t0
    print(f"  dynamic sim in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_abcd_reciprocity,
        test_abcd_symmetry,
        test_pi_converges_to_exact,
        test_surge_and_sil,
        test_ferranti_light_load,
        test_voltage_drop_under_load,
        test_losses_scale_with_current_squared,
        test_energy_conservation,
        test_dynamic_charging_open_end,
        test_dynamic_energy_balance,
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
    print(f"EC181 Transmission Line F2a — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
