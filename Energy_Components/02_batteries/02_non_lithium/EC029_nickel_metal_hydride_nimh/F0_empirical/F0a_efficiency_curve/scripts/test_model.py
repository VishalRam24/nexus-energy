"""
EC029 -- Nickel-Metal Hydride (NiMH) Battery -- F0a test harness (no pytest).
Run:  python3 scripts/test_model.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import NiMHBatteryF0a
from predict import ComponentModel

PARAMS = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
passed = 0
failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \u2713 {msg}")
    else:
        failed += 1
        print(f"  \u2717 {msg}")


def main():
    params = json.load(open(PARAMS))
    m = NiMHBatteryF0a(params)
    cm = ComponentModel(PARAMS)

    # 1. efficiency within physical bounds across the range
    cs = np.linspace(0.0, m.crate_max, 25)
    etas = m.round_trip_efficiency(cs)
    assert_true(np.all(etas > 0.0) and np.all(etas <= 1.0),
                "round-trip efficiency stays in (0, 1] across C-rate range")

    # 2. monotonic non-increasing with C-rate (higher current -> more ohmic loss)
    assert_true(np.all(np.diff(etas) <= 1e-9),
                "efficiency is monotonically non-increasing with C-rate")

    # 3. rated point: eta at 0C == 1.0 (no ohmic loss at zero current)
    assert_true(abs(float(m.round_trip_efficiency(0.0)) - 1.0) < 1e-9,
                "efficiency at 0C equals 1.0 (datasheet rated point)")

    # 4. datasheet-derived endpoint: eta at crate_max matches the analytic formula
    f = m.crate_max * m.capacity * m.r_internal / m.nominal_voltage
    expect = ((1 - f) / (1 + f)) if f < 1 else 0.0
    expect = max(0.0, min(1.0, expect))
    assert_true(abs(float(m.round_trip_efficiency(m.crate_max)) - expect) < 1e-4,
                "efficiency at max C-rate matches ohmic-loss formula")

    # 5. edge inputs clamp (negative and beyond-max C-rate)
    assert_true(abs(float(m.round_trip_efficiency(-5.0)) - 1.0) < 1e-9
                and float(m.round_trip_efficiency(m.crate_max * 10)) <= float(m.round_trip_efficiency(m.crate_max)) + 1e-9,
                "out-of-range C-rate inputs clamp to valid bounds")

    # 6. usable energy <= input energy
    ue = float(m.usable_energy(0.5, 100.0))
    assert_true(0.0 <= ue <= 100.0, "usable energy after round trip <= energy in")

    # 7. predict() interface returns expected keys
    out = cm.predict({"c_rate": 0.5, "energy_in_wh": 100.0})
    assert_true(all(k in out for k in ("round_trip_efficiency", "loss_fraction", "usable_energy_wh")),
                "predict() returns efficiency, loss fraction and usable energy")

    # 8. fast benchmark: 1000 predictions
    t0 = time.time()
    for _ in range(1000):
        cm.predict({"c_rate": 0.5})
    dt = time.time() - t0
    assert_true(dt < 1.0, f"1000 predictions fast ({dt*1000:.1f} ms)")

    print(f"\nEC029 F0a: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
