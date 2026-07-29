"""Tailored tests for EDLC Supercapacitor (EC041) F0a capacitance/ESR lookup. No pytest."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel

_failed = 0


def assert_true(cond, msg):
    global _failed
    if cond:
        print(f"  \u2713 {msg}")
    else:
        _failed += 1
        print(f"  \u2717 {msg}")


def main():
    m = ComponentModel()
    c = m.curve

    # 1. voltage at Q=C*V_max equals V_max (datasheet rated point)
    Qmax = c.C * c.V_max
    assert_true(abs(c.voltage(Qmax) - c.V_max) < 1e-9, "V(Q=C*V_max) == V_max")

    # 2. voltage linear & monotonic in charge
    qs = np.linspace(0, Qmax, 50)
    vs = c.voltage(qs)
    assert_true(np.all(np.diff(vs) > 0), "voltage monotonically increasing in charge")
    assert_true(abs(c.voltage(0.0)) < 1e-12, "V(0)=0")

    # 3. energy 0.5*C*V^2 positive & matches formula
    assert_true(abs(c.energy(c.V_max) - 0.5 * c.C * c.V_max ** 2) < 1e-6, "energy formula exact")
    assert_true(c.energy(c.V_max) > 0, "max energy positive")

    # 4. usable energy = full minus floor, in physical bounds
    eu = c.usable_energy()
    assert_true(0 < eu < c.energy(c.V_max) + 1e-9, "usable energy within (0, E_max]")

    # 5. round-trip efficiency physical and decreasing with current
    assert_true(0.0 < c.roundtrip_efficiency(1.0) <= 1.0, "eff in (0,1]")
    assert_true(c.roundtrip_efficiency(1.0) > c.roundtrip_efficiency(100.0),
                "efficiency drops with current")
    assert_true(abs(c.roundtrip_efficiency(0.0) - 1.0) < 1e-9, "eff(I=0) == 1")

    # 6. predict() interface
    out = m.predict({"charge_C": 0.5 * Qmax, "current": 10.0})
    assert_true("voltage" in out and "energy_J" in out and "roundtrip_efficiency" in out,
                "predict() returns voltage/energy/efficiency")
    info = m.get_info()
    assert_true(info["component_id"] == "EC041" and info["version"] == "1.0.0",
                "get_info() id/version correct")

    # 7. fast benchmark
    t0 = time.time()
    c.energy(np.linspace(0, c.V_max, 1000))
    dt = (time.time() - t0) * 1e3
    assert_true(dt < 50.0, f"1000-pt energy lookup fast ({dt:.2f} ms)")

    print(f"\n{'PASSED' if _failed == 0 else 'FAILED'}: {_failed} failure(s)")
    return _failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
