"""F0a tests for EC205 CO2 Electrolyzer (CO2 to CO/Fuels) (no pytest)."""
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402

_passed = 0
_failed = 0


def assert_true(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1
        print("  \u2713 " + msg)
    else:
        _failed += 1
        print("  \u2717 " + msg)


def main():
    m = ComponentModel()
    info = m.get_info()
    assert_true(info["component_id"] == "EC205", "get_info reports EC205")
    out = m.predict({"co2_in_kg_s": 1.0, "current_density_mA_cm2": 200.0})
    assert_true(abs(out["faradaic_efficiency"] - 0.85) < 1e-6, "design FE == 0.85 (datasheet)")
    assert_true(abs(out["SEC_kWh_kgCO2"] - 8.0) < 1e-6, "design SEC == 8.0 kWh/kgCO2 (datasheet)")
    assert_true(0.0 < out["faradaic_efficiency"] < 1.0, "FE in (0,1)")
    assert_true(out["co_produced_kg_s"] > 0 and out["co_produced_kg_s"] < out["co2_converted_kg_s"], "CO mass < converted CO2 mass")
    # FE drops, SEC rises with current density
    fe_lo = m.predict({"current_density_mA_cm2": 50.0})["faradaic_efficiency"]
    fe_hi = m.predict({"current_density_mA_cm2": 400.0})["faradaic_efficiency"]
    assert_true(fe_hi < fe_lo, "FE drops with current density")
    sec_hi = m.predict({"current_density_mA_cm2": 400.0})["SEC_kWh_kgCO2"]
    assert_true(sec_hi > out["SEC_kWh_kgCO2"], "SEC rises with current density")
    t0 = time.time(); [m.predict({"current_density_mA_cm2": 200.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
