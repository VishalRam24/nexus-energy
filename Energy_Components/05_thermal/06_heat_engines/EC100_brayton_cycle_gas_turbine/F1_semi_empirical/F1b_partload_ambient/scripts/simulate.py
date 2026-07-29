"""EC100 — Brayton Cycle Gas Turbine — F1b — Simulation Scenarios"""

import json, numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    # Scenario 1: PLR sweep (ISO conditions)
    PLR = np.linspace(0.4, 1.0, 60)
    r1 = model.predict({"PLR": PLR, "T_ambient_k": 288.15})
    results["plr_sweep"] = {
        "PLR": PLR.tolist(),
        "efficiency": r1["efficiency"].tolist(),
        "power_output_kw": r1["power_output_kw"].tolist(),
        "heat_rate_kj_kwh": r1["heat_rate_kj_kwh"].tolist(),
    }

    # Scenario 2: Ambient temperature effect
    T_amb = np.linspace(248, 323, 70)
    r2 = model.predict({"PLR": 1.0, "T_ambient_k": T_amb})
    results["T_amb_sweep"] = {
        "T_amb_k": T_amb.tolist(),
        "efficiency": r2["efficiency"].tolist(),
        "power_output_kw": r2["power_output_kw"].tolist(),
        "f_amb_power": r2["f_amb_power"].tolist(),
    }

    # Scenario 3: Exhaust temperature vs PLR
    PLR2 = np.linspace(0.4, 1.0, 60)
    r3 = model.predict({"PLR": PLR2, "T_ambient_k": 288.15})
    results["exhaust_sweep"] = {
        "PLR": PLR2.tolist(),
        "exhaust_temp_k": r3["exhaust_temp_k"].tolist(),
    }

    return results


if __name__ == "__main__":
    results = run_simulations()
    print("EC100 F1b Brayton GT simulation complete.")
    PLR = np.array(results["plr_sweep"]["PLR"])
    eta = np.array(results["plr_sweep"]["efficiency"])
    print(f"  ISO full-load efficiency: {eta[-1]*100:.1f}%")
    P   = np.array(results["plr_sweep"]["power_output_kw"])
    print(f"  ISO full-load power: {P[-1]/1e3:.1f} MW")
