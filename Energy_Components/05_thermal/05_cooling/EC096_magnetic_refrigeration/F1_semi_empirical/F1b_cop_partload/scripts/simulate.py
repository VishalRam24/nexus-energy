"""EC096 — Magnetic Refrigeration — F1b — Simulation Scenarios"""

import json, numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    # Scenario 1: PLR sweep
    PLR = np.linspace(0.3, 1.0, 60)
    r1 = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": PLR})
    results["plr_sweep"] = {"PLR": PLR.tolist(),
                             "cop": r1["cop"].tolist(),
                             "cooling_kw": r1["cooling_kw"].tolist()}

    # Scenario 2: Hot-side temperature sweep
    T_hot = np.linspace(25, 55, 60)
    r2 = model.predict({"T_hot_degC": T_hot, "T_cold_degC": 15.0, "PLR": 1.0})
    results["T_hot_sweep"] = {"T_hot_degC": T_hot.tolist(),
                               "cop": r2["cop"].tolist(),
                               "cop_carnot": r2["cop_carnot"].tolist()}

    # Scenario 3: Temperature span sweep
    T_cold = np.linspace(5, 30, 60)
    r3 = model.predict({"T_hot_degC": 35.0, "T_cold_degC": T_cold, "PLR": 1.0})
    results["span_sweep"] = {"T_cold_degC": T_cold.tolist(),
                              "cop": r3["cop"].tolist(),
                              "delta_T_span_K": r3["delta_T_span_K"].tolist()}

    return results


if __name__ == "__main__":
    results = run_simulations()
    print("EC096 F1b Magnetic Refrigeration simulation complete.")
    PLR = np.array(results["plr_sweep"]["PLR"])
    COP = np.array(results["plr_sweep"]["cop"])
    print(f"  COP at PLR=1.0: {COP[-1]:.3f}")
    print(f"  COP at PLR=0.5: {COP[len(COP)//2]:.3f}")
