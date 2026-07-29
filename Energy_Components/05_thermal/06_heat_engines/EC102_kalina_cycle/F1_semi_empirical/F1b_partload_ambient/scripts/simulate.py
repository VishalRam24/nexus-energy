"""EC102 — Kalina Cycle — F1b — Simulation Scenarios"""

import json, numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    # Scenario 1: PLR sweep
    PLR = np.linspace(0.3, 1.0, 60)
    r1 = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": PLR})
    results["plr_sweep"] = {"PLR": PLR.tolist(),
                             "efficiency": r1["efficiency"].tolist(),
                             "power_output_kw": r1["power_output_kw"].tolist()}

    # Scenario 2: Condenser temperature sensitivity
    T_cond = np.linspace(15, 55, 60)
    r2 = model.predict({"T_heat_source": 150.0, "T_condenser": T_cond, "PLR": 1.0})
    results["T_cond_sweep"] = {"T_cond_degC": T_cond.tolist(),
                                "efficiency": r2["efficiency"].tolist(),
                                "f_condenser": r2["f_condenser"].tolist()}

    # Scenario 3: NH3 fraction sweep
    x_arr = np.linspace(0.70, 0.95, 50)
    r3 = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0,
                         "PLR": 1.0, "x_NH3": x_arr})
    results["x_NH3_sweep"] = {"x_NH3": x_arr.tolist(),
                               "efficiency": r3["efficiency"].tolist(),
                               "f_composition": r3["f_composition"].tolist()}

    # Scenario 4: Heat source temperature sweep
    T_hot = np.linspace(80, 220, 70)
    r4 = model.predict({"T_heat_source": T_hot, "T_condenser": 32.0, "PLR": 1.0})
    results["T_hot_sweep"] = {"T_hot_degC": T_hot.tolist(),
                               "efficiency": r4["efficiency"].tolist(),
                               "eta_carnot": r4["eta_carnot"].tolist()}

    return results


if __name__ == "__main__":
    results = run_simulations()
    print("EC102 F1b Kalina Cycle simulation complete.")
    r = results["plr_sweep"]
    print(f"  Efficiency at PLR=1.0: {r['efficiency'][-1]*100:.2f}%")
    print(f"  Efficiency at PLR=0.5: {r['efficiency'][len(r['efficiency'])//2]*100:.2f}%")
