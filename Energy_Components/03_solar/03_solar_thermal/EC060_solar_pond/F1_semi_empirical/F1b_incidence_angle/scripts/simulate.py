"""EC060 — Solar Pond — F1b — Simulation Scenarios"""

import json, numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    # Scenario 1: Irradiance sweep
    G = np.linspace(0, 1000, 100)
    r1 = model.predict({"irradiance_w_m2": G, "incidence_angle_deg": 30.0,
                         "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    results["irradiance_sweep"] = {"irradiance_w_m2": G.tolist(),
                                    "useful_heat_w": r1["useful_heat_w"].tolist(),
                                    "efficiency": r1["efficiency"].tolist()}

    # Scenario 2: Incidence angle sweep (IAM effect)
    theta = np.linspace(0, 80, 80)
    r2 = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": theta,
                         "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    results["angle_sweep"] = {"theta_deg": theta.tolist(),
                               "iam_factor": r2["iam_factor"].tolist(),
                               "useful_heat_w": r2["useful_heat_w"].tolist()}

    # Scenario 3: LCZ temperature sweep
    T_lcz = np.linspace(30, 95, 60)
    r3 = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                         "T_lcz_degC": T_lcz, "T_ambient_degC": 25.0})
    results["T_lcz_sweep"] = {"T_lcz_degC": T_lcz.tolist(),
                               "useful_heat_w": r3["useful_heat_w"].tolist(),
                               "efficiency": r3["efficiency"].tolist()}

    return results


if __name__ == "__main__":
    results = run_simulations()
    print("EC060 F1b Solar Pond simulation complete.")
    G = np.array(results["irradiance_sweep"]["irradiance_w_m2"])
    Q = np.array(results["irradiance_sweep"]["useful_heat_w"])
    print(f"  Peak useful heat: {Q[-1]/1000:.2f} kW at G=1000 W/m2")
