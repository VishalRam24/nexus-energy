"""EC061 — Unglazed Solar Collector (Pool Heating) — F1b — Simulation Scenarios"""

import json, numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    # Scenario 1: Irradiance sweep
    G = np.linspace(0, 1000, 100)
    r1 = model.predict({"irradiance_w_m2": G, "incidence_angle_deg": 30.0,
                         "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": 2.0})
    results["irradiance_sweep"] = {"irradiance_w_m2": G.tolist(),
                                    "useful_heat_w": r1["useful_heat_w"].tolist(),
                                    "efficiency": r1["efficiency"].tolist()}

    # Scenario 2: Angle sweep
    theta = np.linspace(0, 80, 80)
    r2 = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": theta,
                         "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": 2.0})
    results["angle_sweep"] = {"theta_deg": theta.tolist(),
                               "iam_factor": r2["iam_factor"].tolist(),
                               "useful_heat_w": r2["useful_heat_w"].tolist()}

    # Scenario 3: Wind speed effect
    v_wind = np.linspace(0, 10, 60)
    r3 = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                         "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": v_wind})
    results["wind_sweep"] = {"v_wind_m_s": v_wind.tolist(),
                              "useful_heat_w": r3["useful_heat_w"].tolist(),
                              "efficiency": r3["efficiency"].tolist(),
                              "U_L_effective": r3["U_L_effective"].tolist()}

    return results


if __name__ == "__main__":
    results = run_simulations()
    print("EC061 F1b Unglazed Solar Collector simulation complete.")
    G = np.array(results["irradiance_sweep"]["irradiance_w_m2"])
    Q = np.array(results["irradiance_sweep"]["useful_heat_w"])
    print(f"  Peak useful heat: {Q[-1]/1000:.2f} kW at G=1000 W/m2")
