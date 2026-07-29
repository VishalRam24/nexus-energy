"""EC057 — Stirling Dish CSP — F1b — Simulation Scenarios"""

import json, numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    # Scenario 1: DNI sweep
    dni_arr = np.linspace(0, 1000, 100)
    r1 = model.predict({"dni_w_m2": dni_arr, "theta_deg": 0.0,
                         "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    results["dni_sweep"] = {
        "dni_w_m2": dni_arr.tolist(),
        "power_output_kw": r1["power_output_kw"].tolist(),
        "overall_efficiency": r1["overall_efficiency"].tolist(),
    }

    # Scenario 2: Part-load sweep
    plr_arr = np.linspace(0.3, 1.0, 50)
    r2 = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                         "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": plr_arr})
    results["plr_sweep"] = {
        "PLR": plr_arr.tolist(),
        "power_output_kw": r2["power_output_kw"].tolist(),
        "eta_stirling": r2["eta_stirling"].tolist(),
    }

    # Scenario 3: Receiver temperature sweep
    T_rec_arr = np.linspace(400, 800, 80)
    r3 = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                         "T_receiver_degC": T_rec_arr, "T_ambient_degC": 25.0, "PLR": 1.0})
    results["T_rec_sweep"] = {
        "T_rec_degC": T_rec_arr.tolist(),
        "power_output_kw": r3["power_output_kw"].tolist(),
        "Q_receiver_loss_kw": r3["Q_receiver_loss_kw"].tolist(),
        "eta_stirling": r3["eta_stirling"].tolist(),
    }

    # Scenario 4: Incidence angle sweep
    theta_arr = np.linspace(0, 15, 50)
    r4 = model.predict({"dni_w_m2": 900.0, "theta_deg": theta_arr,
                         "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    results["theta_sweep"] = {
        "theta_deg": theta_arr.tolist(),
        "power_output_kw": r4["power_output_kw"].tolist(),
        "iam_factor": r4["iam_factor"].tolist(),
    }

    return results


if __name__ == "__main__":
    results = run_simulations()
    print("EC057 F1b Simulation complete.")
    dni = np.array(results["dni_sweep"]["dni_w_m2"])
    P   = np.array(results["dni_sweep"]["power_output_kw"])
    peak_idx = np.argmax(P)
    print(f"  Peak power: {P[peak_idx]:.2f} kW at DNI={dni[peak_idx]:.0f} W/m2")
