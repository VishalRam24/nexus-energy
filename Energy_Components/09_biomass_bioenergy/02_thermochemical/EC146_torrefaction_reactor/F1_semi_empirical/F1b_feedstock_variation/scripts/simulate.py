"""EC146 -- Torrefaction Reactor -- F1b -- Simulation Scenarios"""
import numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    feedstocks = ["wood_chips", "pine", "wheat_straw", "bamboo", "miscanthus"]
    temps = np.linspace(210, 295, 20)
    times = np.linspace(10, 90, 15)

    # Temperature sweep at fixed residence time
    temp_sweep = {}
    for fs in feedstocks:
        MY_arr, EDR_arr, EY_arr = [], [], []
        for T in temps:
            r = model.predict({"feedstock_type": fs, "temperature_degC": float(T),
                               "residence_time_min": 30.0, "moisture_fraction": 0.10})
            MY_arr.append(r["mass_yield"])
            EDR_arr.append(r["energy_densification"])
            EY_arr.append(r["energy_yield"])
        temp_sweep[fs] = {"mass_yield": MY_arr, "EDR": EDR_arr, "energy_yield": EY_arr}
    results["temperature_sweep"] = {"temperatures_degC": temps.tolist(), "feedstocks": temp_sweep}

    # Residence time sweep
    time_sweep = {"mass_yield": [], "energy_yield": [], "torrefied_LHV": []}
    for t in times:
        r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 250.0,
                           "residence_time_min": float(t), "moisture_fraction": 0.10})
        time_sweep["mass_yield"].append(r["mass_yield"])
        time_sweep["energy_yield"].append(r["energy_yield"])
        time_sweep["torrefied_LHV"].append(r["torrefied_LHV_MJ_kg"])
    results["residence_time_sweep"] = {"times_min": times.tolist(), "wood_chips_250degC": time_sweep}

    print("EC146 F1b Simulation complete.")
    return results


if __name__ == "__main__":
    run_simulations()
