"""EC145 -- Pyrolysis Reactor -- F1b -- Simulation Scenarios"""
import json
import numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    feedstocks = ["wood_chips", "pine", "corn_stover", "rice_husk", "switchgrass"]
    temps = np.linspace(350, 650, 20)
    moistures = np.linspace(0.05, 0.50, 15)

    # Scenario 1: Temperature sweep for all feedstocks
    temp_sweep = {}
    for fs in feedstocks:
        bio_oils, chars, gases = [], [], []
        for T in temps:
            r = model.predict({"feedstock_type": fs, "temperature_degC": float(T), "moisture_fraction": 0.10})
            bio_oils.append(r["bio_oil_yield"])
            chars.append(r["char_yield"])
            gases.append(r["gas_yield"])
        temp_sweep[fs] = {"bio_oil": bio_oils, "char": chars, "gas": gases}
    results["temperature_sweep"] = {"temperatures_degC": temps.tolist(), "feedstocks": temp_sweep}

    # Scenario 2: Moisture sweep for wood_chips at 500 degC
    moisture_sweep = {}
    for key in ["bio_oil_yield", "char_yield", "LHV_eff_MJ_kg", "moisture_lhv_factor", "energy_recovery"]:
        moisture_sweep[key] = []
    for M in moistures:
        r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": float(M)})
        for key in moisture_sweep:
            moisture_sweep[key].append(r[key])
    results["moisture_sweep"] = {"moisture_fractions": moistures.tolist(), "wood_chips_500degC": moisture_sweep}

    # Scenario 3: Part-load efficiency
    plr_vals = np.linspace(0.2, 1.0, 20)
    plr_eta = [model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0,
                               "moisture_fraction": 0.10, "PLR": float(p)})["thermal_efficiency"]
               for p in plr_vals]
    results["part_load"] = {"PLR": plr_vals.tolist(), "thermal_efficiency": plr_eta}

    print("EC145 F1b Simulation complete.")
    return results


if __name__ == "__main__":
    run_simulations()
