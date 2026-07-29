"""EC147 -- HTL -- F1b -- Simulation Scenarios"""
import numpy as np
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    feedstocks = ["microalgae_chlorella", "microalgae_nannochloropsis",
                  "sewage_sludge", "wood_biomass", "macroalgae_laminaria"]
    temps = np.linspace(270, 380, 20)
    moistures = np.linspace(0.40, 0.90, 15)

    # Temperature sweep
    temp_sweep = {}
    for fs in feedstocks:
        bc_arr, er_arr = [], []
        for T in temps:
            r = model.predict({"feedstock_type": fs, "temperature_degC": float(T), "moisture_fraction": 0.80})
            bc_arr.append(r["bio_crude_yield"])
            er_arr.append(r["energy_recovery"])
        temp_sweep[fs] = {"bio_crude": bc_arr, "energy_recovery": er_arr}
    results["temperature_sweep"] = {"temperatures_degC": temps.tolist(), "feedstocks": temp_sweep}

    # Moisture sweep
    moisture_data = {"bio_crude": [], "LHV_eff": [], "thermal_eff": []}
    for M in moistures:
        r = model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 330.0, "moisture_fraction": float(M)})
        moisture_data["bio_crude"].append(r["bio_crude_yield"])
        moisture_data["LHV_eff"].append(r["LHV_eff_MJ_kg"])
        moisture_data["thermal_eff"].append(r["thermal_efficiency"])
    results["moisture_sweep"] = {"moisture_fractions": moistures.tolist(), "chlorella_330degC": moisture_data}

    print("EC147 F1b Simulation complete.")
    return results


if __name__ == "__main__":
    run_simulations()
