"""EC150 -- FT BtL -- F1b -- Simulation Scenarios"""
import numpy as np
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    feedstocks = ["wood_chips", "pine", "agricultural_residue", "municipal_solid_waste", "torrefied_wood"]
    temps = np.linspace(190, 340, 20)

    # Temperature sweep — alpha and product selectivity
    temp_sweep = {}
    for fs in feedstocks:
        alphas, diesels, waxes = [], [], []
        for T in temps:
            r = model.predict({"feedstock_type": fs, "temperature_degC": float(T)})
            alphas.append(r["alpha"])
            diesels.append(r["product_selectivity"]["diesel"])
            waxes.append(r["product_selectivity"]["wax"])
        temp_sweep[fs] = {"alpha": alphas, "diesel_sel": diesels, "wax_sel": waxes}
    results["temperature_sweep"] = {"temperatures_degC": temps.tolist(), "feedstocks": temp_sweep}

    print("EC150 F1b Simulation complete.")
    return results


if __name__ == "__main__":
    run_simulations()
