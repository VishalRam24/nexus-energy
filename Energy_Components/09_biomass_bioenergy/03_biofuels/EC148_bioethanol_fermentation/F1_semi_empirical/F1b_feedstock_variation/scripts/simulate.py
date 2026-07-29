"""EC148 -- Bioethanol Fermentation -- F1b -- Simulation Scenarios"""
import numpy as np
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    feedstocks = ["sugarcane", "corn", "wheat_straw", "switchgrass", "sweet_sorghum"]
    temps = np.linspace(20, 45, 20)
    moistures = np.linspace(0.05, 0.60, 15)

    # Temperature sweep
    temp_sweep = {}
    for fs in feedstocks:
        yields, f_Ts = [], []
        for T in temps:
            r = model.predict({"feedstock_type": fs, "temperature_degC": float(T)})
            yields.append(r["ethanol_yield"])
            f_Ts.append(r["temperature_factor"])
        temp_sweep[fs] = {"ethanol_yield": yields, "temperature_factor": f_Ts}
    results["temperature_sweep"] = {"temperatures_degC": temps.tolist(), "feedstocks": temp_sweep}

    # Feedstock comparison at optimal T
    comparison = {}
    for fs in feedstocks:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 32.0, "moisture_fraction": 0.20})
        comparison[fs] = {k: r[k] for k in ["ethanol_yield","sugar_fraction","pretreatment_eff"]}
    results["feedstock_comparison"] = comparison

    print("EC148 F1b Simulation complete.")
    return results


if __name__ == "__main__":
    run_simulations()
