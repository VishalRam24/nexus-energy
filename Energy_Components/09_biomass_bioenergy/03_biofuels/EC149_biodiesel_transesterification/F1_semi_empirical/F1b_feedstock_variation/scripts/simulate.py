"""EC149 -- Biodiesel Transesterification -- F1b -- Simulation Scenarios"""
import numpy as np
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    feedstocks = ["soybean_oil", "palm_oil", "rapeseed_oil", "waste_cooking_oil", "jatropha"]
    temps = np.linspace(40, 80, 20)
    ffa_vals = np.linspace(0, 10, 20)

    # Temperature sweep
    temp_sweep = {}
    for fs in feedstocks:
        yields, f_Ts = [], []
        for T in temps:
            r = model.predict({"feedstock_type": fs, "temperature_degC": float(T)})
            yields.append(r["biodiesel_yield"])
            f_Ts.append(r["temperature_factor"])
        temp_sweep[fs] = {"biodiesel_yield": yields, "temp_factor": f_Ts}
    results["temperature_sweep"] = {"temperatures_degC": temps.tolist(), "feedstocks": temp_sweep}

    # FFA sweep (using soybean as baseline)
    ffa_sweep = {"yield": [], "ffa_factor": []}
    for ffa in ffa_vals:
        r = model.predict({"feedstock_type": "soybean_oil", "temperature_degC": 60.0, "ffa_pct": float(ffa)})
        ffa_sweep["yield"].append(r["biodiesel_yield"])
        ffa_sweep["ffa_factor"].append(r["ffa_penalty_factor"])
    results["ffa_sweep"] = {"ffa_pct": ffa_vals.tolist(), "soybean_60degC": ffa_sweep}

    print("EC149 F1b Simulation complete.")
    return results


if __name__ == "__main__":
    run_simulations()
