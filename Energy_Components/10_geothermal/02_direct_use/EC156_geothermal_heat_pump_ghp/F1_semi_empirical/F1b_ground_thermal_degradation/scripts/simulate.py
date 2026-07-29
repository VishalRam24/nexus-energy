"""EC156 -- GHP -- F1b Ground Thermal Degradation -- Simulation Scenarios"""
import numpy as np
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    hours = np.linspace(0, 4000, 40)
    T_sinks = np.linspace(30, 60, 15)
    tds_vals = np.linspace(100, 2000, 15)

    # Time series: COP degradation over heating season
    time_data = {k: [] for k in ["cop_heating", "cop_effective", "T_source_effective",
                                   "ground_dT", "fouling_factor"]}
    for t in hours:
        r = model.predict({"T_sink_c": 35.0, "operation_hours": float(t),
                           "TDS_ppm": 500.0, "PLR": 1.0})
        for k in time_data:
            time_data[k].append(r[k])
    results["time_series"] = {"hours": hours.tolist(), "T_sink=35": time_data}

    # T_sink sweep
    tsink_data = {"cop_heating": [], "cop_advantage": []}
    for T in T_sinks:
        r = model.predict({"T_sink_c": float(T), "operation_hours": 0.0})
        tsink_data["cop_heating"].append(r["cop_heating"])
        tsink_data["cop_advantage"].append(r["cop_advantage_over_ashp"])
    results["T_sink_sweep"] = {"T_sinks_degC": T_sinks.tolist(), "data": tsink_data}

    # TDS fouling sweep
    tds_data = {"cop_effective": [], "fouling_factor": []}
    for tds in tds_vals:
        r = model.predict({"T_sink_c": 35.0, "operation_hours": 2000.0, "TDS_ppm": float(tds)})
        tds_data["cop_effective"].append(r["cop_effective"])
        tds_data["fouling_factor"].append(r["fouling_factor"])
    results["TDS_sweep"] = {"TDS_ppm": tds_vals.tolist(), "at_2000h": tds_data}

    print("EC156 F1b Simulation complete.")
    return results


if __name__ == "__main__":
    run_simulations()
