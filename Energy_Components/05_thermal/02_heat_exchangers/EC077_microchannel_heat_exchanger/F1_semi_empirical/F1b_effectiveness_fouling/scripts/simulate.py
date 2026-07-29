"""EC077 — Microchannel HX — F1b — Simulation Scenarios"""

import json, numpy as np
from pathlib import Path
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    results = {}

    # Scenario 1: Fouling resistance sweep
    Rf = np.linspace(0.0, 0.003, 60)
    r1 = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                         "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                         "Rf_hot": Rf, "Rf_cold": Rf})
    results["fouling_sweep"] = {
        "Rf_m2KW": Rf.tolist(),
        "Q_kw": r1["Q_kw"].tolist(),
        "effectiveness": r1["effectiveness"].tolist(),
        "effectiveness_reduction": r1["effectiveness_reduction"].tolist(),
    }

    # Scenario 2: Part-load sweep
    PLR = np.linspace(0.5, 1.0, 50)
    r2 = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                         "m_dot_hot": 0.5, "m_dot_cold": 0.3, "PLR": PLR})
    results["plr_sweep"] = {
        "PLR": PLR.tolist(),
        "Q_kw": r2["Q_kw"].tolist(),
        "effectiveness": r2["effectiveness"].tolist(),
        "F_lmtd": r2["F_lmtd"].tolist(),
    }

    # Scenario 3: Hot-side flow rate sweep
    m_hot = np.linspace(0.05, 2.0, 60)
    r3 = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                         "m_dot_hot": m_hot, "m_dot_cold": 0.3})
    results["flow_sweep"] = {
        "m_dot_hot_kgs": m_hot.tolist(),
        "Q_kw": r3["Q_kw"].tolist(),
        "effectiveness": r3["effectiveness"].tolist(),
    }

    return results


if __name__ == "__main__":
    results = run_simulations()
    print("EC077 F1b Microchannel HX simulation complete.")
    Rf = np.array(results["fouling_sweep"]["Rf_m2KW"])
    Q  = np.array(results["fouling_sweep"]["Q_kw"])
    print(f"  Clean Q: {Q[0]:.2f} kW; fouled (Rf=0.003): {Q[-1]:.2f} kW")
