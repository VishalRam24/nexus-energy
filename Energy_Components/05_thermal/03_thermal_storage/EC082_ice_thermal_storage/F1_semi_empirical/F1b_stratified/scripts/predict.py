"""EC082 — Ice TES — F1b Stratified — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import IceTESF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = IceTESF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            q_charge_W    : charging power [W]
            q_discharge_W : discharge power [W]
            t_ambient     : ambient temperature [degC]
            duration_s    : simulation duration [s]
            f_initial     : initial ice fractions per node (array N or scalar)
            dt            : time step [s] (optional, default 60)
        """
        q_ch  = float(inputs.get("q_charge_W",    0.0))
        q_dis = float(inputs.get("q_discharge_W", 0.0))
        T_amb = float(inputs.get("t_ambient", self._model.T_amb_default))
        dur   = float(inputs.get("duration_s", 3600.0))
        f_ini = inputs.get("f_initial", None)
        dt    = float(inputs.get("dt", 60.0))
        return self._model.simulate(q_ch, q_dis, T_amb, dur, f_ini, dt)

    def get_info(self) -> dict:
        return {
            "name": "Ice Thermal Storage (Stratified)",
            "ec_id": "EC082",
            "fidelity": "F1b",
            "description": (
                "Multi-node stratified ice TES: ice forms bottom-up during charging, "
                "melts top-down during discharge. Tracks per-node ice fractions with "
                "heat loss and axial conduction."
            ),
            "inputs": {
                "q_charge_W":    {"unit": "W",    "range": [0.0, 150000.0]},
                "q_discharge_W": {"unit": "W",    "range": [0.0, 250000.0]},
                "t_ambient":     {"unit": "degC", "range": [-10.0, 40.0]},
                "duration_s":    {"unit": "s",    "default": 3600.0},
                "f_initial":     {"unit": "-",    "default": 0.0, "shape": "N_nodes"},
                "dt":            {"unit": "s",    "default": 60.0},
            },
            "outputs": {
                "f_nodes":               {"unit": "-",  "shape": "N_nodes"},
                "soc":                   {"unit": "-"},
                "Q_actual_charge_kw":    {"unit": "kW"},
                "Q_actual_discharge_kw": {"unit": "kW"},
                "Q_loss_kw":             {"unit": "kW"},
                "stratification_index":  {"unit": "-"},
                "f_history":             {"unit": "-", "shape": "(n_steps+1, N_nodes)"},
            },
            "source": "ASHRAE (2020) ch.51; MacPhee & Dincer (2009); Jekel et al. (1993)",
        }


if __name__ == "__main__":
    model = ComponentModel()

    # Charge for 4 hours
    r = model.predict({"q_charge_W": 80000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 4 * 3600,
                       "f_initial": 0.0, "dt": 60.0})
    print("=== Charging 4h at 80 kW ===")
    print(f"Final SOC: {r['soc']:.3f}")
    print(f"Node fracs (bottom→top): {np.round(r['f_nodes'], 3)}")
    print(f"Stratification index: {r['stratification_index']:.3f}")

    # Discharge 2 hours
    r2 = model.predict({"q_charge_W": 0.0, "q_discharge_W": 100000.0,
                        "t_ambient": 20.0, "duration_s": 2 * 3600,
                        "f_initial": r["f_nodes"], "dt": 60.0})
    print("\n=== Discharge 2h at 100 kW ===")
    print(f"Final SOC: {r2['soc']:.3f}")
    print(f"Node fracs (bottom→top): {np.round(r2['f_nodes'], 3)}")
    print(f"Stratification index: {r2['stratification_index']:.3f}")
