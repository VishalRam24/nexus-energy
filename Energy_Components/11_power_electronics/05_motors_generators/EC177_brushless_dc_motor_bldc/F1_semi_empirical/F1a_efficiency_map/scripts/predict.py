"""EC177 -- BLDC Motor -- F1a Efficiency Map -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import BLDCF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BLDCF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction : float or array [-]  PLR = P_out / P_rated (0.05 to 1.2)
        returns:
            efficiency       : float or array
            torque_nm        : float or array [Nm]
            current_a        : float or array [A]
            output_power_kw  : float or array [kW]
            input_power_kw   : float or array [kW]
            losses_kw        : float or array [kW]
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)

        return {
            "efficiency": self._model.efficiency(plr),
            "torque_nm": self._model.torque(plr),
            "current_a": self._model.current(plr),
            "output_power_kw": self._model.output_power(plr),
            "input_power_kw": self._model.input_power(plr),
            "losses_kw": self._model.losses(plr),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        m = self._model
        peak_plr = np.sqrt(m.c0 / m.c2)
        return {
            "name": "Brushless DC Motor (BLDC)",
            "ec_id": "EC177",
            "fidelity": "F1a",
            "description": "eta(PLR)=PLR/(PLR+c0+c2*PLR^2); Kt=0.1Nm/A; eta_peak=0.92",
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.05, 1.2]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "torque_nm": {"unit": "Nm"},
                "current_a": {"unit": "A"},
                "output_power_kw": {"unit": "kW"},
                "input_power_kw": {"unit": "kW"},
                "losses_kw": {"unit": "kW"},
            },
            "params": {
                "P_rated_kW": u["rated_power"]["value"],
                "eta_rated": u["eta_rated"]["value"],
                "omega_rated_rpm": u["omega_rated"]["value"],
                "Kt_Nm_per_A": u["Kt"]["value"],
                "T_rated_Nm": round(float(m.T_rated), 4),
                "peak_eta_PLR": round(float(peak_plr), 4),
            },
            "source": "Hanselman (2006); Gieras (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    m = model._model
    print(f"T_rated={m.T_rated:.3f} Nm  peak_eta_PLR={np.sqrt(m.c0/m.c2):.3f}")
    for plr in [0.25, 0.5, 0.75, 1.0, 1.1]:
        r = model.predict({"load_fraction": plr})
        print(f"PLR={plr:.2f}: eta={float(r['efficiency'])*100:.2f}%  "
              f"T={float(r['torque_nm']):.3f}Nm  I={float(r['current_a']):.2f}A  "
              f"P_in={float(r['input_power_kw']):.4f}kW")
