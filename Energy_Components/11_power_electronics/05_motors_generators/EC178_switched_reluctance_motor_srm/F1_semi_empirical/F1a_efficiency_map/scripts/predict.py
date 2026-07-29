"""EC178 -- Switched Reluctance Motor (SRM) -- F1a Efficiency Map -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import SRMF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SRMF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction : float or array [-]  PLR = P_out/P_rated (0.05 to 1.2)
        returns:
            efficiency        : float or array
            torque_avg_nm     : float or array [Nm]
            torque_ripple_nm  : float or array [Nm]  peak-to-peak estimate
            output_power_kw   : float or array [kW]
            input_power_kw    : float or array [kW]
            losses_kw         : float or array [kW]
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)

        return {
            "efficiency": self._model.efficiency(plr),
            "torque_avg_nm": self._model.torque_avg(plr),
            "torque_ripple_nm": self._model.torque_ripple(plr),
            "output_power_kw": self._model.output_power(plr),
            "input_power_kw": self._model.input_power(plr),
            "losses_kw": self._model.losses(plr),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        m = self._model
        peak_plr = np.sqrt(m.c0 / m.c2)
        return {
            "name": "Switched Reluctance Motor (SRM)",
            "ec_id": "EC178",
            "fidelity": "F1a",
            "description": "eta(PLR)=PLR/(PLR+c0+c2*PLR^2); eta_peak=0.88; torque_ripple~15%",
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.05, 1.2]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "torque_avg_nm": {"unit": "Nm"},
                "torque_ripple_nm": {"unit": "Nm", "note": "peak-to-peak estimate"},
                "output_power_kw": {"unit": "kW"},
                "input_power_kw": {"unit": "kW"},
                "losses_kw": {"unit": "kW"},
            },
            "params": {
                "P_rated_kW": u["rated_power"]["value"],
                "eta_rated": u["eta_rated"]["value"],
                "omega_rated_rpm": u["omega_rated"]["value"],
                "T_rated_Nm": round(float(m.T_rated), 4),
                "torque_ripple_factor": u["torque_ripple_factor"]["value"],
                "poles": f"{u['poles_stator']['value']}/{u['poles_rotor']['value']}",
                "peak_eta_PLR": round(float(peak_plr), 4),
            },
            "source": "Krishnan (2001). CRC Press.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    m = model._model
    print(f"T_rated={m.T_rated:.2f} Nm  peak_eta_PLR={np.sqrt(m.c0/m.c2):.3f}")
    for plr in [0.25, 0.5, 0.75, 1.0, 1.1]:
        r = model.predict({"load_fraction": plr})
        print(f"PLR={plr:.2f}: eta={float(r['efficiency'])*100:.2f}%  "
              f"T_avg={float(r['torque_avg_nm']):.2f}Nm  "
              f"T_ripple={float(r['torque_ripple_nm']):.2f}Nm")
