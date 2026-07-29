"""EC179 -- Wound Rotor Synchronous Generator -- F1a Efficiency Map -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import WRSyncGenF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = WRSyncGenF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction : float or array [-]  PLR = P_elec/P_rated (0.1 to 1.1)
            power_factor  : float or array [-]  load power factor (optional, default=0.85)
        returns:
            efficiency          : float or array
            p_elec_out_kw       : float or array [kW]
            p_mech_in_kw        : float or array [kW]
            losses_kw           : float or array [kW]
            terminal_current_ka : float or array [kA]
            sync_speed_rpm      : float         (constant, synchronous)
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        pf = inputs.get("power_factor", self._model.pf_rated)
        pf = np.asarray(pf, dtype=float)

        return {
            "efficiency": self._model.efficiency(plr),
            "p_elec_out_kw": self._model.electrical_output(plr),
            "p_mech_in_kw": self._model.mechanical_input(plr),
            "losses_kw": self._model.losses(plr),
            "terminal_current_ka": self._model.terminal_current(plr, pf) / 1000.0,  # -> kA
            "sync_speed_rpm": np.full_like(plr, self._model.synchronous_speed_rpm()),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        m = self._model
        peak_plr = np.sqrt(m.c0 / m.c2)
        return {
            "name": "Wound Rotor Synchronous Generator",
            "ec_id": "EC179",
            "fidelity": "F1a",
            "description": "eta(PLR)=PLR/(PLR+c0+c2*PLR^2); 50MW; 13.8kV; eta_rated=0.97",
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.1, 1.1]},
                "power_factor": {"unit": "dimensionless", "range": [0.7, 1.0], "optional": True},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_elec_out_kw": {"unit": "kW"},
                "p_mech_in_kw": {"unit": "kW"},
                "losses_kw": {"unit": "kW"},
                "terminal_current_ka": {"unit": "kA"},
                "sync_speed_rpm": {"unit": "rpm"},
            },
            "params": {
                "P_rated_kW": u["rated_power"]["value"],
                "eta_rated": u["eta_rated"]["value"],
                "V_terminal_V": u["v_terminal"]["value"],
                "omega_rated_rpm": u["omega_rated"]["value"],
                "T_rated_Nm": round(float(m.T_rated), 1),
                "sync_speed_rpm": m.synchronous_speed_rpm(),
                "peak_eta_PLR": round(float(peak_plr), 4),
            },
            "source": "Boldea (2015). Synchronous Generators. CRC Press.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    m = model._model
    print(f"Sync speed = {m.synchronous_speed_rpm():.0f} rpm")
    for plr in [0.25, 0.5, 0.75, 1.0, 1.1]:
        r = model.predict({"load_fraction": plr, "power_factor": 0.85})
        print(f"PLR={plr:.2f}: eta={float(r['efficiency'])*100:.3f}%  "
              f"P_elec={float(r['p_elec_out_kw']):.0f}kW  "
              f"P_mech={float(r['p_mech_in_kw']):.0f}kW  "
              f"I={float(r['terminal_current_ka']):.3f}kA")
