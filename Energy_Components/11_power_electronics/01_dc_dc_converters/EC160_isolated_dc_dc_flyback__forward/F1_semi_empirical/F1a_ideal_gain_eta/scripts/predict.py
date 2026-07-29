"""EC160 -- Isolated DC-DC Converter (Flyback/Forward) -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import IsolatedDCDCF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = IsolatedDCDCF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in          : float or array [V]  input voltage
            duty_cycle    : float or array [-]  duty cycle (0.05 to d_max)
            p_in          : float or array [W]  input power
        returns:
            v_out         : float or array [V]  output voltage
            p_out_w       : float or array [W]  output power
            p_loss_w      : float or array [W]  losses
            efficiency    : float (fixed)
        """
        v_in = np.asarray(inputs["v_in"], dtype=float)
        D = np.asarray(inputs["duty_cycle"], dtype=float)
        p_in = np.asarray(inputs["p_in"], dtype=float)

        v_out = self._model.output_voltage(v_in, D)
        p_out = self._model.output_power(p_in)
        p_loss = self._model.losses(p_in)

        return {
            "v_out": v_out,
            "p_out_w": p_out,
            "p_loss_w": p_loss,
            "efficiency": np.full_like(p_in, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Isolated DC-DC Converter (Flyback/Forward)",
            "ec_id": "EC160",
            "fidelity": "F1a",
            "description": "V_out = N*D*V_in; P_out = eta*P_in (flyback ideal gain)",
            "inputs": {
                "v_in": {"unit": "V", "range": [100.0, 600.0]},
                "duty_cycle": {"unit": "dimensionless", "range": [0.05, 0.5]},
                "p_in": {"unit": "W", "range": [0.0, 120.0]},
            },
            "outputs": {
                "v_out": {"unit": "V"},
                "p_out_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "N_turns": u["n_turns"]["value"],
                "eta": u["eta"]["value"],
                "P_rated_W": u["p_rated"]["value"],
                "D_max": u["d_max"]["value"],
            },
            "source": "Erickson & Maksimovic (2020)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_in": 400.0, "duty_cycle": 0.3, "p_in": 80.0})
    print(f"V_out={float(r['v_out']):.2f}V  P_out={float(r['p_out_w']):.2f}W  "
          f"P_loss={float(r['p_loss_w']):.2f}W  eta={float(r['efficiency']):.3f}")
