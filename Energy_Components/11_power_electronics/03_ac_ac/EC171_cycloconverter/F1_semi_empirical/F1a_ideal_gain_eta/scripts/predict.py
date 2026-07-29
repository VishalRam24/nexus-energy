"""EC171 -- Cycloconverter -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import CycloconverterF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CycloconverterF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in_rms  : float or array [V]   input AC voltage (line-to-line RMS)
            alpha_rad : float or array [rad] firing angle (0 to pi/2)
            p_out     : float or array [W]   output power
        returns:
            v_out_rms    : float or array [V]   output fundamental RMS voltage
            power_factor : float or array [-]   approx input PF = cos(alpha)
            p_in_w       : float or array [W]   input power
            p_loss_w     : float or array [W]   losses
            efficiency   : float or array       fixed eta
        """
        v_in = np.asarray(inputs["v_in_rms"], dtype=float)
        alpha = np.asarray(inputs["alpha_rad"], dtype=float)
        p_out = np.asarray(inputs["p_out"], dtype=float)

        v_out = self._model.output_voltage(v_in, alpha)
        pf = self._model.power_factor(alpha)
        p_in = self._model.input_power(p_out)
        p_loss = self._model.losses(p_in)

        return {
            "v_out_rms": v_out,
            "power_factor": pf,
            "p_in_w": p_in,
            "p_loss_w": p_loss,
            "efficiency": np.full_like(p_out, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Cycloconverter",
            "ec_id": "EC171",
            "fidelity": "F1a",
            "description": "V_out=V_in*cos(alpha); f_out<f_in/3; PF=cos(alpha); eta=0.94",
            "inputs": {
                "v_in_rms": {"unit": "V", "range": [1000.0, 15000.0]},
                "alpha_rad": {"unit": "rad", "range": [0.0, 1.5708]},
                "p_out": {"unit": "W", "range": [0.0, 1100000.0]},
            },
            "outputs": {
                "v_out_rms": {"unit": "V"},
                "power_factor": {"unit": "dimensionless"},
                "p_in_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "eta": u["eta"]["value"],
                "V_in_nominal_V": u["v_in_rms"]["value"],
                "f_in_Hz": u["f_in"]["value"],
                "f_out_max_Hz": u["f_out_max"]["value"],
                "P_rated_W": u["p_rated"]["value"],
            },
            "source": "Sen (1997); Mohan et al. (2003)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for alpha_deg in [0, 30, 60, 90]:
        alpha = np.radians(alpha_deg)
        r = model.predict({"v_in_rms": 6000.0, "alpha_rad": alpha, "p_out": 500000.0})
        print(f"alpha={alpha_deg:2d}deg: V_out={float(r['v_out_rms']):.0f}V  "
              f"PF={float(r['power_factor']):.3f}  P_in={float(r['p_in_w'])/1000:.1f}kW")
