"""EC162 -- Resonant LLC Converter -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import LLCF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LLCF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in   : float or array [V]   input voltage
            fn     : float or array [-]   normalized frequency f_sw/f_res (0.5 to 2.0)
            p_in   : float or array [W]   input power
        returns:
            v_out  : float or array [V]   output voltage
            gain_M : float or array [-]   voltage gain M(fn)
            p_out_w: float or array [W]   output power
            p_loss_w: float or array [W]  losses
            efficiency: float or array    fixed eta
        """
        v_in = np.asarray(inputs["v_in"], dtype=float)
        fn = np.asarray(inputs["fn"], dtype=float)
        p_in = np.asarray(inputs["p_in"], dtype=float)

        M = self._model.gain_M(fn)
        v_out = self._model.output_voltage(v_in, fn)
        p_out = self._model.output_power(p_in)
        p_loss = self._model.losses(p_in)

        return {
            "v_out": v_out,
            "gain_M": M,
            "p_out_w": p_out,
            "p_loss_w": p_loss,
            "efficiency": np.full_like(p_in, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Resonant LLC Converter",
            "ec_id": "EC162",
            "fidelity": "F1a",
            "description": "V_out=N*V_in*M(fn); M~1 at resonance; ZVS enables eta=0.97",
            "inputs": {
                "v_in": {"unit": "V", "range": [200.0, 600.0]},
                "fn": {"unit": "dimensionless", "range": [0.5, 2.0], "note": "f_sw/f_res"},
                "p_in": {"unit": "W", "range": [0.0, 120.0]},
            },
            "outputs": {
                "v_out": {"unit": "V"},
                "gain_M": {"unit": "dimensionless"},
                "p_out_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "N_turns": u["n_turns"]["value"],
                "eta": u["eta"]["value"],
                "f_res_Hz": u["f_res"]["value"],
                "P_rated_W": u["p_rated"]["value"],
            },
            "source": "Yang et al. (2002), APEC",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for fn in [0.7, 0.9, 1.0, 1.1, 1.3]:
        r = model.predict({"v_in": 400.0, "fn": fn, "p_in": 80.0})
        print(f"fn={fn:.1f}: M={float(r['gain_M']):.4f}  V_out={float(r['v_out']):.2f}V  "
              f"P_out={float(r['p_out_w']):.2f}W")
