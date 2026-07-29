"""EC166 -- Diode Bridge Rectifier (3-Phase) -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import DiodeBridgeRectifierF1a, _K_3PHASE


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DiodeBridgeRectifierF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_ll   : float or array [V]  3-phase line-to-line RMS AC voltage
            p_out  : float or array [W]  desired DC output power
        returns:
            v_dc        : float or array [V]   ideal DC output voltage
            i_dc        : float or array [A]   DC output current
            p_in_w      : float or array [W]   AC input power
            p_loss_w    : float or array [W]   losses
            efficiency  : float or array       fixed eta
        """
        v_ll = np.asarray(inputs["v_ll"], dtype=float)
        p_out = np.asarray(inputs["p_out"], dtype=float)

        v_dc = self._model.v_dc_ideal(v_ll)
        p_in = self._model.input_power(p_out)
        i_dc = self._model.dc_current(v_ll, p_out)
        p_loss = self._model.losses(p_in)

        return {
            "v_dc": v_dc,
            "i_dc": i_dc,
            "p_in_w": p_in,
            "p_loss_w": p_loss,
            "efficiency": np.full_like(p_out, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Diode Bridge Rectifier (3-Phase Uncontrolled)",
            "ec_id": "EC166",
            "fidelity": "F1a",
            "description": f"V_dc = {_K_3PHASE:.4f} * V_LL (3*sqrt(2)/pi); uncontrolled; eta=0.96",
            "inputs": {
                "v_ll": {"unit": "V", "range": [100.0, 700.0]},
                "p_out": {"unit": "W", "range": [0.0, 60000.0]},
            },
            "outputs": {
                "v_dc": {"unit": "V"},
                "i_dc": {"unit": "A"},
                "p_in_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "eta": u["eta"]["value"],
                "V_dc_factor": u["v_dc_ideal_factor"]["value"],
                "n_phases": u["n_phases"]["value"],
                "P_rated_W": u["p_rated"]["value"],
            },
            "source": "Mohan, Undeland & Robbins (2003)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for v_ll in [200.0, 400.0, 600.0]:
        r = model.predict({"v_ll": v_ll, "p_out": 30000.0})
        print(f"V_LL={v_ll:.0f}V: V_dc={float(r['v_dc']):.1f}V  "
              f"I_dc={float(r['i_dc']):.1f}A  P_in={float(r['p_in_w'])/1000:.1f}kW")
