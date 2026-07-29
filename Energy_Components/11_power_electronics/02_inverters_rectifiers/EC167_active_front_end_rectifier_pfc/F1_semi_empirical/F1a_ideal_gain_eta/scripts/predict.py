"""EC167 -- Active Front End Rectifier PFC -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import AFEPFCRectifierF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AFEPFCRectifierF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_ll      : float or array [V]  AC line-to-line RMS voltage
            v_dc_set  : float or array [V]  DC output voltage setpoint
            p_out     : float or array [W]  DC output power
        returns:
            v_dc          : float or array [V]
            i_dc          : float or array [A]
            p_in_w        : float or array [W]
            p_loss_w      : float or array [W]
            i_ac_rms      : float or array [A]  per-phase AC current
            power_factor  : float              (fixed ~ 1.0)
            efficiency    : float or array
        """
        v_ll = np.asarray(inputs["v_ll"], dtype=float)
        v_dc_set = np.asarray(inputs["v_dc_set"], dtype=float)
        p_out = np.asarray(inputs["p_out"], dtype=float)

        p_in = self._model.input_power(p_out)
        v_dc = self._model.output_voltage(v_dc_set)
        i_dc = self._model.dc_current(v_dc_set, p_out)
        i_ac = self._model.ac_current_rms(v_ll, p_in)
        p_loss = self._model.losses(p_in)

        return {
            "v_dc": v_dc,
            "i_dc": i_dc,
            "p_in_w": p_in,
            "p_loss_w": p_loss,
            "i_ac_rms": i_ac,
            "power_factor": np.full_like(p_out, self._model.pf),
            "efficiency": np.full_like(p_out, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Active Front End Rectifier with PFC",
            "ec_id": "EC167",
            "fidelity": "F1a",
            "description": "V_dc=setpoint (boost); PF=1; I_ac=P_in/(sqrt(3)*V_LL); eta=0.97",
            "inputs": {
                "v_ll": {"unit": "V", "range": [300.0, 500.0]},
                "v_dc_set": {"unit": "V", "range": [600.0, 800.0]},
                "p_out": {"unit": "W", "range": [0.0, 55000.0]},
            },
            "outputs": {
                "v_dc": {"unit": "V"},
                "i_dc": {"unit": "A"},
                "p_in_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "i_ac_rms": {"unit": "A"},
                "power_factor": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "eta": u["eta"]["value"],
                "PF_nominal": u["pf_nominal"]["value"],
                "THD_i": u["thd_i"]["value"],
                "V_dc_set_V": u["v_dc_set"]["value"],
                "P_rated_W": u["p_rated"]["value"],
            },
            "source": "Mohan et al. (2003); IEC 61000-3-2",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for p_out in [10000.0, 30000.0, 50000.0]:
        r = model.predict({"v_ll": 400.0, "v_dc_set": 700.0, "p_out": p_out})
        print(f"P_out={p_out/1000:.0f}kW: V_dc={float(r['v_dc']):.0f}V  "
              f"I_dc={float(r['i_dc']):.1f}A  I_ac={float(r['i_ac_rms']):.1f}A  "
              f"PF={float(r['power_factor']):.2f}")
