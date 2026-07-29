"""EC165 -- Multilevel Inverter (3-Level NPC) -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import MultilevelInverterF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MultilevelInverterF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_dc   : float or array [V]  DC bus voltage
            ma     : float or array [-]  modulation index (0 to 1.15)
            p_in   : float or array [W]  input DC power
        returns:
            v_ac_phase_rms : float or array [V]
            v_ac_line_rms  : float or array [V]
            thd_approx     : float or array [-]  fractional THD
            p_out_w        : float or array [W]
            p_loss_w       : float or array [W]
            efficiency     : float or array
        """
        v_dc = np.asarray(inputs["v_dc"], dtype=float)
        ma = np.asarray(inputs["ma"], dtype=float)
        p_in = np.asarray(inputs["p_in"], dtype=float)

        return {
            "v_ac_phase_rms": self._model.v_ac_rms_phase(v_dc, ma),
            "v_ac_line_rms": self._model.v_ac_rms_line(v_dc, ma),
            "thd_approx": self._model.thd_approx(ma),
            "p_out_w": self._model.output_power(p_in),
            "p_loss_w": self._model.losses(p_in),
            "efficiency": np.full_like(p_in, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Multilevel Inverter (3-Level NPC)",
            "ec_id": "EC165",
            "fidelity": "F1a",
            "description": "V_ac = ma*V_dc/2 (peak); 3-level NPC; THD_approx = 0.02/ma; eta=0.97",
            "inputs": {
                "v_dc": {"unit": "V", "range": [400.0, 1200.0]},
                "ma": {"unit": "dimensionless", "range": [0.0, 1.15]},
                "p_in": {"unit": "W", "range": [0.0, 120000.0]},
            },
            "outputs": {
                "v_ac_phase_rms": {"unit": "V"},
                "v_ac_line_rms": {"unit": "V"},
                "thd_approx": {"unit": "dimensionless"},
                "p_out_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "eta": u["eta"]["value"],
                "n_levels": u["n_levels"]["value"],
                "V_dc_nominal_V": u["v_dc_nominal"]["value"],
                "P_rated_W": u["p_rated"]["value"],
            },
            "source": "Rodriguez et al. (2002), IEEE Trans. Ind. Electron.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for ma in [0.5, 0.8, 1.0]:
        r = model.predict({"v_dc": 800.0, "ma": ma, "p_in": 80000.0})
        print(f"ma={ma:.1f}: V_LL={float(r['v_ac_line_rms']):.1f}V  "
              f"THD={float(r['thd_approx'])*100:.1f}%  P_out={float(r['p_out_w'])/1000:.1f}kW")
