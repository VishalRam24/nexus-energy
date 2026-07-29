"""EC170 -- Solid State Transformer (SST) -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import SSTF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SSTF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in   : float or array [V]  primary (MV) voltage
            p_in   : float or array [W]  input power (>0 forward, <0 reverse)
        returns:
            v_out      : float or array [V]   secondary (LV) ideal voltage
            p_out_w    : float or array [W]   output power
            p_loss_w   : float or array [W]   losses (always >= 0)
            efficiency : float or array       fixed eta
        """
        v_in = np.asarray(inputs["v_in"], dtype=float)
        p_in = np.asarray(inputs["p_in"], dtype=float)

        v_out = self._model.output_voltage(v_in)
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
            "name": "Solid State Transformer (SST)",
            "ec_id": "EC170",
            "fidelity": "F1a",
            "description": "V_out=N*V_in; bidirectional; MV to LV AC-AC via HF DC-DC; eta=0.96",
            "inputs": {
                "v_in": {"unit": "V", "range": [5000.0, 15000.0]},
                "p_in": {"unit": "W", "range": [-110000.0, 110000.0]},
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
                "V_in_nominal_V": u["v_in_nominal"]["value"],
                "V_out_nominal_V": u["v_out_nominal"]["value"],
                "P_rated_W": u["p_rated"]["value"],
                "bidirectional": True,
            },
            "source": "Huang et al. (2011), IEEE Trans. Ind. Electron.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for p in [-80000.0, 0.0, 50000.0, 100000.0]:
        r = model.predict({"v_in": 10000.0, "p_in": p})
        print(f"P_in={p/1000:.0f}kW: V_out={float(r['v_out']):.0f}V  "
              f"P_out={float(r['p_out_w'])/1000:.2f}kW  P_loss={float(r['p_loss_w'])/1000:.2f}kW")
