"""EC169 -- Variable Frequency Drive (VFD) -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import VFDF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = VFDF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            f_out  : float or array [Hz]  output frequency (0 to 120 Hz)
            p_out  : float or array [W]   output mechanical power
        returns:
            v_out_ll   : float or array [V]   output line-to-line voltage
            v_hz_ratio : float or array [V/Hz] V/Hz ratio
            p_in_w     : float or array [W]   input power
            p_loss_w   : float or array [W]   losses
            efficiency : float or array       fixed eta
        """
        f_out = np.asarray(inputs["f_out"], dtype=float)
        p_out = np.asarray(inputs["p_out"], dtype=float)

        v_out = self._model.output_voltage(f_out)
        p_in = self._model.input_power(p_out)
        p_loss = self._model.losses(p_in)
        f_safe = np.where(f_out > 0, f_out, 1.0)
        v_hz = np.where(f_out > 0, v_out / f_safe, 0.0)

        return {
            "v_out_ll": v_out,
            "v_hz_ratio": v_hz,
            "p_in_w": p_in,
            "p_loss_w": p_loss,
            "efficiency": np.full_like(p_out, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Variable Frequency Drive (VFD)",
            "ec_id": "EC169",
            "fidelity": "F1a",
            "description": "V_out=V_rated*f_out/f_rated (V/Hz); field weakening above f_rated; eta=0.96",
            "inputs": {
                "f_out": {"unit": "Hz", "range": [0.0, 120.0]},
                "p_out": {"unit": "W", "range": [0.0, 18000.0]},
            },
            "outputs": {
                "v_out_ll": {"unit": "V"},
                "v_hz_ratio": {"unit": "V/Hz"},
                "p_in_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "eta": u["eta"]["value"],
                "V_rated_V": u["v_rated"]["value"],
                "f_rated_Hz": u["f_rated"]["value"],
                "P_rated_W": u["p_rated"]["value"],
            },
            "source": "Mohan et al. (2003); IEC 61800-9-2",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for f in [0.0, 25.0, 50.0, 75.0, 100.0]:
        r = model.predict({"f_out": f, "p_out": 10000.0})
        print(f"f={f:5.1f}Hz: V_out={float(r['v_out_ll']):.1f}V  "
              f"V/Hz={float(r['v_hz_ratio']):.2f}  P_in={float(r['p_in_w'])/1000:.2f}kW")
