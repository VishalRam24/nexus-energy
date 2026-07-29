"""EC161 -- Dual Active Bridge (DAB) -- F1a Ideal Gain + Efficiency -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import DABF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DABF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v1            : float or array [V]   primary voltage
            v2            : float or array [V]   secondary voltage
            phi_rad       : float or array [rad] phase shift (-pi/2 to +pi/2)
        returns:
            v_out_ideal   : float or array [V]   ideal open-circuit secondary voltage
            p_transfer_w  : float or array [W]   SPS power transfer (+ = forward)
            p_out_w       : float or array [W]   useful output power (after eta)
            p_loss_w      : float or array [W]   losses
            efficiency    : float or array        fixed eta
        """
        v1 = np.asarray(inputs["v1"], dtype=float)
        v2 = np.asarray(inputs["v2"], dtype=float)
        phi = np.asarray(inputs["phi_rad"], dtype=float)

        v_out_ideal = self._model.output_voltage(v1)
        p_transfer = self._model.power_transfer(v1, v2, phi)
        p_out, p_in, p_loss = self._model.efficiency_applied(p_transfer)

        return {
            "v_out_ideal": v_out_ideal,
            "p_transfer_w": p_transfer,
            "p_out_w": p_out,
            "p_loss_w": p_loss,
            "efficiency": np.full_like(phi, self._model.eta),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Dual Active Bridge (DAB) DC-DC Converter",
            "ec_id": "EC161",
            "fidelity": "F1a",
            "description": "V_out=N*V_in; P=N*V1*V2*phi*(pi-|phi|)/(2*pi^2*f*L); bidirectional",
            "inputs": {
                "v1": {"unit": "V", "range": [200.0, 600.0]},
                "v2": {"unit": "V", "range": [200.0, 600.0]},
                "phi_rad": {"unit": "rad", "range": [-1.5708, 1.5708]},
            },
            "outputs": {
                "v_out_ideal": {"unit": "V"},
                "p_transfer_w": {"unit": "W"},
                "p_out_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "N_turns": u["n_turns"]["value"],
                "eta": u["eta"]["value"],
                "f_sw_Hz": u["f_sw"]["value"],
                "L_series_H": u["L_series"]["value"],
                "P_rated_W": u["p_rated"]["value"],
            },
            "source": "De Doncker et al. (1991), IEEE Trans. Ind. Appl.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for phi in [-0.5, 0.0, 0.3, 0.785]:
        r = model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": phi})
        print(f"phi={phi:.3f} rad: P_transfer={float(r['p_transfer_w']):.1f}W  "
              f"P_out={float(r['p_out_w']):.1f}W  eta={float(r['efficiency']):.3f}")
