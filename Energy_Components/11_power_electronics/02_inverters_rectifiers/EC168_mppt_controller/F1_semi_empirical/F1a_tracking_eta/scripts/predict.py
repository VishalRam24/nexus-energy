"""EC168 — MPPT Controller — F1a Tracking Efficiency — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MPPTF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MPPTF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            irradiance (W/m2):    Solar irradiance [0–1200]
            p_mpp_input (W):      Available MPP power from PV array

        returns:
            p_output (W)
            tracking_efficiency (-)
            power_loss (W)
        """
        G = np.asarray(inputs["irradiance"], dtype=float)
        p_in = np.asarray(inputs["p_mpp_input"], dtype=float)
        return {
            "p_output": self._model.output_power(G, p_in),
            "tracking_efficiency": self._model.tracking_efficiency(G),
            "power_loss": self._model.power_loss(G, p_in),
        }

    def get_info(self) -> dict:
        return {
            "name": "MPPT Controller",
            "ec_id": "EC168",
            "fidelity": "F1a",
            "description": (
                "eta_mppt = eta_max * (1 - exp(-k * G / G_ref)), "
                "P_out = P_mpp * eta_mppt"
            ),
            "inputs": {
                "irradiance": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "p_mpp_input": {"unit": "W", "range": [0.0, 12000.0]},
            },
            "outputs": {
                "p_output": {"unit": "W"},
                "tracking_efficiency": {"unit": "dimensionless"},
                "power_loss": {"unit": "W"},
            },
            "source": "Hohm & Ropp (2003), Progress in Photovoltaics, 11, 47-62",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for G in [50, 200, 500, 1000]:
        p_in = G * 10.0  # 10 W per W/m2 for illustrative 10kW at STC
        r = model.predict({"irradiance": float(G), "p_mpp_input": p_in})
        print(f"G={G:4d} W/m2: eta={float(r['tracking_efficiency']):.4f}, "
              f"P_out={float(r['p_output']):.1f}W, loss={float(r['power_loss']):.1f}W")
