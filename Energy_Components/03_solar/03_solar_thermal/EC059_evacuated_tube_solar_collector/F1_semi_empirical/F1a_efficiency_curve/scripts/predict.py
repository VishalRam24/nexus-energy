"""EC059 — Evacuated Tube Solar Collector — F1a HWB — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import EvacuatedTubeF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = EvacuatedTubeF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          irradiance  : solar irradiance on collector plane (W/m2)
          T_inlet     : fluid inlet temperature (degC)
          T_ambient   : ambient temperature (degC)
        returns:
          useful_heat_w, efficiency, T_outlet_approx (degC)
        """
        G = np.asarray(inputs["irradiance"], dtype=float)
        T_in = np.asarray(inputs["T_inlet"], dtype=float)
        T_amb = np.asarray(inputs["T_ambient"], dtype=float)
        r = self._model.predict_all(G, T_in, T_amb)
        return {
            "useful_heat_w": r["useful_heat_w"],
            "efficiency": r["efficiency"],
            "T_outlet_approx": r["T_outlet_c"],
        }

    def get_info(self) -> dict:
        return {
            "name": "Evacuated Tube Solar Collector",
            "ec_id": "EC059",
            "fidelity": "F1a",
            "description": "Q_u = A*[F_R*(tau*alpha)*G - F_R*U_L*(T_in-T_amb)]; HWB with low U_L (vacuum)",
            "inputs": {
                "irradiance": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "T_inlet":    {"unit": "degC", "range": [10.0, 120.0]},
                "T_ambient":  {"unit": "degC", "range": [-20.0, 45.0]},
            },
            "outputs": {
                "useful_heat_w":   {"unit": "W"},
                "efficiency":      {"unit": "dimensionless"},
                "T_outlet_approx": {"unit": "degC"},
            },
            "source": "Duffie & Beckman (2013), Ch.6; SRCC OG-100 ETC ratings",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"irradiance": 800.0, "T_inlet": 60.0, "T_ambient": 10.0})
    print(f"G=800 W/m2, T_in=60C, T_amb=10C: "
          f"Q_u={float(r['useful_heat_w']):.1f}W, eta={float(r['efficiency']):.3f}, "
          f"T_out={float(r['T_outlet_approx']):.2f}C")
