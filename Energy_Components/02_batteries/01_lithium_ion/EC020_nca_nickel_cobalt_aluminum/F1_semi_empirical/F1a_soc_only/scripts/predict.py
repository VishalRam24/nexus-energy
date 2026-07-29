"""EC020 — NCA Battery — F1a SOC-Only — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import NCABatteryF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = NCABatteryF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict NCA battery terminal voltage.

        Args:
            inputs: dict with keys:
                - soc     (0-1):  State of charge, scalar or array
                - current (A):    Positive = discharge, negative = charge

        Returns:
            dict with keys:
                - voltage   (V)
                - ocv       (V)
                - power     (W)
                - dsoc_dt   (1/s)
        """
        soc = np.asarray(inputs["soc"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)

        return {
            "voltage":  self._model.terminal_voltage(soc, current),
            "ocv":      self._model.ocv(soc),
            "power":    self._model.power(soc, current),
            "dsoc_dt":  self._model.soc_derivative(soc, current),
        }

    def get_info(self) -> dict:
        return {
            "name": "NCA Battery (Nickel-Cobalt-Aluminum)",
            "ec_id": "EC020",
            "fidelity": "F1a",
            "description": "V = OCV(SOC) - I*R_int; OCV is 5th-order polynomial for Panasonic NCR18650B",
            "inputs": {
                "soc":     {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-15.0, 15.0], "default": 0.0,
                            "note": "positive=discharge, negative=charge"},
            },
            "outputs": {
                "voltage":  {"unit": "V"},
                "ocv":      {"unit": "V"},
                "power":    {"unit": "W"},
                "dsoc_dt":  {"unit": "1/s"},
            },
            "source": "Tremblay & Dessaint (2009); Panasonic NCR18650B datasheet",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for soc in [1.0, 0.8, 0.5, 0.2, 0.0]:
        r = model.predict({"soc": soc, "current": 1.0})
        print(f"SOC={soc:.1f}: V={float(r['voltage']):.3f} V, OCV={float(r['ocv']):.3f} V")
