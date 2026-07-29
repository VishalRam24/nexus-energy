"""EC219 — Piezoelectric Energy Harvester — F1a — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PiezoF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PiezoF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            acceleration : float or array — base excitation [m/s^2]
            frequency    : float or array — excitation frequency [Hz]
        returns:
            power_w               : W
            power_uw              : uW
            voltage_v             : V
            frequency_ratio       : f/f_n
            at_resonance_power_w  : W (power if at resonance at same acceleration)
        """
        a = np.asarray(inputs["acceleration"], dtype=float)
        f = np.asarray(inputs["frequency"], dtype=float)
        return self._model.compute(a, f)

    def get_info(self) -> dict:
        return {
            "name": "Piezoelectric Energy Harvester",
            "ec_id": "EC219",
            "fidelity": "F1a",
            "description": "P = m^2*a^2/(4*c_mech)*H^2(omega); P~a^2; peak at resonance",
            "inputs": {
                "acceleration": {"unit": "m/s^2", "range": [0.1, 50.0]},
                "frequency":    {"unit": "Hz",    "range": [10.0, 1000.0]},
            },
            "outputs": {
                "power_w":              {"unit": "W"},
                "power_uw":             {"unit": "uW"},
                "voltage_v":            {"unit": "V"},
                "frequency_ratio":      {"unit": "-"},
                "at_resonance_power_w": {"unit": "W"},
            },
            "source": "Roundy et al. (2003) Smart Mater. Struct.; Erturk & Inman (2011) Wiley",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"acceleration": 9.81, "frequency": 100.0})
    print(f"Piezo at 1g, 100Hz (resonance): "
          f"P={float(r['power_uw']):.1f} uW, "
          f"V={float(r['voltage_v']):.3f} V")
