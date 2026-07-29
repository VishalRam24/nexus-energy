"""EC028 — Lead-Acid Battery — F1a SOC-Voltage — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import LeadAcidF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LeadAcidF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        soc     = np.asarray(inputs["soc"],     dtype=float)
        current = np.asarray(inputs["current"], dtype=float)
        return {
            "voltage":  self._model.voltage(soc, current),
            "ocv":      self._model.ocv(soc),
            "power":    self._model.power_w(soc, current),
            "dsoc_dt":  self._model.dsoc_dt(current),
        }

    def get_info(self) -> dict:
        return {
            "name":        "Lead-Acid Battery",
            "ec_id":       "EC028",
            "fidelity":    "F1a",
            "description": "V = OCV(SOC) - I*R_int; OCV cubic polynomial; positive current = discharge",
            "inputs": {
                "soc":     {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-50.0, 50.0], "note": "positive=discharge"},
            },
            "outputs": {
                "voltage":  {"unit": "V"},
                "ocv":      {"unit": "V"},
                "power":    {"unit": "W", "note": "positive=discharge"},
                "dsoc_dt":  {"unit": "1/s"},
            },
            "source":  "Copetti et al. (1993); Manwell & McGowan (1993)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"soc": 1.0, "current": 0.0})
    print(f"Full charge, no current: V={float(r['voltage']):.3f} V, OCV={float(r['ocv']):.3f} V")
    r2 = model.predict({"soc": 0.5, "current": 20.0})
    print(f"SOC=50%, 20A discharge: V={float(r2['voltage']):.3f} V")
