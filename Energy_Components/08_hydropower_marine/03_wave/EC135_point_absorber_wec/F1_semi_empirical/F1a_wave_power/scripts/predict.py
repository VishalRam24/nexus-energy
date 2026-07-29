"""EC135 — Point Absorber WEC — F1a — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import PointAbsorberF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PointAbsorberF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          H_s : significant wave height [m]
          T_e : wave energy period [s]
        returns:
          wave_power_per_m_kw, power_kw, capture_width_ratio, overall_efficiency
        """
        H_s = np.asarray(inputs["H_s"], dtype=float)
        T_e = np.asarray(inputs["T_e"], dtype=float)

        J   = self._model.wave_power_per_metre(H_s, T_e) / 1e3
        P   = self._model.power_kw(H_s, T_e)
        cwr = self._model.capture_width_ratio(T_e)
        eta = self._model.overall_efficiency(T_e)
        return {
            "wave_power_per_m_kw":  J,
            "power_kw":             P,
            "capture_width_ratio":  cwr,
            "overall_efficiency":   eta,
        }

    def get_info(self) -> dict:
        return {
            "name":        "Point Absorber WEC (heaving buoy + linear generator)",
            "ec_id":       "EC135",
            "fidelity":    "F1a",
            "description": "CWR Gaussian resonance model; P = J * D * CWR(T_e) * eta_pto * eta_elec",
            "inputs": {
                "H_s": {"unit": "m", "range": [0.0, 8.0]},
                "T_e": {"unit": "s", "range": [4.0, 20.0]},
            },
            "outputs": {
                "wave_power_per_m_kw": {"unit": "kW/m"},
                "power_kw":            {"unit": "kW"},
                "capture_width_ratio": {"unit": "-"},
                "overall_efficiency":  {"unit": "-"},
            },
            "source": "Falnes (2002); Babarit et al. (2012) Renew. Energy 41:44-63",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    print(f"\nAt resonance (H_s=2m, T_e=10s): P={float(r['power_kw']):.2f} kW, "
          f"CWR={float(r['capture_width_ratio']):.3f}, eta={float(r['overall_efficiency']):.4f}")
