"""EC136 — Overtopping Device WEC — F1a — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import OvertoppingWECF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OvertoppingWECF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          H_s     : significant wave height [m]
          T_e     : wave energy period [s]
          Q_m3s   : overtopping flow [m^3/s] (optional; overrides wave-based calc)
        returns:
          wave_power_per_m_kw, power_kw, overall_efficiency
        """
        H_s = np.asarray(inputs["H_s"], dtype=float)
        T_e = np.asarray(inputs["T_e"], dtype=float)

        J   = self._model.wave_power_per_metre(H_s, T_e) / 1e3
        P   = self._model.power_kw(H_s, T_e)
        eta = self._model.overall_efficiency()
        return {
            "wave_power_per_m_kw": J,
            "power_kw":            P,
            "overall_efficiency":  eta,
        }

    def get_info(self) -> dict:
        return {
            "name":        "Overtopping WEC (ramp + reservoir + low-head turbine)",
            "ec_id":       "EC136",
            "fidelity":    "F1a",
            "description": "P = J * W * eta_ramp * eta_turbine * eta_gen; eta_ramp ~15-25%",
            "inputs": {
                "H_s": {"unit": "m", "range": [0.5, 6.0]},
                "T_e": {"unit": "s", "range": [5.0, 20.0]},
            },
            "outputs": {
                "wave_power_per_m_kw": {"unit": "kW/m"},
                "power_kw":            {"unit": "kW"},
                "overall_efficiency":  {"unit": "-"},
            },
            "source": "Kofoed et al. (2006) Coastal Eng. 53:859-867; Wave Dragon device data",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    print(f"\nH_s=2m, T_e=10s: J={float(r['wave_power_per_m_kw']):.2f} kW/m, "
          f"P={float(r['power_kw']):.2f} kW, eta={float(r['overall_efficiency']):.4f}")
