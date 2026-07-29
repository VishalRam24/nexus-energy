"""EC134 — Oscillating Water Column (OWC) — F1a — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import OWCF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OWCF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          H_s : significant wave height [m]
          T_e : wave energy period [s]
          cwr : capture width ratio override (optional, default from params)
        returns:
          wave_power_per_m_kw, power_kw, overall_efficiency, capture_width_m
        """
        H_s = np.asarray(inputs["H_s"], dtype=float)
        T_e = np.asarray(inputs["T_e"], dtype=float)
        cwr = inputs.get("cwr", None)

        J   = self._model.wave_power_per_metre(H_s, T_e) / 1e3  # kW/m
        P   = self._model.power_kw(H_s, T_e, cwr)
        eff = self._model.overall_efficiency(cwr)
        cw  = self._model.capture_width_m(H_s, T_e, cwr)
        return {
            "wave_power_per_m_kw": J,
            "power_kw":            P,
            "overall_efficiency":  eff,
            "capture_width_m":     cw,
        }

    def get_info(self) -> dict:
        return {
            "name":        "Oscillating Water Column (OWC) WEC",
            "ec_id":       "EC134",
            "fidelity":    "F1a",
            "description": "P_wave=(rho*g^2*H_s^2*T_e)/(64pi); P_elec=P_wave*width*CWR*eta_turbine*eta_gen",
            "inputs": {
                "H_s": {"unit": "m",   "range": [0.0, 8.0]},
                "T_e": {"unit": "s",   "range": [5.0, 20.0]},
                "cwr": {"unit": "-",   "range": [0.1, 0.3], "default": 0.20},
            },
            "outputs": {
                "wave_power_per_m_kw": {"unit": "kW/m"},
                "power_kw":            {"unit": "kW"},
                "overall_efficiency":  {"unit": "-"},
                "capture_width_m":     {"unit": "m"},
            },
            "source": "Falnes (2002); Folley (2016); EMEC TR-001",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    print(f"\nH_s=2m, T_e=10s: J={float(r['wave_power_per_m_kw']):.2f} kW/m, "
          f"P={float(r['power_kw']):.2f} kW, eta={float(r['overall_efficiency']):.4f}")
