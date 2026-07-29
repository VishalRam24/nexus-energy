"""EC137 — Oscillating Body / Attenuator WEC — F1a — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import AttenuatorWECF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AttenuatorWECF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          H_s : significant wave height [m]
          T_e : wave energy period [s]
          cwr : capture width ratio override (optional)
        returns:
          wave_power_per_m_kw, power_kw, overall_efficiency, power_density_kw_per_m
        """
        H_s = np.asarray(inputs["H_s"], dtype=float)
        T_e = np.asarray(inputs["T_e"], dtype=float)
        cwr = inputs.get("cwr", None)

        J   = self._model.wave_power_per_metre(H_s, T_e) / 1e3
        P   = self._model.power_kw(H_s, T_e, cwr)
        eta = self._model.overall_efficiency(cwr)
        pd  = self._model.rated_power_density_kw_per_m(H_s, T_e)
        return {
            "wave_power_per_m_kw":     J,
            "power_kw":                P,
            "overall_efficiency":      eta,
            "power_density_kw_per_m":  pd,
        }

    def get_info(self) -> dict:
        return {
            "name":        "Oscillating Body / Attenuator WEC (Pelamis-type)",
            "ec_id":       "EC137",
            "fidelity":    "F1a",
            "description": "P = J * W * CWR * eta_pto * eta_elec; CWR 0.20-0.35 for attenuators",
            "inputs": {
                "H_s": {"unit": "m", "range": [0.0, 8.0]},
                "T_e": {"unit": "s", "range": [5.0, 20.0]},
                "cwr": {"unit": "-", "range": [0.20, 0.35], "default": 0.25},
            },
            "outputs": {
                "wave_power_per_m_kw":    {"unit": "kW/m"},
                "power_kw":               {"unit": "kW"},
                "overall_efficiency":     {"unit": "-"},
                "power_density_kw_per_m": {"unit": "kW/m"},
            },
            "source": "Henderson (2006) Appl. Ocean Res. 28:297-307; Yemm et al. (2012) Phil. Trans. A 370:365",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    print(f"\nH_s=2m, T_e=10s: P={float(r['power_kw']):.2f} kW, "
          f"eta={float(r['overall_efficiency']):.4f}, "
          f"pd={float(r['power_density_kw_per_m']):.4f} kW/m")
