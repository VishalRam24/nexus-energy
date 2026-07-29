"""EC220 — Triboelectric Nanogenerator (TENG) — F1a — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TENGF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TENGF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            frequency : float or array — contact-separation frequency [Hz]
            R_load    : float or array — load resistance [ohm]
        returns:
            V_oc_peak_V        : V (open-circuit peak voltage)
            C_avg_F            : F (average capacitance)
            R_internal_ohm     : ohm
            power_avg_w        : W
            power_density_mwcm2: mW/cm^2
            efficiency         : dimensionless
        """
        f = np.asarray(inputs["frequency"], dtype=float)
        R = np.asarray(inputs["R_load"], dtype=float)
        return self._model.compute(f, R)

    def get_info(self) -> dict:
        return {
            "name": "Triboelectric Nanogenerator (TENG)",
            "ec_id": "EC220",
            "fidelity": "F1a",
            "description": "V_oc=sigma*x/eps0; P=V_oc^2*R/(R+R_int)^2/2; high-impedance capacitive source",
            "inputs": {
                "frequency": {"unit": "Hz",  "range": [0.1, 100.0]},
                "R_load":    {"unit": "ohm", "range": [1e4, 1e10]},
            },
            "outputs": {
                "V_oc_peak_V":         {"unit": "V"},
                "C_avg_F":             {"unit": "F"},
                "R_internal_ohm":      {"unit": "ohm"},
                "power_avg_w":         {"unit": "W"},
                "power_density_mwcm2": {"unit": "mW/cm^2"},
                "efficiency":          {"unit": "-"},
            },
            "source": "Wang (2013) ACS Nano; Niu & Wang (2015) Nano Energy; Fan et al. (2012) Nano Lett.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"frequency": 3.0, "R_load": 1e7})
    print(f"TENG at 3Hz, 10MOhm: "
          f"V_oc={float(r['V_oc_peak_V']):.1f} V, "
          f"P={float(r['power_avg_w'])*1000:.3f} mW, "
          f"P_density={float(r['power_density_mwcm2']):.4f} mW/cm^2, "
          f"eta={float(r['efficiency'])*100:.1f}%")
