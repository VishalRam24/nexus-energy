"""EC221 — MHD Generator — F1a — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MHDF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MHDF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            sigma : float or array — plasma conductivity [S/m]
            u     : float or array — plasma velocity [m/s]
            B     : float or array — magnetic field [T]
            K     : float or array — load factor [-] (0=short, 1=open, 0.5=max power)
        returns:
            EMF_V              : V
            J_Am2              : A/m^2
            power_density_Wm3  : W/m^3
            power_w            : W
            heat_input_w       : W
            eta_mhd            : dimensionless (max 0.25 at K=0.5)
            eta_plant          : dimensionless
        """
        sigma = np.asarray(inputs["sigma"], dtype=float)
        u = np.asarray(inputs["u"], dtype=float)
        B = np.asarray(inputs["B"], dtype=float)
        K = np.asarray(inputs["K"], dtype=float)
        return self._model.compute(sigma, u, B, K)

    def get_info(self) -> dict:
        return {
            "name": "Magnetohydrodynamic (MHD) Generator",
            "ec_id": "EC221",
            "fidelity": "F1a",
            "description": "P = sigma*u^2*B^2*K*(1-K)*V_channel; max power at K=0.5",
            "inputs": {
                "sigma": {"unit": "S/m", "range": [1.0, 100.0]},
                "u":     {"unit": "m/s", "range": [100.0, 2000.0]},
                "B":     {"unit": "T",   "range": [0.5, 10.0]},
                "K":     {"unit": "-",   "range": [0.0, 1.0]},
            },
            "outputs": {
                "EMF_V":             {"unit": "V"},
                "J_Am2":             {"unit": "A/m^2"},
                "power_density_Wm3": {"unit": "W/m^3"},
                "power_w":           {"unit": "W"},
                "heat_input_w":      {"unit": "W"},
                "eta_mhd":           {"unit": "-"},
                "eta_plant":         {"unit": "-"},
            },
            "source": "Rosa (1987) MHD Energy Conversion; Messerle (1995) Wiley",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5})
    print(f"MHD at sigma=10, u=800m/s, B=5T, K=0.5: "
          f"P={float(r['power_w'])/1e6:.2f} MW, "
          f"eta_mhd={float(r['eta_mhd'])*100:.1f}%, "
          f"eta_plant={float(r['eta_plant'])*100:.1f}%")
