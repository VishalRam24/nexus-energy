"""EC223 — RTG — F1a Decay + TEG Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import RTGF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RTGF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            t_years : float or array — time since deployment [years]
        returns:
            P_thermal_W              : W (decay heat)
            eta_teg                  : dimensionless
            P_electric_W             : W (electrical output)
            T_hot_K                  : K (hot-side temperature)
            eta_carnot               : dimensionless
            fraction_thermal_remaining: dimensionless
            power_fraction           : dimensionless (P_electric / P0_electric)
        """
        t = np.asarray(inputs["t_years"], dtype=float)
        return self._model.compute(t)

    def get_info(self) -> dict:
        return {
            "name": "Radioisotope Thermoelectric Generator (RTG)",
            "ec_id": "EC223",
            "fidelity": "F1a",
            "description": "P_th=P0*exp(-ln2*t/t_half); T_hot~P^0.25; eta_TEG from ZT + degradation",
            "inputs": {
                "t_years": {"unit": "years", "range": [0.0, 200.0]},
            },
            "outputs": {
                "P_thermal_W":               {"unit": "W"},
                "eta_teg":                   {"unit": "-"},
                "P_electric_W":              {"unit": "W"},
                "T_hot_K":                   {"unit": "K"},
                "eta_carnot":                {"unit": "-"},
                "fraction_thermal_remaining":{"unit": "-"},
                "power_fraction":            {"unit": "-"},
            },
            "source": "Bennett (2006) AIAA; El-Genk & Saber (2005) Energy Convers. Mgmt.; NASA GPHS-RTG",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    design_life = model.params["unit"]["design_life_years"]["value"]
    t_test = [0, 10, 20, 30, 45, 50]
    print(f"RTG (Pu-238 GPHS-RTG):")
    for t in t_test:
        r = model.predict({"t_years": float(t)})
        print(f"  t={t:3d}y: P_th={float(r['P_thermal_W']):.1f}W, "
              f"P_el={float(r['P_electric_W']):.1f}W, "
              f"eta={float(r['eta_teg'])*100:.2f}%, "
              f"remaining={float(r['fraction_thermal_remaining'])*100:.1f}%")
