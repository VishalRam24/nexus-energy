"""EC223 — RTG — F1b TEG Layered Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import RTGF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RTGF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict RTG performance with temperature-dependent SiGe TEG model.

        Parameters
        ----------
        inputs : dict
            t_years : float or array — time since launch [years]

        Returns
        -------
        dict with P_thermal_W, T_hj_K, T_cj_K, ZT_avg, eta_teg, eta_carnot,
                  P_electric_W, P_max_circuit_W, V_oc_V, I_mp_A, R_int_ohm,
                  fraction_thermal_remaining, power_fraction
        """
        t = inputs.get("t_years", 0.0)
        return self._model.compute(t)

    def get_info(self) -> dict:
        return {
            "name": "Radioisotope Thermoelectric Generator (RTG)",
            "ec_id": "EC223",
            "fidelity": "F1b",
            "description": (
                "Multi-layer RTG model: Pu-238 decay heat, temperature-dependent SiGe "
                "material properties (alpha(T), k(T), sigma(T)), ZT averaged over "
                "junction temperature gradient, thermal resistance network for junction "
                "temperatures, self-consistent P_electric via Angist formula, "
                "and matched-load V_oc / I_mp / R_int outputs."
            ),
            "inputs": {
                "t_years": {"unit": "years", "range": [0, 200], "default": 0.0},
            },
            "outputs": {
                "P_thermal_W": {"unit": "W"},
                "T_hj_K": {"unit": "K"},
                "T_cj_K": {"unit": "K"},
                "ZT_avg": {"unit": "-"},
                "eta_teg": {"unit": "-"},
                "eta_carnot": {"unit": "-"},
                "P_electric_W": {"unit": "W"},
                "P_max_circuit_W": {"unit": "W"},
                "V_oc_V": {"unit": "V"},
                "I_mp_A": {"unit": "A"},
                "R_int_ohm": {"unit": "ohm"},
                "fraction_thermal_remaining": {"unit": "-"},
                "power_fraction": {"unit": "-"},
            },
            "source": "Bennett (2006); El-Genk & Saber (2005); Fleurial et al. (1997); Rowe (2006)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("RTG F1b — Design Point (t=0):")
    r = model.predict({"t_years": 0.0})
    for k, v in r.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k:30s} = {val:.4g}")
    print("\nAfter 50 years (Voyager mission analog):")
    r50 = model.predict({"t_years": 50.0})
    for k, v in r50.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k:30s} = {val:.4g}")
