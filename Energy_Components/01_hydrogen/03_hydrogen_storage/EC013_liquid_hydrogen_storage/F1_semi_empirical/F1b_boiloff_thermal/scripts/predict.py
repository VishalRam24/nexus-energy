"""
EC013 -- LH2 Storage -- F1b Boil-Off Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LH2ThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)


class ComponentModel:
    """
    Standardised wrapper for the LH2 storage thermal boil-off model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"fill_fraction": 0.8, "T_ambient": 298.15})
    """

    component_id   = "EC013"
    component_name = "Liquid Hydrogen (LH2) Storage"
    fidelity       = "F1b -- Boil-Off Thermal (T_amb variation, MLI k(T), pressurization)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        self._raw = defaults
        self._physics = LH2ThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Predict steady-state boil-off.

        Parameters
        ----------
        inputs : dict
            fill_fraction : float -- 0.05 – 0.95
            T_ambient     : float -- K (233 – 333)
            P_tank        : float -- bar (1.0 – 6.0), optional, default 1.013

        Returns
        -------
        dict with stored_mass_kg, energy_stored_MJ, heat_leak_W,
             boiloff_rate_kg_s, BOR_pct_day, U_eff_W_m2_K, T_sat_K
        """
        f = float(inputs["fill_fraction"])
        T = float(inputs.get("T_ambient", self._physics.T_amb_default))
        P = float(inputs.get("P_tank", 1.01325))

        if not (0.05 <= f <= 0.95):
            raise ValueError(f"fill_fraction must be in [0.05, 0.95], got {f}")
        if not (233.15 <= T <= 333.15):
            raise ValueError(f"T_ambient must be in [233.15, 333.15] K, got {T}")
        if not (1.0 <= P <= 6.0):
            raise ValueError(f"P_tank must be in [1.0, 6.0] bar, got {P}")

        res = self._physics.evaluate(f, T, P)
        return {
            "stored_mass_kg":    round(float(res["stored_mass_kg"]),    4),
            "energy_stored_MJ":  round(float(res["energy_stored_MJ"]),  4),
            "heat_leak_W":       round(float(res["heat_leak_W"]),        4),
            "boiloff_rate_kg_s": round(float(res["boiloff_rate_kg_s"]), 8),
            "BOR_pct_day":       round(float(res["BOR_pct_day"]),        6),
            "U_eff_W_m2_K":      round(float(res["U_eff_W_m2_K"]),      8),
            "T_sat_K":           round(float(res["T_sat_K"]),            4),
        }

    def get_info(self) -> dict:
        p = self._physics
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "fill_fraction": {"unit": "-",   "range": [0.05, 0.95]},
                "T_ambient":     {"unit": "K",   "range": [233.15, 333.15]},
                "P_tank":        {"unit": "bar", "range": [1.0, 6.0], "optional": True},
            },
            "outputs": {
                "stored_mass_kg":    "kg",
                "energy_stored_MJ":  "MJ",
                "heat_leak_W":       "W",
                "boiloff_rate_kg_s": "kg/s",
                "BOR_pct_day":       "%/day",
                "U_eff_W_m2_K":      "W/(m2.K)",
                "T_sat_K":           "K",
            },
            "source": "Sherif et al. (1997); Johnson (2010) NREL; Petitpas (2018) NREL",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"fill_fraction": 0.8, "T_ambient": 298.15}))
    print(model.predict({"fill_fraction": 0.8, "T_ambient": 313.15}))  # hotter ambient
