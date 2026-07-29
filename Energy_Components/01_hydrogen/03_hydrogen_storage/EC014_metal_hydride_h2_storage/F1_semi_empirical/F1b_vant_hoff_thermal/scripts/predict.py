"""
EC014 -- Metal Hydride H2 Storage -- F1b van't Hoff Thermal -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"temperature": 298.15, "pressure_bar": 5.0, "soc": 0.5})
"""

import json
import numpy as np
from pathlib import Path
from model import MetalHydrideH2F1b


class ComponentModel:
    """Standardized interface for EC014 Metal Hydride H2 Storage -- F1b thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MetalHydrideH2F1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict metal hydride storage state with kinetics and thermal balance.

        Args:
            inputs: dict with keys:
                - temperature (K):     Bed temperature
                - pressure_bar (bar):  Gas-phase H2 pressure
                - soc (0-1):           State of charge
                - mode (str):          'absorption' or 'desorption' (default: 'desorption')
                - T_amb_K (K):         Coolant/ambient temperature (default: 298.15 K)

        Returns:
            dict with plateau_pressure_bar, sorption_rate_kg_s, reaction_heat_W,
                      dTdt_K_s, stored_mass_kg, gravimetric_wt_pct, volumetric_kg_per_m3
        """
        T     = np.asarray(inputs["temperature"], dtype=float)
        P     = np.asarray(inputs.get("pressure_bar", 5.0), dtype=float)
        soc   = np.asarray(inputs["soc"], dtype=float)
        mode  = inputs.get("mode", "desorption")
        T_amb = np.asarray(inputs.get("T_amb_K", 298.15), dtype=float)

        return self._model.evaluate(T, P, soc, mode=mode, T_amb_K=T_amb)

    def get_info(self) -> dict:
        return {
            "name": "Metal Hydride H2 Storage",
            "ec_id": "EC014",
            "fidelity": "F1b",
            "description": (
                "van't Hoff equilibrium + first-order Arrhenius kinetics + lumped thermal balance. "
                "Sandrock (1999) sign convention: ln(P/P_ref) = -ΔH_des/(RT) + ΔS_des/R."
            ),
            "material": "LaNi5",
            "inputs": {
                "temperature":  {"unit": "K",             "range": [253.0, 373.0]},
                "pressure_bar": {"unit": "bar",            "range": [0.1, 100.0]},
                "soc":          {"unit": "dimensionless",  "range": [0.0, 1.0]},
                "mode":         {"values": ["absorption", "desorption"]},
                "T_amb_K":      {"unit": "K",              "range": [253.0, 373.0], "optional": True},
            },
            "outputs": {
                "plateau_pressure_bar":  {"unit": "bar"},
                "sorption_rate_kg_s":    {"unit": "kg/s", "note": "positive=absorption, negative=desorption"},
                "reaction_heat_W":       {"unit": "W",    "note": "positive=heat released (absorption)"},
                "dTdt_K_s":              {"unit": "K/s"},
                "stored_mass_kg":        {"unit": "kg"},
                "gravimetric_wt_pct":    {"unit": "wt%"},
                "volumetric_kg_per_m3":  {"unit": "kg/m3"},
            },
            "source": "Sandrock (1999); Mayer et al. (1987); Lototskyy et al. (2014)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC014 Metal Hydride H2 Storage -- F1b van't Hoff Thermal")
    print("\n-- Desorption plateau pressure vs temperature --")
    for T in [273.15, 298.15, 323.15, 353.15]:
        r = model.predict({"temperature": T, "pressure_bar": 5.0, "soc": 0.5, "mode": "desorption"})
        print(f"  T={T:.0f} K: P_eq={float(r['plateau_pressure_bar']):.2f} bar, "
              f"rate={float(r['sorption_rate_kg_s'])*1e3:.3f} g/s")
