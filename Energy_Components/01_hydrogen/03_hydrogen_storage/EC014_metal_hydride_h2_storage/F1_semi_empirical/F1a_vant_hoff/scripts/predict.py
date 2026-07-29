"""EC014 — Metal Hydride H2 Storage — F1a van't Hoff — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MetalHydrideH2F1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MetalHydrideH2F1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict metal hydride H2 storage state.

        Args:
            inputs: dict with keys:
                - temperature (K):    Bed temperature
                - soc (0-1):          State of charge (0=empty, 1=full)
                - mode (str):         'absorption' or 'desorption' (default: 'desorption')
                - delta_m_H2_kg (kg): H2 mass change step (optional, default 0)

        Returns:
            dict with keys:
                - plateau_pressure_bar, stored_mass_kg, heat_of_reaction_kJ,
                  gravimetric_wt_pct, volumetric_kg_per_m3, fill_fraction
        """
        T = np.asarray(inputs["temperature"], dtype=float)
        soc = np.asarray(inputs["soc"], dtype=float)
        mode = inputs.get("mode", "desorption")
        delta_m = np.asarray(inputs.get("delta_m_H2_kg", 0.0), dtype=float)

        P_eq = self._model.plateau_pressure(T, mode=mode, soc=soc)
        m_h2 = self._model.stored_mass(soc)
        q_rxn = self._model.heat_of_reaction(delta_m)

        return {
            "plateau_pressure_bar":   P_eq,
            "stored_mass_kg":         m_h2,
            "heat_of_reaction_kJ":    q_rxn,
            "gravimetric_wt_pct":     self._model.gravimetric_density(soc),
            "volumetric_kg_per_m3":   self._model.volumetric_density(soc),
            "fill_fraction":          self._model.fill_fraction(soc),
        }

    def get_info(self) -> dict:
        return {
            "name": "Metal Hydride H2 Storage",
            "ec_id": "EC014",
            "fidelity": "F1a",
            "description": "van't Hoff plateau pressure model; ln(P_eq) = ΔH/(RT) - ΔS/R",
            "material": "LaNi5 (reference intermetallic)",
            "inputs": {
                "temperature":   {"unit": "K",            "range": [253.0, 373.0]},
                "soc":           {"unit": "dimensionless", "range": [0.0, 1.0]},
                "mode":          {"unit": "str",           "values": ["absorption", "desorption"]},
                "delta_m_H2_kg": {"unit": "kg",            "range": [-1.0, 1.0], "optional": True},
            },
            "outputs": {
                "plateau_pressure_bar":  {"unit": "bar"},
                "stored_mass_kg":        {"unit": "kg"},
                "heat_of_reaction_kJ":   {"unit": "kJ"},
                "gravimetric_wt_pct":    {"unit": "wt%"},
                "volumetric_kg_per_m3":  {"unit": "kg/m3"},
                "fill_fraction":         {"unit": "dimensionless"},
            },
            "source": "Lototskyy et al. (2014) Prog. Nat. Sci. Mater.; Sakintuna et al. (2007) Int. J. Hydrogen Energy",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC014 Metal Hydride H2 Storage — F1a van't Hoff")
    for T in [273.15, 298.15, 323.15, 353.15]:
        r = model.predict({"temperature": T, "soc": 0.5, "mode": "desorption"})
        print(f"  T={T:.0f} K: P_eq={float(r['plateau_pressure_bar']):.2f} bar, "
              f"m_H2={float(r['stored_mass_kg']):.4f} kg")
