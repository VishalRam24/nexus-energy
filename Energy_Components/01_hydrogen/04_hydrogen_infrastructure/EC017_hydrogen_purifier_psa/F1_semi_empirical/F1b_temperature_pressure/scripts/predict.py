"""
EC017 -- PSA -- F1b Temperature-Pressure -- Standardized Predict Interface
"""
import json, numpy as np
from pathlib import Path
from model import HydrogenPurifierPSAF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HydrogenPurifierPSAF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict PSA performance with temperature and pressure effects.

        Args:
            inputs: dict with keys:
                - feed_flow_kg_s (kg/s):    Feed mass flow
                - feed_h2_fraction (0-1):   Feed H2 mole fraction
                - feed_pressure_bar (bar):  Feed pressure
                - temperature_K (K):        Adsorbent temperature (optional, default T_ref)
                - target_purity (0-1):      Target product purity (optional)
        """
        F   = np.asarray(inputs["feed_flow_kg_s"], dtype=float)
        y   = np.asarray(inputs["feed_h2_fraction"], dtype=float)
        P   = np.asarray(inputs["feed_pressure_bar"], dtype=float)
        T   = inputs.get("temperature_K", None)
        pur = inputs.get("target_purity", self._model.purity_nom)
        if T is not None:
            T = np.asarray(T, dtype=float)
        pur = np.asarray(pur, dtype=float)

        return self._model.evaluate(F, y, P, T_K=T, target_purity=pur)

    def get_info(self) -> dict:
        return {
            "name": "Hydrogen Purifier (PSA)",
            "ec_id": "EC017",
            "fidelity": "F1b",
            "description": (
                "PSA with temperature-dependent recovery (lower T -> better selectivity) "
                "and specific energy W = W_nom*(P_ref/P)^0.15*(T/T_ref)^0.5."
            ),
            "inputs": {
                "feed_flow_kg_s":    {"unit": "kg/s",         "range": [1e-6, 100.0]},
                "feed_h2_fraction":  {"unit": "mol_fraction", "range": [0.3, 0.999]},
                "feed_pressure_bar": {"unit": "bar",          "range": [5.0, 80.0]},
                "temperature_K":     {"unit": "K",            "range": [253.15, 353.15], "optional": True},
                "target_purity":     {"unit": "mol_fraction", "range": [0.99, 0.99999], "optional": True},
            },
            "outputs": {
                "recovery":                   {"unit": "dimensionless"},
                "product_flow_kg_s":          {"unit": "kg/s"},
                "tail_gas_flow_kg_s":         {"unit": "kg/s"},
                "specific_energy_kWh_per_kg": {"unit": "kWh/kg_H2"},
                "electric_power_kW":          {"unit": "kW"},
                "pressure_ratio":             {"unit": "dimensionless"},
                "h2_yield_kg_s":              {"unit": "kg/s"},
            },
            "source": "Sircar & Golden (2000); Yang (1987); Ruthven (1984)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC017 PSA F1b Temperature-Pressure")
    print("\n-- Effect of temperature (P=20 bar, y=0.75) --")
    for T in [253.15, 273.15, 298.15, 323.15, 353.15]:
        r = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                           "feed_pressure_bar": 20.0, "temperature_K": T})
        print(f"  T={T:.0f}K: eta={float(r['recovery']):.3f}, "
              f"W={float(r['specific_energy_kWh_per_kg']):.2f} kWh/kg")
