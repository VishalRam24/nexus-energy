"""EC017 — Hydrogen Purifier (PSA) — F1a Recovery-Purity — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HydrogenPurifierPSAF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HydrogenPurifierPSAF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict PSA purifier performance.

        Args:
            inputs: dict with keys:
                - feed_flow_kg_s (kg/s):    Total feed mass flow rate
                - feed_h2_fraction (0-1):   H2 mole fraction in feed
                - feed_pressure_bar (bar):  Feed pressure
                - target_purity (0-1):      Target product H2 purity (mol fraction)

        Returns:
            dict with keys:
                - recovery, product_flow_kg_s, tail_gas_flow_kg_s,
                  specific_energy_kWh_per_kg, electric_power_kW,
                  pressure_ratio, h2_yield_kg_s
        """
        F_feed = np.asarray(inputs["feed_flow_kg_s"], dtype=float)
        y_H2 = np.asarray(inputs["feed_h2_fraction"], dtype=float)
        P_feed = np.asarray(inputs["feed_pressure_bar"], dtype=float)
        purity = np.asarray(inputs.get("target_purity", self._model.purity_nom), dtype=float)

        eta_rec = self._model.recovery(P_feed, y_H2, target_purity=purity)
        F_product = self._model.product_flow(F_feed, y_H2, eta_rec, purity)
        F_tail = self._model.tail_gas_flow(F_feed, F_product)
        W_spec = self._model.specific_energy(P_feed)
        P_kW = self._model.electric_power(F_product, P_feed)
        P_ratio = self._model.pressure_ratio(P_feed)

        # H2 yield: actual H2 mass flow in product
        F_H2_product = F_product * purity

        return {
            "recovery":                   eta_rec,
            "product_flow_kg_s":          F_product,
            "tail_gas_flow_kg_s":         F_tail,
            "specific_energy_kWh_per_kg": W_spec,
            "electric_power_kW":          P_kW,
            "pressure_ratio":             P_ratio,
            "h2_yield_kg_s":              F_H2_product,
        }

    def get_info(self) -> dict:
        return {
            "name": "Hydrogen Purifier (PSA)",
            "ec_id": "EC017",
            "fidelity": "F1a",
            "description": (
                "PSA semi-empirical model: recovery η_rec(P, y_H2, purity), "
                "product/tail flows, specific energy ~1–3 kWh/kg H2"
            ),
            "inputs": {
                "feed_flow_kg_s":    {"unit": "kg/s",          "range": [1e-6, 100.0]},
                "feed_h2_fraction":  {"unit": "mol_fraction",  "range": [0.3, 0.999]},
                "feed_pressure_bar": {"unit": "bar",           "range": [5.0, 80.0]},
                "target_purity":     {"unit": "mol_fraction",  "range": [0.99, 0.99999]},
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
            "source": "Sircar & Golden (2000) Sep. Sci. Technol.; Yang (1987); DOE H2 Program",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC017 Hydrogen Purifier PSA — F1a Recovery-Purity")
    for P in [10, 20, 40, 60]:
        r = model.predict({
            "feed_flow_kg_s": 0.1,
            "feed_h2_fraction": 0.75,
            "feed_pressure_bar": P,
            "target_purity": 0.9999,
        })
        print(f"  P={P} bar: η_rec={float(r['recovery']):.3f}, "
              f"W={float(r['specific_energy_kWh_per_kg']):.2f} kWh/kg, "
              f"F_prod={float(r['product_flow_kg_s'])*3600:.2f} kg/h")
