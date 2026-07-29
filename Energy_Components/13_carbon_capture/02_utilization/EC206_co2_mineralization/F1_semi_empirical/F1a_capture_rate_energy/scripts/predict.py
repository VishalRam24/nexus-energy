"""EC206 — CO2 Mineralization — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2MineralizationF1a


class ComponentModel:
    component_id = "EC206"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2MineralizationF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction : float or array [0.1, 1.0]
        Returns:
            conversion, co2_stored_tCO2_h, W_elec_GJ_h,
            SEC_GJ_tCO2, carbonate_produced_t_h
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)

        return {
            "conversion": float(self._model.conversion),
            "co2_stored_tCO2_h": self._model.co2_stored(cf),
            "W_elec_GJ_h": self._model.electric_energy(cf),
            "SEC_GJ_tCO2": float(self._model.SEC),
            "carbonate_produced_t_h": self._model.carbonate_produced(cf),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "CO2 Mineralization",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Permanent CO2 storage via mineral carbonation. MgO/CaO + CO2 -> carbonates. "
                "conversion=0.80, SEC=0.5 GJ/tCO2 (mechanical energy)."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
            },
            "outputs": {
                "conversion": {"unit": "dimensionless"},
                "co2_stored_tCO2_h": {"unit": "tCO2/h"},
                "W_elec_GJ_h": {"unit": "GJ/h"},
                "SEC_GJ_tCO2": {"unit": "GJ/tCO2"},
                "carbonate_produced_t_h": {"unit": "t/h"},
            },
            "params": {
                "conversion": str(u["conversion"]["value"]),
                "SEC": f"{u['SEC_GJ_tCO2']['value']} GJ/tCO2",
                "sorbent": u["sorbent"]["value"],
                "reaction": u["reaction"]["value"],
            },
            "source": "Sanna et al. (2014); IPCC (2005)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for cf in [0.25, 0.5, 0.75, 1.0]:
        r = model.predict({"capacity_fraction": cf})
        print(f"CF={cf:.2f}: CO2_stored={float(r['co2_stored_tCO2_h']):.3f} tCO2/h  "
              f"carbonate={float(r['carbonate_produced_t_h']):.3f} t/h  "
              f"W={float(r['W_elec_GJ_h']):.4f} GJ/h")
