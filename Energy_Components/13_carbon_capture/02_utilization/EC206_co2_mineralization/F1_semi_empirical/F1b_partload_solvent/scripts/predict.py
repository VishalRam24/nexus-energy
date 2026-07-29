"""EC206 — CO2 Mineralization — F1b Part-Load + Conversion Degradation — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import CO2MineralizationF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2MineralizationF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict CO2 mineralization performance at part-load with conversion degradation.

        Parameters
        ----------
        inputs : dict
            co2_flow_kg_h    : float (kg/h, default 1000)
            PLR              : float (0.3-1.0, default 1.0)
            operating_hours  : float (hours, default 0)
        """
        flow = inputs.get("co2_flow_kg_h", 1000.0)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        hours = inputs.get("operating_hours", 0.0)

        return self._model.compute(flow, plr, hours)

    def get_info(self) -> dict:
        return {
            "name": "CO2 Mineralization (Accelerated Carbonation)",
            "ec_id": "EC206",
            "fidelity": "F1b",
            "description": (
                "Aqueous mineral carbonation with surface passivation degradation "
                "(0.03%/h conversion loss) and part-load grinding energy penalty."
            ),
            "inputs": {
                "co2_flow_kg_h": {"unit": "kg/h", "range": [100, 100000], "default": 1000},
                "PLR": {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "operating_hours": {"unit": "hours", "range": [0, 50000], "default": 0},
            },
            "outputs": {
                "co2_stored_kg_h": {"unit": "kg/h"},
                "sec_kwh_tco2": {"unit": "kWh/tCO2"},
                "conversion_efficiency": {"unit": "dimensionless"},
                "conversion_relative_pct": {"unit": "%"},
                "carbonate_product_kg_h": {"unit": "kg/h"},
                "mineral_feed_t_per_tco2": {"unit": "t_mineral/tCO2"},
            },
            "source": "Sanna et al. (2014); Lackner et al. (1995)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"co2_flow_kg_h": 1000.0, "PLR": 1.0, "operating_hours": 0})
    print("Design point (PLR=1.0, fresh reactor):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
