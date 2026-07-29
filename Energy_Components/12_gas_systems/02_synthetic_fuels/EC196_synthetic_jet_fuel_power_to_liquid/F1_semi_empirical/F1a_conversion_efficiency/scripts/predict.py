"""EC196 — Synthetic Jet Fuel — F1a Conversion Efficiency — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import FTJetFuelF1a


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = FTJetFuelF1a(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        return self.model.compute(
            n_co_in      = np.asarray(inputs.get("n_co_in", self.model.n_CO_in), dtype=float),
            temperature_C= np.asarray(inputs["temperature_C"], dtype=float),
            pressure_bar = np.asarray(inputs["pressure_bar"], dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC196",
            "component": "Synthetic Jet Fuel (Power-to-Liquid)",
            "fidelity": "F1a",
            "sub_fidelity": "conversion_efficiency",
            "description": "LTFT reactor: CO conversion via Gaussian equilibrium fit, ASF C8-C16 selectivity, jet fuel yield.",
            "inputs": ["n_co_in (mol/s)", "temperature_C", "pressure_bar"],
            "outputs": ["co_conversion", "selectivity_jet_C8_C16", "jet_fuel_mol_s",
                        "energy_efficiency", "heat_released_kW"],
            "references": ["Dry (2002) Catalysis Today", "Schulz (1999)", "Anderson (1956)"],
        }
