"""EC203 — Membrane CO2 Separation — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import MembraneF1a


class ComponentModel:
    component_id = "EC203"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MembraneF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction : float or array [0.1, 1.0]
            pressure_ratio    : float [5, 20], default 10
        Returns:
            CO2_recovery, CO2_purity, co2_captured_tCO2_h,
            W_elec_GJ_h, SEC_MJ_kgCO2
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        pr = inputs.get("pressure_ratio", None)

        co2 = self._model.co2_captured(cf)
        W_el = self._model.electric_energy(cf, pr)
        sec = self._model.sec_MJ_kgCO2(pr)

        return {
            "CO2_recovery": float(self._model.CO2_recovery),
            "CO2_purity": float(self._model.CO2_purity),
            "co2_captured_tCO2_h": co2,
            "W_elec_GJ_h": W_el,
            "SEC_MJ_kgCO2": float(sec) if np.ndim(sec) == 0 else sec,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Membrane-Based CO2 Separation",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Pressure-driven membrane CO2 capture. CO2_recovery=0.80, purity=0.95. "
                "SEC=0.5-1.0 MJ/kgCO2 depending on pressure ratio."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "pressure_ratio": {"unit": "dimensionless", "range": [5.0, 20.0], "default": 10.0},
            },
            "outputs": {
                "CO2_recovery": {"unit": "dimensionless"},
                "CO2_purity": {"unit": "dimensionless"},
                "co2_captured_tCO2_h": {"unit": "tCO2/h"},
                "W_elec_GJ_h": {"unit": "GJ/h"},
                "SEC_MJ_kgCO2": {"unit": "MJ/kgCO2"},
            },
            "params": {
                "CO2_recovery": str(u["CO2_recovery"]["value"]),
                "CO2_purity": str(u["CO2_purity"]["value"]),
                "SEC_range": f"{u['SEC_min_MJ_kgCO2']['value']}-{u['SEC_max_MJ_kgCO2']['value']} MJ/kgCO2",
            },
            "source": "Baker (2002); Merkel et al. (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for pr in [5, 10, 15, 20]:
        r = model.predict({"capacity_fraction": 1.0, "pressure_ratio": pr})
        print(f"PR={pr}: SEC={r['SEC_MJ_kgCO2']:.3f} MJ/kgCO2  "
              f"CO2={float(r['co2_captured_tCO2_h']):.3f} tCO2/h  "
              f"W={float(r['W_elec_GJ_h']):.4f} GJ/h")
