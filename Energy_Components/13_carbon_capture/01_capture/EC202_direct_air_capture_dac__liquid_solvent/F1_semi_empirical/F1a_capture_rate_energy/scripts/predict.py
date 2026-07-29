"""EC202 — DAC Liquid Solvent — F1a Capture Rate & Energy — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import DACF1a


class ComponentModel:
    component_id = "EC202"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DACF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction : float or array [0.1, 1.0] — fraction of design capacity
        Returns:
            capture_rate, co2_captured_tCO2_h, Q_thermal_GJ_h, W_elec_GJ_h,
            SEC_thermal_GJ_tCO2, SEC_elec_GJ_tCO2, SEC_total_GJ_tCO2
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)

        co2 = self._model.co2_captured(cf)
        Q_th = self._model.thermal_energy(cf)
        W_el = self._model.electric_energy(cf)

        return {
            "capture_rate": float(self._model.capture_rate),
            "co2_captured_tCO2_h": co2,
            "Q_thermal_GJ_h": Q_th,
            "W_elec_GJ_h": W_el,
            "SEC_thermal_GJ_tCO2": float(self._model.SEC_thermal),
            "SEC_elec_GJ_tCO2": float(self._model.SEC_elec),
            "SEC_total_GJ_tCO2": float(self._model.sec_total()),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Direct Air Capture (DAC) Liquid Solvent",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "KOH liquid solvent DAC. capture_rate=0.90, SEC_thermal=6.0 GJ/tCO2, "
                "SEC_elec=0.5 GJ/tCO2. Calciner regeneration at 900 C."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
            },
            "outputs": {
                "capture_rate": {"unit": "dimensionless"},
                "co2_captured_tCO2_h": {"unit": "tCO2/h"},
                "Q_thermal_GJ_h": {"unit": "GJ/h"},
                "W_elec_GJ_h": {"unit": "GJ/h"},
                "SEC_thermal_GJ_tCO2": {"unit": "GJ/tCO2"},
                "SEC_elec_GJ_tCO2": {"unit": "GJ/tCO2"},
                "SEC_total_GJ_tCO2": {"unit": "GJ/tCO2"},
            },
            "params": {
                "capture_rate": str(u["capture_rate"]["value"]),
                "SEC_thermal": f"{u['SEC_thermal_GJ_tCO2']['value']} GJ/tCO2",
                "SEC_elec": f"{u['SEC_elec_GJ_tCO2']['value']} GJ/tCO2",
                "T_regen": f"{u['T_regen_C']['value']} C",
                "solvent": u["solvent"]["value"],
            },
            "source": "Keith et al. (2018) Joule; Fasihi et al. (2019)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for cf in [0.25, 0.5, 0.75, 1.0]:
        r = model.predict({"capacity_fraction": cf})
        print(f"CF={cf:.2f}: CO2={float(r['co2_captured_tCO2_h']):.3f} tCO2/h  "
              f"Q_th={float(r['Q_thermal_GJ_h']):.2f} GJ/h  "
              f"SEC_total={r['SEC_total_GJ_tCO2']:.1f} GJ/tCO2")
