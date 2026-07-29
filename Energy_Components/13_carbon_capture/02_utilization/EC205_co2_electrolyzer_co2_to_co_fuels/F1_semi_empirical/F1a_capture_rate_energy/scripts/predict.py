"""EC205 — CO2 Electrolyzer — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2ElectrolyzerF1a


class ComponentModel:
    component_id = "EC205"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2ElectrolyzerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction     : float or array [0.1, 1.0]
            current_density_mA_cm2: float [50, 400], default 200
        Returns:
            faradaic_efficiency, co2_converted_kg_h, co_produced_kg_h,
            W_elec_kWh_h, SEC_kWh_kgCO2
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        j = inputs.get("current_density_mA_cm2", None)

        co2 = self._model.co2_converted(cf)
        co = self._model.co_produced(cf)
        W_el = self._model.electric_energy(cf, j)
        sec = self._model.sec_kWh_kgCO2(j)

        return {
            "faradaic_efficiency": float(self._model.FE),
            "co2_converted_kg_h": co2,
            "co_produced_kg_h": co,
            "W_elec_kWh_h": W_el,
            "SEC_kWh_kgCO2": float(sec) if np.ndim(sec) == 0 else sec,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "CO2 Electrolyzer (CO2 to CO/Fuels)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "CO2RR to CO. Faradaic_eff=0.85, V_cell=3.0 V, SEC=8 kWh/kgCO2. "
                "Products: CO (syngas). n_e=2 per CO2."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "current_density_mA_cm2": {"unit": "mA/cm2", "range": [50, 400], "default": 200},
            },
            "outputs": {
                "faradaic_efficiency": {"unit": "dimensionless"},
                "co2_converted_kg_h": {"unit": "kg/h"},
                "co_produced_kg_h": {"unit": "kg/h"},
                "W_elec_kWh_h": {"unit": "kWh/h"},
                "SEC_kWh_kgCO2": {"unit": "kWh/kgCO2"},
            },
            "params": {
                "faradaic_efficiency": str(u["faradaic_efficiency"]["value"]),
                "V_cell": f"{u['V_cell']['value']} V",
                "SEC": f"{u['SEC_kWh_kgCO2']['value']} kWh/kgCO2",
                "product": u["product"]["value"],
            },
            "source": "Jouny et al. (2018); Bushuyev et al. (2018)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for j in [100, 200, 300]:
        r = model.predict({"capacity_fraction": 1.0, "current_density_mA_cm2": j})
        print(f"j={j} mA/cm2: SEC={r['SEC_kWh_kgCO2']:.2f} kWh/kgCO2  "
              f"CO={float(r['co_produced_kg_h']):.1f} kg/h  "
              f"W={float(r['W_elec_kWh_h']):.0f} kWh/h")
