"""EC213 — MED — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import MEDF1a


class ComponentModel:
    component_id = "EC213"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MEDF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction : float or array [0.1, 1.0]
            N_effects         : int [4, 20], default 12
        Returns:
            GOR, recovery, SEC_thermal_kJ_kg, SEC_elec_kWh_m3,
            distillate_flow_m3_h, Q_thermal_GJ_h, W_elec_kWh_h, steam_consumption_kg_h
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        N = inputs.get("N_effects", None)

        return {
            "GOR": float(self._model.GOR(N)) if (N is None or np.ndim(np.asarray(N)) == 0) else self._model.GOR(N),
            "recovery": float(self._model.recovery),
            "SEC_thermal_kJ_kg": float(self._model.SEC_thermal),
            "SEC_elec_kWh_m3": float(self._model.SEC_elec),
            "distillate_flow_m3_h": self._model.distillate_flow(cf),
            "Q_thermal_GJ_h": self._model.thermal_energy(cf),
            "W_elec_kWh_h": self._model.electric_power(cf),
            "steam_consumption_kg_h": self._model.steam_consumption(cf, N),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Multi-Effect Distillation (MED)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Thermal desalination: GOR=10, SEC_thermal=200 kJ/kg, SEC_elec=1.5 kWh/m3. "
                "T_top=70 C, 12 effects. Recovery=0.35."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "N_effects": {"unit": "dimensionless", "range": [4, 20], "default": 12},
            },
            "outputs": {
                "GOR": {"unit": "dimensionless"},
                "recovery": {"unit": "dimensionless"},
                "SEC_thermal_kJ_kg": {"unit": "kJ/kg"},
                "SEC_elec_kWh_m3": {"unit": "kWh/m3"},
                "distillate_flow_m3_h": {"unit": "m3/h"},
                "Q_thermal_GJ_h": {"unit": "GJ/h"},
                "W_elec_kWh_h": {"unit": "kWh/h"},
                "steam_consumption_kg_h": {"unit": "kg/h"},
            },
            "params": {
                "GOR": str(u["GOR"]["value"]),
                "SEC_thermal": f"{u['SEC_thermal_kJ_kg']['value']} kJ/kg",
                "SEC_elec": f"{u['SEC_elec_kWh_m3']['value']} kWh/m3",
                "N_effects": str(u["N_effects"]["value"]),
                "T_top": f"{u['T_top_C']['value']} C",
            },
            "source": "El-Dessouky & Ettouney (2002); Al-Sahali & Ettouney (2007)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for N in [6, 8, 12, 16]:
        r = model.predict({"capacity_fraction": 1.0, "N_effects": N})
        print(f"N={N:2d}: GOR={r['GOR']:.2f}  Q_th={float(r['Q_thermal_GJ_h']):.2f} GJ/h  "
              f"W={float(r['W_elec_kWh_h']):.0f} kWh/h")
