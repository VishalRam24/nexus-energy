"""EC212 — MSF — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import MSFF1a


class ComponentModel:
    component_id = "EC212"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MSFF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction : float or array [0.1, 1.0]
            T_top_brine_C     : float [90, 120], default 110
        Returns:
            GOR, recovery, SEC_thermal_kJ_kg, SEC_elec_kWh_m3,
            distillate_flow_m3_h, Q_thermal_GJ_h, W_elec_kWh_h, steam_consumption_kg_h
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        T = inputs.get("T_top_brine_C", None)

        return {
            "GOR": float(self._model.GOR(T)) if (T is None or np.ndim(np.asarray(T)) == 0) else self._model.GOR(T),
            "recovery": float(self._model.recovery),
            "SEC_thermal_kJ_kg": float(self._model.SEC_thermal),
            "SEC_elec_kWh_m3": float(self._model.SEC_elec),
            "distillate_flow_m3_h": self._model.distillate_flow(cf),
            "Q_thermal_GJ_h": self._model.thermal_energy(cf, T),
            "W_elec_kWh_h": self._model.electric_power(cf),
            "steam_consumption_kg_h": self._model.steam_consumption(cf, T),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Multi-Stage Flash Distillation (MSF)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Thermal desalination: GOR=8, SEC_thermal=250 kJ/kg, SEC_elec=3.5 kWh/m3. "
                "T_top=110 C, 20 stages. Recovery=0.20 (typical seawater)."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "T_top_brine_C": {"unit": "degC", "range": [90, 120], "default": 110},
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
                "N_stages": str(u["N_stages"]["value"]),
                "T_top_brine": f"{u['T_top_brine_C']['value']} C",
            },
            "source": "El-Dessouky & Ettouney (2002)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T in [90, 100, 110, 120]:
        r = model.predict({"capacity_fraction": 1.0, "T_top_brine_C": T})
        print(f"T_top={T}C: GOR={r['GOR']:.2f}  Q_th={float(r['Q_thermal_GJ_h']):.2f} GJ/h  "
              f"W={float(r['W_elec_kWh_h']):.0f} kWh/h")
