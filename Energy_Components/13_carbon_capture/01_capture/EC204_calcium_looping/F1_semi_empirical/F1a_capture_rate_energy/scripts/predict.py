"""EC204 — Calcium Looping — F1a — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import CalciumLoopingF1a


class ComponentModel:
    component_id = "EC204"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CalciumLoopingF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            capacity_fraction : float or array [0.1, 1.0]
            cycle_number      : int or float [1, 500], default 1
        Returns:
            capture_rate, co2_captured_tCO2_h, Q_thermal_GJ_h,
            W_elec_GJ_h, SEC_thermal_GJ_tCO2, SEC_elec_GJ_tCO2, SEC_total_GJ_tCO2
        """
        cf = np.asarray(inputs.get("capacity_fraction", 1.0), dtype=float)
        N = inputs.get("cycle_number", 1)

        cr = self._model.capture_rate(N)
        co2 = self._model.co2_captured(cf, N)
        Q_th = self._model.thermal_energy(cf, N)
        W_el = self._model.electric_energy(cf, N)

        return {
            "capture_rate": float(cr) if np.ndim(cr) == 0 else cr,
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
            "name": "Calcium Looping CO2 Capture",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "CaO sorbent looping: carbonator at 650 C, calciner at 900 C. "
                "capture_rate=0.90 (fresh sorbent), decays with cycling. "
                "SEC_thermal=3.2 GJ/tCO2."
            ),
            "inputs": {
                "capacity_fraction": {"unit": "dimensionless", "range": [0.1, 1.0]},
                "cycle_number": {"unit": "dimensionless", "range": [1, 500], "default": 1},
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
                "capture_rate_0": str(u["capture_rate"]["value"]),
                "SEC_thermal": f"{u['SEC_thermal_GJ_tCO2']['value']} GJ/tCO2",
                "T_calciner": f"{u['T_calciner_C']['value']} C",
                "T_carbonator": f"{u['T_carbonator_C']['value']} C",
                "sorbent": u["sorbent"]["value"],
            },
            "source": "Abanades et al. (2004); Dean et al. (2011)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for N in [1, 50, 100, 200]:
        r = model.predict({"capacity_fraction": 1.0, "cycle_number": N})
        print(f"N={N:3d}: capture_rate={r['capture_rate']:.4f}  "
              f"CO2={float(r['co2_captured_tCO2_h']):.4f} tCO2/h")
