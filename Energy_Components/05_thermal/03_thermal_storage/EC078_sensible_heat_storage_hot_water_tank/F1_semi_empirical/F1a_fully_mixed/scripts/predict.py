"""EC078 — Hot Water Tank TES — F1a Fully Mixed — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HotWaterTankF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HotWaterTankF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        T            = np.asarray(inputs["temperature"],  dtype=float)
        q_charge     = np.asarray(inputs.get("q_charge",    0.0), dtype=float)
        q_discharge  = np.asarray(inputs.get("q_discharge", 0.0), dtype=float)
        T_amb        = np.asarray(
            inputs.get("t_ambient", self.params["unit"]["T_amb_default"]["value"]),
            dtype=float,
        )
        return {
            "dT_dt":             self._model.dT_dt(T, q_charge, q_discharge, T_amb),
            "energy_stored_kwh": self._model.energy_stored_kwh(T),
            "soc":               self._model.soc(T),
            "heat_loss_w":       self._model.heat_loss(T, T_amb),
        }

    def get_info(self) -> dict:
        return {
            "name":        "Sensible Heat Storage — Hot Water Tank (fully mixed)",
            "ec_id":       "EC078",
            "fidelity":    "F1a",
            "description": "dT/dt = (Q_charge - Q_discharge - UA*(T-T_amb)) / (m*cp); soc = (T-T_min)/(T_max-T_min)",
            "inputs": {
                "temperature":  {"unit": "degC",  "range": [0.0, 100.0]},
                "q_charge":     {"unit": "W",     "range": [0.0, 50000.0], "default": 0.0},
                "q_discharge":  {"unit": "W",     "range": [0.0, 50000.0], "default": 0.0},
                "t_ambient":    {"unit": "degC",  "range": [-10.0, 40.0],  "default": 20.0},
            },
            "outputs": {
                "dT_dt":             {"unit": "K/s"},
                "energy_stored_kwh": {"unit": "kWh"},
                "soc":               {"unit": "dimensionless"},
                "heat_loss_w":       {"unit": "W"},
            },
            "source":  "Duffie & Beckman (2013), Solar Engineering of Thermal Processes, ch.8",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"temperature": 60.0, "q_charge": 5000.0, "q_discharge": 3000.0, "t_ambient": 20.0})
    print(f"T=60C, Q_in=5kW, Q_out=3kW: "
          f"dT/dt={float(r['dT_dt'])*1000:.4f} mK/s, "
          f"E={float(r['energy_stored_kwh']):.2f} kWh, "
          f"SOC={float(r['soc']):.2f}, "
          f"Q_loss={float(r['heat_loss_w']):.1f} W")
