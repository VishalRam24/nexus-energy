"""EC082 — Ice Thermal Storage — F1a Capacity Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import IceTESF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = IceTESF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        soc = np.asarray(inputs["soc"], dtype=float)
        q_c = np.asarray(inputs.get("q_charge",    0.0), dtype=float)
        q_d = np.asarray(inputs.get("q_discharge", 0.0), dtype=float)
        T_amb = np.asarray(
            inputs.get("t_ambient", self.params["unit"]["T_amb_default"]["value"]),
            dtype=float,
        )
        state = self._model.operating_state(soc, q_c, q_d, T_amb)
        return {
            "dSOC_dt":               self._model.dSOC_dt(soc, q_c, q_d, T_amb),
            "energy_stored_kwh":     self._model.energy_stored_kwh(soc),
            "max_charge_kw":         state["max_charge_w"]    / 1000.0,
            "max_discharge_kw":      state["max_discharge_w"] / 1000.0,
            "q_charge_effective_kw": state["q_charge_effective_w"]    / 1000.0,
            "q_discharge_effective_kw": state["q_discharge_effective_w"] / 1000.0,
            "heat_loss_w":           state["heat_loss_w"],
        }

    def get_info(self) -> dict:
        return {
            "name":        "Ice Thermal Storage (latent, water/ice)",
            "ec_id":       "EC082",
            "fidelity":    "F1a",
            "description": "Capacity model: SOC = m_ice/m_water; rates limited by SOC; charge/discharge at 0 C",
            "inputs": {
                "soc":         {"unit": "-",   "range": [0.0, 1.0]},
                "q_charge":    {"unit": "W",   "range": [0.0, 200000.0], "default": 0.0},
                "q_discharge": {"unit": "W",   "range": [0.0, 300000.0], "default": 0.0},
                "t_ambient":   {"unit": "degC","range": [-10.0, 40.0],   "default": 20.0},
            },
            "outputs": {
                "dSOC_dt":                  {"unit": "1/s"},
                "energy_stored_kwh":        {"unit": "kWh_th"},
                "max_charge_kw":            {"unit": "kW_th"},
                "max_discharge_kw":         {"unit": "kW_th"},
                "q_charge_effective_kw":    {"unit": "kW_th"},
                "q_discharge_effective_kw": {"unit": "kW_th"},
                "heat_loss_w":              {"unit": "W"},
            },
            "source":  "ASHRAE Handbook (2020) ch.51; Dincer & Rosen (2021)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"soc": 0.5, "q_charge": 80000.0, "q_discharge": 0.0, "t_ambient": 25.0})
    print(f"SOC=0.5, charging 80kW @ 25C: dSOC/dt={float(r['dSOC_dt'])*3600:.4f}/h, "
          f"E={float(r['energy_stored_kwh']):.1f}kWh, "
          f"Q_eff={float(r['q_charge_effective_kw']):.1f}kW, Q_loss={float(r['heat_loss_w']):.1f}W")
