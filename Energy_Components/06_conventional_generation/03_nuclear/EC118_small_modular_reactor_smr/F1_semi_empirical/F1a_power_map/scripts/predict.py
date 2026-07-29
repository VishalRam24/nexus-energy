"""EC118 — SMR — F1a Steady-State Power Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SMRF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SMRF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio  : float or array — PLR [0.2–1.0]
            coolant_flow_kgs : float or array — kg/s (optional)
        returns:
            electric_power_mw    : MW_e
            thermal_power_mw     : MW_th
            efficiency           : -
            coolant_outlet_temp_c: degC
        """
        PLR = np.asarray(inputs["part_load_ratio"], dtype=float)
        m_dot = inputs.get("coolant_flow_kgs", None)
        return {
            "electric_power_mw": self._model.electric_power(PLR),
            "thermal_power_mw": self._model.thermal_power(PLR),
            "efficiency": self._model.efficiency(PLR),
            "coolant_outlet_temp_c": self._model.coolant_outlet_temp(PLR, m_dot),
        }

    def get_info(self) -> dict:
        return {
            "name": "Small Modular Reactor (SMR)",
            "ec_id": "EC118",
            "fidelity": "F1a",
            "model": "Steady-State Power Map",
            "description": "Integral PWR SMR with extended load-following (PLR_min ~ 0.2)",
            "inputs": {
                "part_load_ratio":  {"unit": "-", "range": [0.2, 1.0]},
                "coolant_flow_kgs": {"unit": "kg/s", "range": [800.0, 1900.0], "default": 1900.0},
            },
            "outputs": {
                "electric_power_mw":     {"unit": "MW_e"},
                "thermal_power_mw":      {"unit": "MW_th"},
                "efficiency":            {"unit": "-"},
                "coolant_outlet_temp_c": {"unit": "degC"},
            },
            "source": "IAEA SMR Booklet (2022); NuScale VOYGR design documents",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0})
    print(f"Full power: P_th={float(r['thermal_power_mw']):.0f} MW, "
          f"P_e={float(r['electric_power_mw']):.0f} MW, "
          f"eta={float(r['efficiency'])*100:.1f} %, "
          f"T_out={float(r['coolant_outlet_temp_c']):.1f} degC")
