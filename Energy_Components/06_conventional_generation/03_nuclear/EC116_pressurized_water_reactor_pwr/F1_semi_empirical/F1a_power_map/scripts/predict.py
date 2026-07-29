"""EC116 — PWR — F1a Steady-State Power Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PWRF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PWRF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio   : float or array, PLR [0.5–1.0]
            coolant_flow_kgs  : float or array, primary coolant flow [kg/s] (optional)
        returns:
            electric_power_mw : net electrical output [MW_e]
            thermal_power_mw  : reactor thermal power [MW_th]
            efficiency        : net efficiency P_e/P_th [-]
            coolant_outlet_temp_c : hot-leg temperature [degC]
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
            "name": "Pressurized Water Reactor (PWR)",
            "ec_id": "EC116",
            "fidelity": "F1a",
            "model": "Steady-State Power Map",
            "description": "P_electric = P_thermal * eta_cycle * eta_gen * f_PLR",
            "inputs": {
                "part_load_ratio": {"unit": "dimensionless", "range": [0.5, 1.0]},
                "coolant_flow_kgs": {"unit": "kg/s", "range": [9000.0, 18000.0], "default": 18000.0},
            },
            "outputs": {
                "electric_power_mw": {"unit": "MW_e"},
                "thermal_power_mw": {"unit": "MW_th"},
                "efficiency": {"unit": "dimensionless"},
                "coolant_outlet_temp_c": {"unit": "degC"},
            },
            "source": "Todreas & Kazimi (2012), Nuclear Systems, 2nd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0})
    print(f"At full power (PLR=1.0):")
    print(f"  Thermal power : {float(r['thermal_power_mw']):.0f} MW_th")
    print(f"  Electric power: {float(r['electric_power_mw']):.0f} MW_e")
    print(f"  Efficiency    : {float(r['efficiency'])*100:.1f} %")
    print(f"  Coolant T_out : {float(r['coolant_outlet_temp_c']):.1f} degC")
