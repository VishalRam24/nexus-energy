"""EC117 — BWR — F1a Steady-State Power Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BWRF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BWRF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio : float or array — PLR [0.6–1.0]
        returns:
            electric_power_mw  : MW_e
            thermal_power_mw   : MW_th
            efficiency         : dimensionless
            steam_mass_flow_kgs: kg/s
        """
        PLR = np.asarray(inputs["part_load_ratio"], dtype=float)
        return {
            "electric_power_mw": self._model.electric_power(PLR),
            "thermal_power_mw": self._model.thermal_power(PLR),
            "efficiency": self._model.efficiency(PLR),
            "steam_mass_flow_kgs": self._model.steam_mass_flow(PLR),
        }

    def get_info(self) -> dict:
        return {
            "name": "Boiling Water Reactor (BWR)",
            "ec_id": "EC117",
            "fidelity": "F1a",
            "model": "Steady-State Power Map",
            "description": "Direct-cycle BWR — no SG; P_e = P_th * eta_cycle * eta_gen * f_PLR",
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.6, 1.0]},
            },
            "outputs": {
                "electric_power_mw":  {"unit": "MW_e"},
                "thermal_power_mw":   {"unit": "MW_th"},
                "efficiency":         {"unit": "-"},
                "steam_mass_flow_kgs":{"unit": "kg/s"},
            },
            "source": "Todreas & Kazimi (2012); GE BWR/6 reference plant",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0})
    print(f"Full power: P_th={float(r['thermal_power_mw']):.0f} MW, "
          f"P_e={float(r['electric_power_mw']):.0f} MW, "
          f"eta={float(r['efficiency'])*100:.1f} %, "
          f"m_steam={float(r['steam_mass_flow_kgs']):.0f} kg/s")
