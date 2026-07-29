"""EC138 — OTEC — F1a — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import OTECF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OTECF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          T_warm        : warm surface water temperature [degC]
          T_cold        : cold deep water temperature [degC]
          Q_thermal_kw  : thermal input [kW] (optional)
        returns:
          eta_carnot, eta_gross, eta_net, P_gross_kw, P_net_kw, P_parasitic_kw
        """
        T_warm       = np.asarray(inputs["T_warm"], dtype=float)
        T_cold       = np.asarray(inputs["T_cold"], dtype=float)
        Q_thermal_kw = inputs.get("Q_thermal_kw", None)

        flows = self._model.power_flows(T_warm, T_cold, Q_thermal_kw)
        return {
            "eta_carnot":     flows["eta_carnot"],
            "eta_gross":      flows["eta_gross"],
            "eta_net":        flows["eta_net"],
            "P_gross_kw":     flows["P_gross_kw"],
            "P_net_kw":       flows["P_net_kw"],
            "P_parasitic_kw": flows["P_parasitic_kw"],
        }

    def get_info(self) -> dict:
        return {
            "name":        "Ocean Thermal Energy Conversion (OTEC)",
            "ec_id":       "EC138",
            "fidelity":    "F1a",
            "description": "Closed-cycle ammonia ORC; eta_net = eta_Carnot * eta_cycle_frac * (1 - parasitic_frac)",
            "inputs": {
                "T_warm":       {"unit": "degC", "range": [20.0, 32.0]},
                "T_cold":       {"unit": "degC", "range": [2.0,  10.0]},
                "Q_thermal_kw": {"unit": "kW",   "range": [0.0, None], "default": "from rated"},
            },
            "outputs": {
                "eta_carnot":     {"unit": "-"},
                "eta_gross":      {"unit": "-"},
                "eta_net":        {"unit": "-"},
                "P_gross_kw":     {"unit": "kW"},
                "P_net_kw":       {"unit": "kW"},
                "P_parasitic_kw": {"unit": "kW"},
            },
            "source": "Vega (2002) Mar. Technol. Soc. J.; Nihous (2007) J. Energy Resour. Technol.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"T_warm": 26.0, "T_cold": 5.0})
    print(f"\nT_warm=26°C, T_cold=5°C:")
    print(f"  eta_Carnot={float(r['eta_carnot'])*100:.2f}%, "
          f"eta_gross={float(r['eta_gross'])*100:.2f}%, "
          f"eta_net={float(r['eta_net'])*100:.2f}%")
    print(f"  P_gross={float(r['P_gross_kw']):.1f} kW, P_net={float(r['P_net_kw']):.1f} kW")
