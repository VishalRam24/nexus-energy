"""EC010 -- SOEC -- F2a Electrochemical -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SOECF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SOECF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        j = inputs["current_density"]
        T = inputs.get("T_K", 1073.15)
        U = inputs.get("steam_utilization", 0.5)
        dt = inputs.get("dt", 1.0)
        duration_s = inputs.get("duration_s", 3600.0)
        return self._model.simulate(j, T, U, dt, duration_s)

    def predict_steady_state(self, inputs: dict) -> dict:
        j = inputs["current_density"]
        T = inputs.get("T_K", 1073.15)
        U = inputs.get("steam_utilization", 0.5)
        return {
            "cell_voltage": float(self._model.cell_voltage(j, T, U)),
            "stack_voltage": float(self._model.stack_voltage(j, T, U)),
            "h2_rate_mol_s": float(self._model.h2_production_rate(j)),
            "efficiency": float(self._model.efficiency(j, T, U)),
            "thermal_mode": int(self._model.thermal_mode(j, T, U)),
        }

    def get_info(self) -> dict:
        return {
            "name": "Solid Oxide Electrolyser Cell (SOEC)",
            "ec_id": "EC010",
            "fidelity": "F2a",
            "sub_fidelity": "electrochemical",
            "description": (
                "Physics-lumped SOEC model: Nernst (with steam utilization) + "
                "Butler-Volmer activation + YSZ ohmic. Tracks endothermic/exothermic mode."
            ),
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.01, 2.0]},
                "T_K": {"unit": "K", "range": [923, 1173], "default": 1073.15},
                "steam_utilization": {"unit": "dimensionless", "range": [0.1, 0.9], "default": 0.5},
            },
            "outputs": {
                "t": {"unit": "s"},
                "voltage": {"unit": "V"},
                "h2_production": {"unit": "mol/s"},
                "efficiency": {"unit": "dimensionless"},
                "thermal_mode": {"unit": "1=endothermic, -1=exothermic"},
            },
            "source": "Ni et al. (2007); Udagawa et al. (2007)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict_steady_state({"current_density": 0.5, "T_K": 1073.15, "steam_utilization": 0.5})
    print(f"Cell voltage: {r['cell_voltage']:.4f} V")
    print(f"Efficiency: {r['efficiency']:.4f}")
    print(f"Thermal mode: {'endothermic' if r['thermal_mode'] > 0 else 'exothermic'}")
