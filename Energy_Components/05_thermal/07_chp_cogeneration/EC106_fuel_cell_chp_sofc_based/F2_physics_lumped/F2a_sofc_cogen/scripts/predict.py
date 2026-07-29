"""
EC106 -- Fuel Cell CHP (SOFC-Based) -- F2a SOFC Cogeneration
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SOFC_CHP_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC106 SOFC-CHP F2a physics-lumped model."""

    component_id = "EC106"
    component_name = "Fuel Cell CHP (SOFC-Based)"
    fidelity = "F2a -- SOFC Electrochemical Stack + Cogeneration Heat Recovery with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SOFC_CHP_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic SOFC-CHP simulation.

        inputs:
            current_density_A_cm2 : float (or callable for time-varying)
            T_cell_K   : float  (initial stack temperature, default 1073.15)
            dt         : float  (output time step [s], default 5.0)
            duration_s : float  (total duration [s], default 1200.0)
        """
        j = inputs.get("current_density_A_cm2", 0.5)
        T0 = inputs.get("T_cell_K", 1073.15)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 1200.0)
        return self._model.simulate(j, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [0, 2.5]},
                "T_cell_K": {"unit": "K", "range": [873.15, 1273.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "voltage": "V (per cell)",
                "P_e_W": "W (electrical)",
                "P_e_kW": "kW (electrical)",
                "Q_fuel_W": "W (LHV fuel input)",
                "Q_useful_thermal_W": "W (recovered heat)",
                "Q_useful_thermal_kW": "kW (recovered heat)",
                "Q_loss_W": "W (heat to ambient)",
                "eta_electrical": "-",
                "eta_thermal": "-",
                "eta_total": "- (CHP cogeneration efficiency)",
                "power_to_heat": "-",
                "E_nernst": "V",
                "steady_state": "dict of scalar final-step CHP metrics",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"current_density_A_cm2": 0.5, "T_cell_K": 1073.15,
                   "dt": 10.0, "duration_s": 600.0})
    ss = r["steady_state"]
    print(f"Steady state: T={ss['T_K']:.1f} K, P_e={ss['P_e_kW']:.3f} kW, "
          f"Q_th={ss['Q_useful_thermal_kW']:.3f} kW")
    print(f"  eta_e={ss['eta_electrical']:.4f}, eta_th={ss['eta_thermal']:.4f}, "
          f"eta_total={ss['eta_total']:.4f}, P/H={ss['power_to_heat']:.3f}")
