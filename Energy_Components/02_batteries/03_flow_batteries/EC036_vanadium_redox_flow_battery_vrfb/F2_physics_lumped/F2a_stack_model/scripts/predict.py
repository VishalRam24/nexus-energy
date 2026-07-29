"""EC036 -- VRFB -- F2a Stack Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import VRFBF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = VRFBF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of VRFB stack + tank.

        inputs:
            current_A:        float or callable(t) [A]
            flow_rate_L_min:  float or callable(t) [L/min]
            dt:               float [s] (default 1.0)
            duration_s:       float [s] (default 3600)
            soc_init:         float [0-1] (default 0.5)

        returns:
            t, voltage, soc, power_stack, power_pump, net_power, efficiency
        """
        current_A = inputs["current_A"]
        flow_rate = inputs.get("flow_rate_L_min", 10.0)
        dt = inputs.get("dt", 1.0)
        duration_s = inputs.get("duration_s", 3600.0)
        soc_init = inputs.get("soc_init", 0.5)

        return self._model.simulate(current_A, flow_rate, dt, duration_s, soc_init)

    def predict_steady_state(self, inputs: dict) -> dict:
        """Return instantaneous cell/stack voltage for given SOC, I, Q."""
        soc = inputs["soc"]
        I = inputs["current_A"]
        Q = inputs.get("flow_rate_L_min", 10.0)
        V_cell = self._model.cell_voltage(soc, I, Q)
        V_stack = self._model.stack_voltage(soc, I, Q)
        P_pump = self._model.pump_power_w(Q)
        return {
            "cell_voltage": float(V_cell),
            "stack_voltage": float(V_stack),
            "pump_power_w": float(P_pump),
        }

    def get_info(self) -> dict:
        return {
            "name": "Vanadium Redox Flow Battery (VRFB)",
            "ec_id": "EC036",
            "fidelity": "F2a",
            "sub_fidelity": "stack_model",
            "description": (
                "Dynamic VRFB stack model: E_cell = E_nernst - eta_act - eta_ohm - eta_conc. "
                "Coupled tank SOC dynamics via dSOC/dt = -I/(n*F*c*V). "
                "Pump hydraulic losses included."
            ),
            "inputs": {
                "current_A": {"unit": "A", "range": [-120, 120]},
                "flow_rate_L_min": {"unit": "L/min", "range": [1, 30]},
                "dt": {"unit": "s", "default": 1.0},
                "duration_s": {"unit": "s", "default": 3600},
                "soc_init": {"unit": "dimensionless", "default": 0.5},
            },
            "outputs": {
                "t": {"unit": "s"},
                "voltage": {"unit": "V"},
                "soc": {"unit": "dimensionless"},
                "power_stack": {"unit": "W"},
                "power_pump": {"unit": "W"},
                "net_power": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
            },
            "source": "Blanc & Rufer (2010); Shah et al. (2011)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 10.0, "duration_s": 600.0, "soc_init": 0.8,
    })
    print(f"Final SOC: {r['soc'][-1]:.4f}")
    print(f"Final voltage: {r['voltage'][-1]:.2f} V")
    print(f"Final net power: {r['net_power'][-1]:.1f} W")
