"""EC158 -- Boost Converter -- F2a Averaged SSM -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BoostConverterF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BoostConverterF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of averaged boost converter.

        inputs:
            v_in, duty_cycle, R_load, dt, duration_s
        returns:
            t, v_out, i_L, i_out, power
        """
        v_in = inputs["v_in"]
        duty = inputs["duty_cycle"]
        R_load = inputs["R_load"]
        dt = inputs.get("dt", 1e-6)
        duration_s = inputs.get("duration_s", 0.01)
        x0 = inputs.get("x0", None)

        return self._model.simulate(v_in, duty, R_load, dt, duration_s, x0=x0)

    def predict_steady_state(self, inputs: dict) -> dict:
        return self._model.steady_state(
            inputs["v_in"], inputs["duty_cycle"], inputs["R_load"]
        )

    def get_info(self) -> dict:
        return {
            "name": "Boost Converter (Step-Up)",
            "ec_id": "EC158",
            "fidelity": "F2a",
            "sub_fidelity": "averaged_state_space",
            "description": (
                "Averaged state-space ODE model. "
                "States: [i_L, v_C]. "
                "di_L/dt = (V_in - (1-D)*v_C - i_L*R_L)/L, "
                "dv_C/dt = ((1-D)*i_L - v_C/R_load)/C"
            ),
            "inputs": {
                "v_in": {"unit": "V", "range": [5.0, 60.0]},
                "duty_cycle": {"unit": "dimensionless", "range": [0.05, 0.90]},
                "R_load": {"unit": "Ohm", "range": [5.0, 500.0]},
                "dt": {"unit": "s", "default": 1e-6},
                "duration_s": {"unit": "s", "default": 0.01},
            },
            "outputs": {
                "t": {"unit": "s"},
                "v_out": {"unit": "V"},
                "i_L": {"unit": "A"},
                "i_out": {"unit": "A"},
                "power": {"unit": "W"},
            },
            "source": "Erickson & Maksimovic (2020), Fundamentals of Power Electronics, 3rd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0})
    r = model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.05,
    })
    print(f"Steady-state: V_out={ss['v_out_ss']:.3f}V  I_L={ss['i_L_ss']:.3f}A")
    print(f"Simulated final: V_out={r['v_out'][-1]:.3f}V  I_L={r['i_L'][-1]:.3f}A")
