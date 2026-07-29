"""EC160 -- Isolated DC-DC Flyback -- F2a Averaged SSM -- Standardized Predict Interface"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import FlybackConverterF2a


class ComponentModel:
    component_id = "EC160"
    component_name = "Isolated DC-DC Converter (Flyback)"
    fidelity = "F2a -- Averaged State-Space Model with Parasitic Losses"
    version = "1.0.0"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FlybackConverterF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of the averaged flyback converter.

        inputs:
            v_in        : float or callable(t) [V]
            duty_cycle  : float or callable(t) [0-1]
            R_load      : float or callable(t) [Ohm]
            dt          : float [s]  (default 1e-6)
            duration_s  : float [s]  (default 0.01)
            x0          : optional [i_m_0, v_C_0]

        returns:
            t, v_out, i_m, i_out, power (all arrays)
        """
        v_in = inputs["v_in"]
        duty = inputs["duty_cycle"]
        R_load = inputs["R_load"]
        dt = inputs.get("dt", 1e-6)
        duration_s = inputs.get("duration_s", 0.01)
        x0 = inputs.get("x0", None)
        return self._model.simulate(v_in, duty, R_load, dt, duration_s, x0=x0)

    def predict_steady_state(self, inputs: dict) -> dict:
        """Return analytic averaged steady-state values + ideal gain."""
        return self._model.steady_state(
            inputs["v_in"], inputs["duty_cycle"], inputs["R_load"]
        )

    def predict_efficiency(self, inputs: dict) -> float:
        """Return steady-state efficiency in (0, 1)."""
        return self._model.efficiency(
            inputs["v_in"], inputs["duty_cycle"], inputs["R_load"]
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "sub_fidelity": "averaged_state_space",
            "version": self.version,
            "description": (
                "Averaged state-space ODE model of an isolated flyback. "
                "States: [i_m, v_C]. "
                "Lm*di_m/dt = d*V_in - (1-d)*n*(v_C+V_f) - i_m*R_series; "
                "C*dv_C/dt = (1-d)*n*i_m - v_C/R_load. "
                "Ideal gain V_out = d/(1-d) * V_in/n. Galvanic isolation via n."
            ),
            "inputs": {
                "v_in": {"unit": "V", "range": [20.0, 100.0]},
                "duty_cycle": {"unit": "dimensionless", "range": [0.05, 0.90]},
                "R_load": {"unit": "Ohm", "range": [0.5, 100.0]},
                "dt": {"unit": "s", "default": 1e-6},
                "duration_s": {"unit": "s", "default": 0.01},
            },
            "outputs": {
                "t": {"unit": "s"},
                "v_out": {"unit": "V"},
                "i_m": {"unit": "A", "note": "primary-referred magnetizing current"},
                "i_out": {"unit": "A"},
                "power": {"unit": "W"},
            },
            "source": "Erickson & Maksimovic (2020), Fundamentals of Power Electronics, 3rd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    inp = {"v_in": 48.0, "duty_cycle": 0.5, "R_load": 1.2}
    ss = model.predict_steady_state(inp)
    eta = model.predict_efficiency(inp)
    r = model.predict({**inp, "dt": 1e-6, "duration_s": 0.01})
    print(f"Ideal gain V_out = {ss['v_out_ideal']:.3f} V")
    print(f"Steady-state: V_out={ss['v_out_ss']:.3f} V  i_m={ss['i_m_ss']:.3f} A  eta={eta:.4f}")
    print(f"Simulated final: V_out={r['v_out'][-1]:.3f} V  i_m={r['i_m'][-1]:.3f} A")
