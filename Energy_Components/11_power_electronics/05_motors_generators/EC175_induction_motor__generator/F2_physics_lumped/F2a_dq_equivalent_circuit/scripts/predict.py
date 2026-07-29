"""EC175 -- Induction Motor -- F2a dq-Frame -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import InductionMotorF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = InductionMotorF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_supply_rms, frequency_hz, T_load_Nm, dt, duration_s
        returns:
            t, speed_rpm, torque, current, power, slip
        """
        return self._model.simulate(
            inputs["v_supply_rms"],
            inputs["frequency_hz"],
            inputs["T_load_Nm"],
            inputs.get("dt", 1e-4),
            inputs.get("duration_s", 1.0),
            x0=inputs.get("x0", None),
        )

    def get_info(self) -> dict:
        return {
            "name": "Induction Motor/Generator",
            "ec_id": "EC175",
            "fidelity": "F2a",
            "sub_fidelity": "dq_frame",
            "description": (
                "dq-frame dynamic model: 4 electrical ODEs + 1 mechanical ODE. "
                "Electromagnetic torque: T_e = 1.5*P*Lm*(i_qs*i_dr - i_ds*i_qr)"
            ),
            "inputs": {
                "v_supply_rms": {"unit": "V", "range": [100, 600]},
                "frequency_hz": {"unit": "Hz", "range": [10, 100]},
                "T_load_Nm": {"unit": "Nm", "range": [0, 100]},
                "dt": {"unit": "s", "default": 1e-4},
                "duration_s": {"unit": "s", "default": 1.0},
            },
            "outputs": {
                "t": {"unit": "s"},
                "speed_rpm": {"unit": "rpm"},
                "torque": {"unit": "Nm"},
                "current": {"unit": "A"},
                "power": {"unit": "W"},
                "slip": {"unit": "dimensionless"},
            },
            "source": "Boldea & Nasar (2010), The Induction Machine Handbook, CRC Press.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 30.0,
        "dt": 1e-4, "duration_s": 2.0,
    })
    print(f"Final: speed={r['speed_rpm'][-1]:.1f} rpm, "
          f"torque={r['torque'][-1]:.2f} Nm, slip={r['slip'][-1]:.4f}")
