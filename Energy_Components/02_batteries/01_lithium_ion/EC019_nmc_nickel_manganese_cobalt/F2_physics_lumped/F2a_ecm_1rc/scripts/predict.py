"""
EC019 -- NMC Battery -- F2a ECM 1-RC -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"current": [5.0]*3600, "dt": 1.0, "soc_init": 1.0})
"""

import json
import numpy as np
from pathlib import Path
from model import NMCBatteryECM1RC


class ComponentModel:
    """Standardized interface for EC019 NMC Battery -- F2a 1-RC ECM."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = NMCBatteryECM1RC(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run time-series ECM simulation.

        Args:
            inputs: {
                "current": float or array of current values (A, positive=discharge),
                "dt": float, time step in seconds (default 1.0),
                "soc_init": float, initial SOC (default 1.0)
            }

        Returns:
            {
                "voltage": array of terminal voltages (V),
                "soc": array of SOC values,
                "power": array of power values (W),
                "v_rc": array of RC polarization voltages (V),
                "time": array of time values (s)
            }
        """
        current = np.atleast_1d(np.asarray(inputs["current"], dtype=float))
        dt = float(inputs.get("dt", 1.0))
        soc_init = float(inputs.get("soc_init", 1.0))

        self._model.reset(soc_init)
        result = self._model.simulate(current, dt)
        return result

    def get_info(self) -> dict:
        return {
            "name": "NMC Battery (Nickel Manganese Cobalt)",
            "ec_id": "EC019",
            "fidelity": "F2a",
            "description": "1-RC Equivalent Circuit Model: V = OCV(SOC) - I*R0 - V_rc",
            "inputs": {
                "current": {"unit": "A", "note": "positive=discharge, array for time series"},
                "dt": {"unit": "s", "default": 1.0},
                "soc_init": {"unit": "dimensionless", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "voltage": {"unit": "V"},
                "soc": {"unit": "dimensionless"},
                "power": {"unit": "W"},
                "v_rc": {"unit": "V"},
                "time": {"unit": "s"},
            },
            "parameters": {
                "R0": f"{self.params['cell']['R0']['value']} Ohm (nominal)",
                "R1": f"{self.params['cell']['R1']['value']} Ohm (nominal)",
                "C1": f"{self.params['cell']['C1']['value']} F (nominal)",
                "tau1": f"{self.params['cell']['R1']['value'] * self.params['cell']['C1']['value']:.1f} s",
                "Q_nom": f"{self.params['cell']['capacity']['value']} Ah",
            },
            "source": self.params.get("source", ""),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))

    # Quick 1C discharge test
    Q = model.params["cell"]["capacity"]["value"]
    I_1C = Q  # 1C rate
    n_steps = int(3600 / 1.0)  # 1 hour at dt=1s
    current_profile = [I_1C] * n_steps
    result = model.predict({"current": current_profile, "dt": 1.0, "soc_init": 1.0})
    print(f"\n1C Discharge ({I_1C:.1f}A):")
    print(f"  Start voltage: {result['voltage'][0]:.4f} V")
    print(f"  End voltage:   {result['voltage'][-1]:.4f} V")
    print(f"  Final SOC:     {result['soc'][-1]:.4f}")
