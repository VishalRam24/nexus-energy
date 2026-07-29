"""EC219 — Piezoelectric Harvester — F1b Coupling+Damping — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import PiezoF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PiezoF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict piezoelectric harvester power with coupled electromechanical model.

        Parameters
        ----------
        inputs : dict
            acceleration_ms2 : float or array (m/s^2, 0.1-100)
            frequency_hz     : float or array (Hz, 10-1000)
            R_load_ohm       : float or array (ohm, 100-1e7)

        Returns
        -------
        dict with power_w, power_uw, voltage_v, frequency_ratio,
                  zeta_electrical, zeta_total, optimal_R_ohm, at_resonance_power_w
        """
        a = inputs.get("acceleration_ms2", 9.81)
        f = inputs.get("frequency_hz", self._model.f_n)
        R = inputs.get("R_load_ohm", 1.0 / (2.0 * np.pi * self._model.f_n * self._model.C_p))
        return self._model.compute(a, f, R)

    def get_info(self) -> dict:
        return {
            "name": "Piezoelectric Energy Harvester",
            "ec_id": "EC219",
            "fidelity": "F1b",
            "description": (
                "PZT-5A bimorph with coupled electromechanical frequency-domain model. "
                "Includes: k31 coupling coefficient, mechanical and electrical damping, "
                "optimal load analysis, and full frequency response via complex admittance."
            ),
            "inputs": {
                "acceleration_ms2": {"unit": "m/s^2", "range": [0.1, 100], "default": 9.81},
                "frequency_hz": {"unit": "Hz", "range": [10, 1000], "default": 100.0},
                "R_load_ohm": {"unit": "ohm", "range": [100, 1e7], "default": "optimal"},
            },
            "outputs": {
                "power_w": {"unit": "W"},
                "power_uw": {"unit": "uW"},
                "voltage_v": {"unit": "V"},
                "frequency_ratio": {"unit": "-"},
                "zeta_electrical": {"unit": "-"},
                "zeta_total": {"unit": "-"},
                "optimal_R_ohm": {"unit": "ohm"},
                "at_resonance_power_w": {"unit": "W"},
            },
            "source": "Erturk & Inman (2011); duToit et al. (2005); Roundy (2005)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    f_n = model._model.f_n
    R_opt = model._model.compute(9.81, f_n, 1.0 / (2.0 * np.pi * f_n * model._model.C_p))["optimal_R_ohm"]
    r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": R_opt})
    print(f"Design point (a=1g, f=fn={f_n:.0f}Hz, R_opt={R_opt:.0f} ohm):")
    for k, v in r.items():
        val = v if isinstance(v, (int, float)) else float(np.atleast_1d(v)[0])
        print(f"  {k} = {val:.4e}")
