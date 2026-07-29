"""
EC065 -- Offshore Fixed-Bottom Wind Turbine -- F2a BEM Steady -- predict interface.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OffshoreWindBEM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC065"
    component_name = "Offshore Fixed-Bottom Wind Turbine"
    fidelity = "F2a -- BEM Steady"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["turbine"].update(params)
        self._model = OffshoreWindBEM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a single BEM analysis.

        Parameters (inputs dict)
        ------------------------
        wind_speed_m_s : float
            Hub-height wind speed (m/s). Default 10.0.
        pitch_deg : float
            Blade pitch angle (deg). Default 0.0.
        rpm : float or None
            Rotor speed. If None, derived from TSR.
        tip_speed_ratio : float or None
            Override TSR.
        air_density : float or None
            Air density (kg/m3). Default from parameters.

        Returns
        -------
        dict with keys: power_kw, thrust_kN, torque_kNm, Cp, Ct,
                        omega_rad_s, rpm, blade_loads
        """
        V = inputs.get("wind_speed_m_s", 10.0)
        pitch = inputs.get("pitch_deg", 0.0)
        rpm = inputs.get("rpm", None)
        tsr = inputs.get("tip_speed_ratio", None)
        rho = inputs.get("air_density", None)
        return self._model.solve(V, pitch, rpm, tsr, rho)

    def predict_curve(self, wind_speeds=None, pitch_control=True):
        """
        Compute full power curve with pitch control.

        Parameters
        ----------
        wind_speeds : array-like or None
            Wind speeds (m/s). Default: 3-25 m/s in 0.5 steps.
        pitch_control : bool
            If True, pitch above rated to limit power.

        Returns
        -------
        dict of arrays: power_kw, thrust_kN, torque_kNm, Cp, Ct, pitch_deg
        """
        import numpy as np
        if wind_speeds is None:
            wind_speeds = np.arange(3.0, 25.5, 0.5)
        return self._model.power_curve(wind_speeds, pitch_control)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [3, 25]},
                "pitch_deg": {"unit": "deg", "default": 0.0},
                "rpm": {"unit": "rpm", "note": "Optional, derived from TSR if omitted"},
                "tip_speed_ratio": {"unit": "-"},
                "air_density": {"unit": "kg/m3", "default": 1.225},
            },
            "outputs": {
                "power_kw": "kW",
                "thrust_kN": "kN",
                "torque_kNm": "kNm",
                "Cp": "-",
                "Ct": "-",
                "blade_loads": "list of dicts per element",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"wind_speed_m_s": 11.4})
    print(f"P={r['power_kw']:.1f} kW, Cp={r['Cp']:.3f}, Ct={r['Ct']:.3f}")
    print(f"Thrust={r['thrust_kN']:.1f} kN, Torque={r['torque_kNm']:.1f} kNm")
