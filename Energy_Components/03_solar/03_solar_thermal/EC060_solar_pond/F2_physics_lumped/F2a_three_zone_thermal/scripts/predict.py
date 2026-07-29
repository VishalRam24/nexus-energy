"""
EC060 -- Solar Pond (Salinity-Gradient) -- F2a Three-Zone Lumped
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarPondF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the solar-pond F2a three-zone lumped model."""

    component_id = "EC060"
    component_name = "Solar Pond (Salinity-Gradient)"
    fidelity = "F2a -- Three-Zone Lumped Energy Balance (UCZ/NCZ/LCZ)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SolarPondF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic three-zone simulation.

        inputs:
            G : float or callable    surface irradiance (W/m2), default 250
            T_lcz_init : float       initial LCZ storage temp (degC), default 20
            T_ucz_init : float       initial UCZ temp (degC), default = T_amb
            T_amb : float or callable ambient temp (degC), default 20
            Q_extract_W : float      heat withdrawal from LCZ (W), default 0
            duration_days : float    horizon (days), default 10
            dt_hours : float         output step (hours), default 1
            diurnal : bool           synthesise day/night profile, default False
            G_peak : float           peak of diurnal profile (W/m2), default = G
        """
        G = inputs.get("G", 250.0)
        T_lcz0 = inputs.get("T_lcz_init", 20.0)
        T_ucz0 = inputs.get("T_ucz_init", None)
        T_amb = inputs.get("T_amb", 20.0)
        Q_ext = inputs.get("Q_extract_W", 0.0)
        dur = inputs.get("duration_days", 10.0)
        dt_h = inputs.get("dt_hours", 1.0)
        diurnal = inputs.get("diurnal", False)
        G_peak = inputs.get("G_peak", None)

        return self._model.simulate(
            G, T_lcz_init=T_lcz0, T_ucz_init=T_ucz0, T_amb=T_amb,
            Q_extract_W=Q_ext, duration_days=dur, dt_hours=dt_h,
            diurnal=diurnal, G_peak=G_peak,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "G": {"unit": "W/m2", "range": [0, 1100]},
                "T_lcz_init": {"unit": "degC", "range": [10, 95]},
                "T_ucz_init": {"unit": "degC", "range": [-10, 45]},
                "T_amb": {"unit": "degC", "range": [-10, 45]},
                "Q_extract_W": {"unit": "W", "range": [0, 5e6]},
                "duration_days": {"unit": "days", "range": [0.1, 365]},
                "dt_hours": {"unit": "hours", "range": [0.05, 6]},
                "diurnal": {"unit": "bool"},
            },
            "outputs": {
                "t_days": "days",
                "T_lcz": "degC (storage, can reach 70-90 C)",
                "T_ucz": "degC (surface zone)",
                "Q_solar_W": "W (solar reaching LCZ via Beer-Lambert)",
                "Q_ncz_W": "W (loss up through gradient zone)",
                "Q_ground_W": "W (loss to ground)",
                "Q_top_W": "W (surface loss)",
                "Q_extract_W": "W (extractable / extracted heat)",
                "f_lcz": "- (fraction of surface radiation reaching LCZ)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # 30-day charge-up from cold start under steady ~250 W/m2 daily-mean insolation
    r = m.predict({"G": 250.0, "T_lcz_init": 20.0, "T_amb": 20.0,
                   "duration_days": 30.0, "dt_hours": 6.0})
    print(f"f_LCZ (Beer-Lambert) = {r['f_lcz']:.4f}")
    print(f"LCZ start = {r['T_lcz'][0]:.1f} C  ->  LCZ after 30 d = {r['T_lcz'][-1]:.1f} C")
    print(f"UCZ final = {r['T_ucz'][-1]:.1f} C")
