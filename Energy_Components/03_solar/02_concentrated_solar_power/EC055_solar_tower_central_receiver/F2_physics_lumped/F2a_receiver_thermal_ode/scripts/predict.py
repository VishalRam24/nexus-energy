"""
EC055 -- Solar Tower / Central Receiver CSP -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarTowerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC055 F2a lumped central-receiver model."""

    component_id = "EC055"
    component_name = "Solar Tower Central Receiver (CSP)"
    fidelity = "F2a -- Physics-Lumped Heliostat Optics + Receiver Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SolarTowerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic central-receiver simulation.

        inputs:
            dni            : float|list|callable  DNI [W/m2]            (default 950)
            solar_zenith   : float|list|callable  zenith angle [deg]   (default 30)
            T_amb_C        : float|list|callable  ambient [degC]       (default 25)
            wind_speed     : float|list|callable  wind [m/s]           (default 5)
            mdot_salt      : float|list|callable  HTF flow [kg/s]      (default design)
            T_HTF_in_C     : float|list|callable  cold-salt inlet [degC] (default 290)
            T0_C           : float  initial receiver temperature [degC] (default 290)
            dt             : float  output step [s]                    (default 10)
            duration_s     : float  total time [s]                     (default 3600)
        """
        return self._model.simulate(
            dni=inputs.get("dni", 950.0),
            zenith_deg=inputs.get("solar_zenith", 30.0),
            T_amb_C=inputs.get("T_amb_C", 25.0),
            wind_speed=inputs.get("wind_speed", 5.0),
            mdot_salt=inputs.get("mdot_salt", None),
            T_HTF_in_C=inputs.get("T_HTF_in_C", 290.0),
            T0_C=inputs.get("T0_C", 290.0),
            dt=inputs.get("dt", 10.0),
            duration_s=inputs.get("duration_s", 3600.0),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "dni": {"unit": "W/m2", "range": [0, 1100]},
                "solar_zenith": {"unit": "deg", "range": [0, 85]},
                "T_amb_C": {"unit": "degC", "range": [-15, 55]},
                "wind_speed": {"unit": "m/s", "range": [0, 25]},
                "mdot_salt": {"unit": "kg/s", "range": [10, 600]},
                "T_HTF_in_C": {"unit": "degC", "range": [250, 350]},
                "T0_C": {"unit": "degC"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_receiver_C": "degC",
                "Q_field_W": "W (concentrated solar on aperture)",
                "Q_absorbed_W": "W",
                "Q_rad_loss_W": "W (∝ T^4)",
                "Q_conv_loss_W": "W",
                "Q_thermal_to_PB_MWth": "MW_th to power block",
                "P_electric_MWe": "MW_e gross",
                "field_efficiency": "-",
                "receiver_efficiency": "-",
                "overall_efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"dni": 950.0, "solar_zenith": 25.0, "duration_s": 1800.0, "dt": 30.0})
    print(f"Final T_receiver: {r['T_receiver_C'][-1]:.1f} degC, "
          f"Q_to_PB: {r['Q_thermal_to_PB_MWth'][-1]:.1f} MWth, "
          f"P_elec: {r['P_electric_MWe'][-1]:.1f} MWe, "
          f"eta_recv: {r['receiver_efficiency'][-1]:.3f}")
