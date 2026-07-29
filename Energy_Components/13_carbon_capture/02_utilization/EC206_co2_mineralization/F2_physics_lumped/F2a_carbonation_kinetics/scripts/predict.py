"""
EC206 -- CO2 Mineralization (Mineral Carbonation) -- F2a Carbonation Kinetics
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2Mineralization_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC206 CO2-mineralization F2a kinetic model."""

    component_id = "EC206"
    component_name = "CO2 Mineralization (Mineral Carbonation)"
    fidelity = "F2a -- Carbonation Kinetics + Reactor Energy Balance (lumped ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CO2Mineralization_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run batch carbonation simulation.

        inputs:
            T0_K            : float  initial slurry temperature (default 458.15 = 185 C)
            P_CO2_atm       : float  CO2 partial pressure (default 115)
            particle_radius_m : float  grain radius (default 37 um)
            dt              : float  output step [s] (default 30)
            duration_s      : float  batch time [s] (default 3600)

        returns dict with time-series:
            t, conversion, temperature, co2_bound_kg, carbonate_kg,
            heat_released_J, co2_rate_mol_s
            + scalar summary: final_conversion, co2_stored_kg, peak_temperature_K
        """
        T0 = inputs.get("T0_K", 458.15)
        P_CO2 = inputs.get("P_CO2_atm", 115.0)
        R0 = inputs.get("particle_radius_m", self._model.R0_ref)
        dt = inputs.get("dt", 30.0)
        dur = inputs.get("duration_s", 3600.0)

        r = self._model.simulate(T0, P_CO2, R0, dt, dur)
        r["final_conversion"] = float(r["conversion"][-1])
        r["co2_stored_kg"] = float(r["co2_bound_kg"][-1])
        r["peak_temperature_K"] = float(r["temperature"].max())
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T0_K": {"unit": "K", "range": [298.15, 523.15]},
                "P_CO2_atm": {"unit": "atm", "range": [1.0, 200.0]},
                "particle_radius_m": {"unit": "m", "range": [5.0e-6, 1.0e-3]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "conversion": "- (fraction of mineral carbonated)",
                "temperature": "K",
                "co2_bound_kg": "kg (CO2 permanently mineralized)",
                "carbonate_kg": "kg (MgCO3 produced)",
                "heat_released_J": "J (cumulative exothermic heat)",
                "co2_rate_mol_s": "mol/s",
            },
            "reaction": self._raw["unit"]["reaction"]["value"],
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 3600.0, "dt": 60.0})
    print(f"Final conversion: {r['final_conversion']:.3f}, "
          f"CO2 stored: {r['co2_stored_kg']:.1f} kg, "
          f"peak T: {r['peak_temperature_K']:.2f} K")
