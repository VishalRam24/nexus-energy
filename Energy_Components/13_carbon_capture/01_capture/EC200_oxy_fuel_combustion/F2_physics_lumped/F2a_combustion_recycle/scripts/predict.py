"""
EC200 -- Oxy-Fuel Combustion Capture -- F2a Combustion + Recycle Model
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OxyFuelF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC200 oxy-fuel F2a furnace model."""

    component_id = "EC200"
    component_name = "Oxy-Fuel Combustion Capture"
    fidelity = "F2a -- Combustion + Flue-Gas Recycle (lumped ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = OxyFuelF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run lumped oxy-fuel furnace simulation.

        inputs:
            mdot_fuel     : float  fuel mass flow (kg/s)
            recycle_ratio : float  m_recycle/m_product (0..0.8), default 0.6
            T0            : float  initial gas temperature (K), default T_wall+200
            dt            : float  output time step (s), default 0.5
            duration_s    : float  simulation length (s), default 120.0

        returns time-series + steady-state derived quantities (see model.simulate).
        """
        mdot = inputs.get("mdot_fuel", 50.0)
        R = inputs.get("recycle_ratio", 0.6)
        T0 = inputs.get("T0", None)
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 120.0)
        return self._model.simulate(mdot, R, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "description": (
                "Lumped oxy-fuel furnace. Combustion in O2 + recycled flue gas; "
                "energy-balance ODE with recycle ratio moderating flame "
                "temperature; high-purity (>0.9 dry) CO2 after water knockout."
            ),
            "inputs": {
                "mdot_fuel": {"unit": "kg/s", "range": [1.0, 100.0]},
                "recycle_ratio": {"unit": "-", "range": [0.0, 0.8], "default": 0.6},
                "T0": {"unit": "K", "range": [800.0, 2500.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "T_steady": "K",
                "T_adiabatic": "K",
                "co2_purity_dry": "-",
                "co2_purity_wet": "-",
                "o2_demand_kgs": "kg/s",
                "co2_produced_kgs": "kg/s",
                "product_gas_kgs": "kg/s",
                "recycle_kgs": "kg/s",
            },
            "source": self._raw.get("source", ""),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "duration_s": 120.0})
    print("EC200 Oxy-Fuel F2a -- fuel=50 kg/s, recycle=0.6:")
    print(f"  Steady furnace T   = {r['T_steady']:.0f} K")
    print(f"  Adiabatic flame T  = {r['T_adiabatic']:.0f} K")
    print(f"  CO2 purity (dry)   = {r['co2_purity_dry']:.3f}")
    print(f"  CO2 purity (wet)   = {r['co2_purity_wet']:.3f}")
    print(f"  O2 demand          = {r['o2_demand_kgs']:.2f} kg/s")
    print(f"  CO2 produced       = {r['co2_produced_kgs']:.2f} kg/s")
    print(f"  Product gas        = {r['product_gas_kgs']:.2f} kg/s")
    print(f"  Recycle            = {r['recycle_kgs']:.2f} kg/s")
    info = m.get_info()
    print(f"  {info['component_id']} / {info['fidelity']}")
