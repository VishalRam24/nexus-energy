"""
EC123 — Compressed Air Energy Storage (CAES), Diabatic — F2a Cavern Thermodynamics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CAESF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for CAES F2a cavern-thermodynamics model."""

    component_id = "EC123"
    component_name = "Compressed Air Energy Storage (CAES) — Diabatic"
    fidelity = "F2a — Physics-Lumped Cavern Thermodynamics with Fuel-Fired Expansion"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CAESF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a cavern dynamic simulation under one charge/discharge/idle command.

        inputs:
            mode : str           "charge" | "discharge" | "idle"  (default "charge")
            m_dot_kg_s : float   air mass flow [kg/s]             (default 100.0)
            T0_K : float         initial cavern temperature [K]   (default T_rock)
            P0_Pa : float        initial cavern pressure [Pa]     (default p_min)
            dt : float           output time step [s]             (default 60.0)
            duration_s : float   total duration [s]               (default 3600.0)

        Returns a dict of time series plus cumulative energy/fuel accounting and
        the steady round-trip efficiency.
        """
        mode = inputs.get("mode", "charge")
        m_dot = inputs.get("m_dot_kg_s", 100.0)
        T0 = inputs.get("T0_K", self._model.T_rock)
        P0 = inputs.get("P0_Pa", self._model.p_min)
        dt = inputs.get("dt", 60.0)
        dur = inputs.get("duration_s", 3600.0)

        result = self._model.simulate(mode, m_dot, T0, P0, dt, dur)
        result["round_trip_efficiency"] = self._model.round_trip_efficiency()
        result["electric_rte"] = self._model.electric_rte()
        result["heat_rate_kJ_per_kWh"] = self._model.heat_rate()
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"unit": "-", "values": ["charge", "discharge", "idle"]},
                "m_dot_kg_s": {"unit": "kg/s", "range": [0.0, 600.0]},
                "T0_K": {"unit": "K", "range": [293.15, 360.15]},
                "P0_Pa": {"unit": "Pa", "range": [4.3e6, 7.0e6]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "mass": "kg",
                "temperature": "K",
                "pressure": "Pa",
                "soc": "-",
                "P_elec": "W",
                "P_fuel": "W",
                "E_elec_J": "J",
                "E_fuel_J": "J",
                "m_fuel_kg": "kg",
                "round_trip_efficiency": "-",
                "electric_rte": "-",
                "heat_rate_kJ_per_kWh": "kJ/kWh_e",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mode": "charge", "m_dot_kg_s": 100.0, "duration_s": 3600.0, "dt": 300.0})
    print(f"\nCharge 1h @100 kg/s:")
    print(f"  P_cavern: {r['pressure'][0]/1e5:.1f} -> {r['pressure'][-1]/1e5:.1f} bar")
    print(f"  T_cavern: {r['temperature'][0]:.1f} -> {r['temperature'][-1]:.1f} K")
    print(f"  SOC: {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")
    print(f"  E_elec_in: {r['E_elec_J']/3.6e9:.2f} MWh")
    print(f"  RTE (incl. fuel): {r['round_trip_efficiency']:.3f}")
    print(f"  Electric RTE: {r['electric_rte']:.3f}")
    print(f"  Heat rate: {r['heat_rate_kJ_per_kWh']:.0f} kJ/kWh_e")
