"""
EC124 -- Liquid Air Energy Storage (LAES / CES) -- F2a Cryo-Tank Thermo
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LAES_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for LAES F2a physics-lumped cryo-tank model."""

    component_id = "EC124"
    component_name = "Liquid Air Energy Storage (LAES / CES)"
    fidelity = "F2a -- Physics-Lumped Cryo-Tank Thermodynamic ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LAES_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a LAES operation.

        inputs:
            mode : 'charge' | 'discharge' | 'store' | 'round_trip' (default 'round_trip')
            For 'round_trip':
                charge_mass_kg   : float (default = full tank)
                store_hours      : float (default 0)
                T_amb_K          : float (default 298.15)
                cold_recycle_eff : float (default 0.60)
                hot_recycle_dT_K : float (default 0.0)
                m_dot_kgs        : float (default 100)
            For single mode ('charge'/'discharge'/'store'):
                duration_s, m_dot_kgs, m_liq0_kg, Q_cold0_kWh,
                T_amb_K, cold_recycle_eff, hot_recycle_dT_K
        """
        mode = inputs.get("mode", "round_trip")
        T_amb = inputs.get("T_amb_K", 298.15)
        eps_cr = inputs.get("cold_recycle_eff", None)
        hot_dT = inputs.get("hot_recycle_dT_K", 0.0)

        if mode == "round_trip":
            return self._model.round_trip(
                charge_mass_kg=inputs.get("charge_mass_kg", None),
                store_hours=inputs.get("store_hours", 0.0),
                T_amb_K=T_amb,
                eps_cr=eps_cr,
                hot_recycle_dT_K=hot_dT,
                m_dot=inputs.get("m_dot_kgs", 100.0),
            )

        return self._model.simulate(
            mode=mode,
            duration_s=inputs.get("duration_s", 3600.0),
            m_dot=inputs.get("m_dot_kgs", 100.0),
            m_liq0=inputs.get("m_liq0_kg", 0.0),
            Q_cold0=inputs.get("Q_cold0_kWh", 0.0),
            T_amb_K=T_amb,
            eps_cr=eps_cr,
            hot_recycle_dT_K=hot_dT,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"values": ["charge", "discharge", "store", "round_trip"]},
                "charge_mass_kg": {"unit": "kg"},
                "store_hours": {"unit": "h", "range": [0, 336]},
                "T_amb_K": {"unit": "K", "range": [253.15, 323.15]},
                "cold_recycle_eff": {"unit": "-", "range": [0.0, 0.95]},
                "hot_recycle_dT_K": {"unit": "K", "range": [0.0, 320.0]},
                "m_dot_kgs": {"unit": "kg/s", "range": [0.0, 200.0]},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "round_trip": "E_in_kWh, E_out_kWh, eta_RT, boil_off_loss_kg, w_liq/w_exp",
                "single_mode": "t, m_liq, soc, Q_cold, power_kW, energy_kWh",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mode": "round_trip", "store_hours": 6.0,
                   "cold_recycle_eff": 0.60, "hot_recycle_dT_K": 0.0})
    print(f"E_in={r['E_in_kWh']/1e3:.1f} MWh  E_out={r['E_out_kWh']/1e3:.1f} MWh  "
          f"RTE={r['eta_RT']*100:.1f}%  boil-off={r['boil_off_loss_kg']:.0f} kg")
    r2 = m.predict({"mode": "round_trip", "cold_recycle_eff": 0.60,
                    "hot_recycle_dT_K": 120.0})
    print(f"With 120 K waste-heat hot recycle: RTE={r2['eta_RT']*100:.1f}%")
