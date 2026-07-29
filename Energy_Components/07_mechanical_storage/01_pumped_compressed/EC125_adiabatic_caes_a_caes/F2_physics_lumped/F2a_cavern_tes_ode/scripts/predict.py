"""
EC125 — Adiabatic CAES (A-CAES) — F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ACAES_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for A-CAES F2a coupled cavern+TES ODE model."""

    component_id = "EC125"
    component_name = "Adiabatic CAES (A-CAES)"
    fidelity = "F2a -- Coupled Cavern + TES Energy-Balance ODEs (physics-lumped)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ACAES_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic A-CAES simulation (single phase) via coupled ODEs.

        inputs:
            mode        : 'charge' | 'discharge' | 'idle'  (default 'charge')
            m_dot       : air mass-flow magnitude [kg/s]   (default 100.0)
            duration_s  : horizon [s]                      (default 3600.0)
            dt          : output interval [s]              (default 60.0)
            soc0        : initial SOC [0,1]                (default 0.5)
            T_cav0      : initial cavern temp [K]          (default nominal)
            T_tes0      : initial TES temp [K]             (default design 550 K)
            T_amb       : ambient/intake temp [K]          (default 288.15)
        """
        return self._model.simulate(
            mode=inputs.get("mode", "charge"),
            m_dot=inputs.get("m_dot", 100.0),
            duration_s=inputs.get("duration_s", 3600.0),
            dt=inputs.get("dt", 60.0),
            soc0=inputs.get("soc0", 0.5),
            T_cav0=inputs.get("T_cav0", None),
            T_tes0=inputs.get("T_tes0", None),
            T_amb=inputs.get("T_amb", 288.15),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"values": ["charge", "discharge", "idle"]},
                "m_dot": {"unit": "kg/s", "range": [0, 500]},
                "duration_s": {"unit": "s", "range": [1, 86400]},
                "dt": {"unit": "s", "range": [1, 600]},
                "soc0": {"unit": "-", "range": [0, 1]},
                "T_cav0": {"unit": "K", "range": [288.15, 333.15]},
                "T_tes0": {"unit": "K", "range": [298.15, 600.0]},
                "T_amb": {"unit": "K", "range": [253.15, 313.15]},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "T_cav": "K",
                "T_tes": "K",
                "pressure": "Pa",
                "power_elec_kw": "kW",
                "fuel_power_kw": "kW (always 0)",
                "E_elec_kwh": "kWh",
                "rte_design": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # charge phase
    r = m.predict({"mode": "charge", "m_dot": 100.0, "duration_s": 3600.0, "dt": 60.0,
                   "soc0": 0.0, "T_tes0": 298.15})
    print(f"\nCharge: SOC {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}, "
          f"T_tes {r['T_tes'][0]:.1f} -> {r['T_tes'][-1]:.1f} K, "
          f"E_in {r['E_elec_kwh']:.1f} kWh, fuel {r['fuel_power_kw'][-1]:.1f} kW")
    # full round trip
    rte, ch, dis = m._model.round_trip_simulation(m_dot=100.0, charge_s=3600.0, dt=60.0)
    print(f"Round-trip efficiency (ODE-integrated): {rte:.3f}")
    print(f"Design RTE: {m._model.round_trip_efficiency(288.15, 550.0):.3f}")
