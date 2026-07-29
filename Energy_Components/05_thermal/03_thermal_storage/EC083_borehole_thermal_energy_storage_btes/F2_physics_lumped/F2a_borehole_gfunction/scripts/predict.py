"""
EC083 -- Borehole Thermal Energy Storage (BTES) -- F2a Physics-Lumped
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BTES_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for BTES F2a physics-lumped g-function model."""

    component_id = "EC083"
    component_name = "Borehole Thermal Energy Storage (BTES)"
    fidelity = "F2a -- Physics-Lumped Borehole HX + g-function Ground Coupling"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BTES_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic BTES simulation.

        inputs:
            Q_fluid_W   : float or callable(t)->W. + = charge (inject heat),
                          - = discharge (extract heat). Default 500 kW charge.
            T_store0_C  : float, initial mean store temperature [degC]
                          (default = undisturbed ground temperature)
            T_amb_C     : float, surrounding temperature [degC] (default 8.0)
            duration_s  : float, simulation horizon [s] (default 180 days)
            n_out       : int, number of output samples (default 400)
        """
        Q = inputs.get("Q_fluid_W", 500000.0)
        T0 = inputs.get("T_store0_C", None)
        T_amb = inputs.get("T_amb_C", 8.0)
        dur = inputs.get("duration_s", 180 * 24 * 3600.0)
        n_out = inputs.get("n_out", 400)

        return self._model.simulate(Q, T0, T_amb, dur, n_out)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Q_fluid_W": {"unit": "W", "range": [-1.5e6, 1.5e6],
                              "note": "+ charge / - discharge"},
                "T_store0_C": {"unit": "degC", "range": [0, 95]},
                "T_amb_C": {"unit": "degC", "range": [-10, 30]},
                "duration_s": {"unit": "s", "range": [3600, 6.3e7]},
                "n_out": {"unit": "-"},
            },
            "outputs": {
                "t": "s",
                "t_days": "days",
                "T_store": "degC (mean ground store temperature)",
                "T_wall": "degC (borehole wall temperature)",
                "T_fluid_mean": "degC",
                "T_in": "degC (fluid inlet)",
                "T_out": "degC (fluid outlet)",
                "Q_fluid": "W",
                "Q_loss": "W",
                "E_stored_MWh": "MWh (relative to undisturbed ground)",
            },
            "derived": {
                "t_s_years": self._model.t_s / (365.25 * 86400.0),
                "C_store_GJperK": self._model.C_store / 1e9,
                "V_store_m3": self._model.V_store,
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(info["fidelity"])
    print(f"  storage time scale t_s = {info['derived']['t_s_years']:.2f} yr,"
          f" C_store = {info['derived']['C_store_GJperK']:.2f} GJ/K,"
          f" V_store = {info['derived']['V_store_m3']:.0f} m3")
    # 90-day charge at 500 kW from undisturbed ground
    r = m.predict({"Q_fluid_W": 500000.0, "duration_s": 90 * 24 * 3600.0})
    print(f"  After 90 d charge @500 kW: T_store = {r['T_store'][-1]:.2f} C,"
          f" T_out = {r['T_out'][-1]:.2f} C, E = {r['E_stored_MWh'][-1]:.1f} MWh")
    # 90-day discharge from a warmed store
    r2 = m.predict({"Q_fluid_W": -300000.0, "T_store0_C": 40.0,
                    "duration_s": 90 * 24 * 3600.0})
    print(f"  After 90 d discharge @300 kW: T_store = {r2['T_store'][-1]:.2f} C,"
          f" T_out = {r2['T_out'][-1]:.2f} C, E = {r2['E_stored_MWh'][-1]:.1f} MWh")
