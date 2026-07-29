"""
EC071 -- Absorption Heat Pump (LiBr-H2O) -- F2a Physics-Lumped Absorption Cycle
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AbsorptionHeatPumpF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC071 F2a absorption-cycle model."""

    component_id = "EC071"
    component_name = "Absorption Heat Pump (LiBr-H2O, single-effect)"
    fidelity = "F2a -- Physics-Lumped Absorption Cycle with Solution Circuit + Transient ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AbsorptionHeatPumpF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient generator-loop simulation plus the steady cycle.

        inputs:
            T_gen_drive_C : float (or callable)  driving heat-source temp [C]
            T_gen0_C      : float  initial generator solution temp [C]
            T_evap_C, T_cond_C, T_abs_C : optional override operating temps [C]
            plr           : float  part-load ratio (default 1.0)
            dt            : float  output time step [s] (default 5.0)
            duration_s    : float  simulation duration [s] (default 1800.0)

        Returns time-series (transient) merged with steady-state duties/COP.
        """
        T_drive = inputs.get("T_gen_drive_C", None)
        T_gen0 = inputs.get("T_gen0_C", None)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 1800.0)
        plr = inputs.get("plr", 1.0)

        # optional operating-point overrides
        for key, attr in (("T_evap_C", "T_evap"), ("T_cond_C", "T_cond"),
                          ("T_abs_C", "T_abs")):
            if key in inputs:
                setattr(self._model, attr, inputs[key])
        if T_drive is not None and not callable(T_drive):
            self._model.T_drive = T_drive

        ts = self._model.simulate(T_drive_c=T_drive, T_gen0_c=T_gen0,
                                  dt=dt, duration_s=dur)
        steady = self._model.rate_duties(plr=plr)
        ts.update({
            "Q_gen_kW_design": steady["Q_gen_kW"],
            "Q_heat_kW_design": steady["Q_heat_kW"],
            "Q_evap_kW_design": steady["Q_evap_kW"],
            "cop_heating_design": steady["cop_heating"],
            "cop_cooling_design": steady["cop_cooling"],
            "f_circulation": steady["f_circulation"],
            "P_aux_kW": steady["P_aux_kW"],
        })
        return ts

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_gen_drive_C": {"unit": "degC", "range": [70, 110]},
                "T_gen0_C": {"unit": "degC", "range": [20, 110]},
                "T_evap_C": {"unit": "degC", "range": [2, 25]},
                "T_cond_C": {"unit": "degC", "range": [25, 50]},
                "T_abs_C": {"unit": "degC", "range": [25, 45]},
                "plr": {"unit": "-", "range": [0, 1.2]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_gen_C": "degC",
                "Q_gen_kW": "kW (driving heat)",
                "Q_evap_kW": "kW (low-grade source)",
                "Q_heat_kW": "kW (useful heating = cond+abs)",
                "cop_heating": "-",
                "cop_cooling": "-",
                "f_circulation": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_gen0_C": 40.0, "duration_s": 1200.0, "dt": 60.0})
    print(f"Final T_gen: {r['T_gen_C'][-1]:.2f} C, "
          f"COP_heat: {r['cop_heating_design']:.3f}, "
          f"COP_cool: {r['cop_cooling_design']:.3f}, "
          f"Q_heat: {r['Q_heat_kW_design']:.1f} kW, f={r['f_circulation']:.2f}")
