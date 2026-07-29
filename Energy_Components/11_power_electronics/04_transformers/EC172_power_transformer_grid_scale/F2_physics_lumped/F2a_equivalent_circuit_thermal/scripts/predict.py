"""
EC172 -- Power Transformer (Grid-Scale) -- F2a Equivalent-Circuit + Thermal ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PowerTransformerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC172 F2a equivalent-circuit + thermal model."""

    component_id = "EC172"
    component_name = "Power Transformer (Grid-Scale)"
    fidelity = "F2a -- Equivalent Circuit + Lumped Oil/Winding Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PowerTransformerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a loading transient + report electrical/thermal trajectory.

        inputs:
            load_fraction        : float or callable f(t_min)->K  (per-unit load)
            power_factor         : float (default 0.9 lagging)
            voltage_pu           : float (default 1.0)
            ambient_temperature_C: float (default 20.0)
            dt_min               : float (default 1.0)
            duration_min         : float (default 240.0)

        returns: dict with time arrays plus scalar steady-state summary fields.
        """
        load = inputs.get("load_fraction", 1.0)
        pf = inputs.get("power_factor", 0.9)
        v_pu = inputs.get("voltage_pu", 1.0)
        T_amb = inputs.get("ambient_temperature_C", 20.0)
        dt = inputs.get("dt_min", 1.0)
        dur = inputs.get("duration_min", 240.0)

        res = self._model.simulate(
            load, ambient_temperature=T_amb, dt=dt, duration=dur,
            power_factor=pf, voltage_pu=v_pu,
        )

        # steady-state scalar summary (constant-load convenience)
        K0 = load(0.0) if callable(load) else float(load)
        res["regulation_pct"] = float(self._model.voltage_regulation(K0, pf) * 100.0)
        res["efficiency_ss"] = float(res["efficiency"][-1])
        res["hotspot_temp_final_C"] = float(res["hotspot_temperature"][-1])
        res["max_efficiency_load"] = self._model.max_efficiency_load()
        return res

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "load_fraction": {"unit": "pu", "range": [0.0, 1.5]},
                "power_factor": {"unit": "-", "range": [0.0, 1.0]},
                "voltage_pu": {"unit": "pu", "range": [0.8, 1.2]},
                "ambient_temperature_C": {"unit": "degC", "range": [-40, 50]},
                "dt_min": {"unit": "min"},
                "duration_min": {"unit": "min"},
            },
            "outputs": {
                "t": "min",
                "hotspot_temperature": "degC",
                "top_oil_temperature": "degC",
                "efficiency": "-",
                "voltage_regulation": "fraction",
                "p_total_loss": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"load_fraction": 1.0, "power_factor": 0.9, "duration_min": 240.0})
    print(f"Regulation: {r['regulation_pct']:.2f} %  |  "
          f"Efficiency: {r['efficiency_ss']*100:.3f} %  |  "
          f"Hot-spot (final): {r['hotspot_temp_final_C']:.1f} C  |  "
          f"Peak-eff load: {r['max_efficiency_load']:.3f} pu")
