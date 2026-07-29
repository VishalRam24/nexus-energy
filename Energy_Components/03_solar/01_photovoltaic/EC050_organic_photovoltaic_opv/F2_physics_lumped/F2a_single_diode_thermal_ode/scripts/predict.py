"""
EC050 -- Organic Photovoltaic (OPV) -- F2a Physics-Lumped Single-Diode + Thermal ODE
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OPV_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for OPV F2a physics-lumped single-diode + thermal ODE."""

    component_id = "EC050"
    component_name = "Organic Photovoltaic (OPV)"
    fidelity = "F2a -- Physics-Lumped Single-Diode + Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = OPV_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient (thermal-ODE) OPV simulation and report the final
        steady operating point plus the time-series.

        inputs:
            irradiance   : float W/m2 (or callable t->G)   default 1000.0
            T_ambient_C  : float degC (or callable t->Ta)  default 25.0
            T0_C         : float degC initial cell temp     default = T_ambient
            dt           : float s output step              default 5.0
            duration_s   : float s total duration           default 600.0

        returns: dict with time-series arrays + scalar final-state summary.
        """
        G = inputs.get("irradiance", 1000.0)
        Ta = inputs.get("T_ambient_C", 25.0)
        T0 = inputs.get("T0_C", None)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 600.0)

        ts = self._model.simulate(G, Ta, T0_C=T0, dt=dt, duration_s=dur)
        ts["final"] = {
            "T_cell_C": float(ts["T_cell_C"][-1]),
            "Voc": float(ts["Voc"][-1]),
            "Isc": float(ts["Isc"][-1]),
            "Vmp": float(ts["Vmp"][-1]),
            "Imp": float(ts["Imp"][-1]),
            "power_W": float(ts["power"][-1]),
            "efficiency": float(ts["efficiency"][-1]),
            "FF": float(ts["FF"][-1]),
        }
        return ts

    def iv_curve(self, irradiance=1000.0, T_cell_C=25.0, n_points=200) -> dict:
        """Convenience: full I-V / P-V curve at fixed irradiance and cell temp."""
        return self._model.iv_curve(irradiance, T_cell_C + 273.15, n_points)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "irradiance": {"unit": "W/m2", "range": [0, 1200]},
                "T_ambient_C": {"unit": "degC", "range": [-10, 60]},
                "T0_C": {"unit": "degC"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_cell_C": "degC",
                "Voc": "V", "Isc": "A", "Vmp": "V", "Imp": "A",
                "power": "W", "efficiency": "-", "FF": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"irradiance": 1000.0, "T_ambient_C": 25.0,
                   "dt": 10.0, "duration_s": 300.0})
    f = r["final"]
    print(f"STC: P={f['power_W']:.3f} W, eta={f['efficiency']*100:.2f} %, "
          f"FF={f['FF']:.3f}, Voc={f['Voc']:.2f} V, Isc={f['Isc']:.3f} A, "
          f"T_cell={f['T_cell_C']:.1f} C")
    # Low-light / indoor demonstration
    r2 = m.predict({"irradiance": 100.0, "T_ambient_C": 25.0,
                    "dt": 10.0, "duration_s": 300.0})
    print(f"Low-light (100 W/m2): eta={r2['final']['efficiency']*100:.2f} %")
