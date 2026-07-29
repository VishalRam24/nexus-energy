"""
EC191 -- Gas Compressor Station -- F2a Centrifugal Polytropic
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import NGCompressorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC191 F2a centrifugal compressor model."""

    component_id = "EC191"
    component_name = "Gas Compressor Station"
    fidelity = "F2a -- Centrifugal Compressor Physics-Lumped (head-flow + polytropic real-gas + plenum/thermal ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = NGCompressorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic discharge-pressure / thermal simulation.

        inputs:
            mass_flow_kg_s : float (or callable t->float for time-varying)
            speed_ratio    : float  (shaft speed fraction of design, default 1.0)
            P_inlet_bar    : float  (default from params)
            T_inlet_K      : float  (default from params)
            dt             : float  (default 0.1 s)
            duration_s     : float  (default 60.0 s)
        """
        m = inputs.get("mass_flow_kg_s", 60.0)
        sr = inputs.get("speed_ratio", 1.0)
        P_in = inputs.get("P_inlet_bar", None)
        T_in = inputs.get("T_inlet_K", None)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 60.0)

        return self._model.simulate(m, sr, P_in, T_in, dt, dur)

    def operating_point(self, inputs: dict) -> dict:
        """Steady-state operating-point summary (no integration)."""
        m = inputs.get("mass_flow_kg_s", 60.0)
        sr = inputs.get("speed_ratio", 1.0)
        P_in = inputs.get("P_inlet_bar", None)
        T_in = inputs.get("T_inlet_K", None)
        return self._model.operating_point(m, sr, P_in, T_in)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mass_flow_kg_s": {"unit": "kg/s", "range": [1.0, 200.0]},
                "speed_ratio": {"unit": "-", "range": [0.5, 1.05]},
                "P_inlet_bar": {"unit": "bar", "range": [5.0, 100.0]},
                "T_inlet_K": {"unit": "K", "range": [263.0, 323.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "P_discharge_bar": "bar",
                "T_discharge_K": "K",
                "phi": "- (flow coefficient)",
                "psi": "- (head coefficient)",
                "H_poly_J_per_kg": "J/kg",
                "pressure_ratio": "-",
                "mass_flow_kg_s": "kg/s",
                "shaft_power_MW": "MW",
                "fuel_power_MW": "MW",
                "in_surge": "bool",
                "in_choke": "bool",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    op = m.operating_point({"mass_flow_kg_s": 60.0, "speed_ratio": 1.0})
    print(f"\nSteady op-point @ 60 kg/s: PR={op['pressure_ratio']:.3f}, "
          f"P_disch={op['P_discharge_bar']:.1f} bar, "
          f"T_disch={op['T_discharge_K']:.1f} K, "
          f"shaft={op['shaft_power_MW']:.2f} MW, surge={op['in_surge']}")
    r = m.predict({"mass_flow_kg_s": 60.0, "duration_s": 30.0, "dt": 0.5})
    print(f"Transient: P_disch {r['P_discharge_bar'][0]:.1f} -> "
          f"{r['P_discharge_bar'][-1]:.1f} bar, "
          f"T_disch {r['T_discharge_K'][0]:.1f} -> {r['T_discharge_K'][-1]:.1f} K")
