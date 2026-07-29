"""
EC208 -- CO2 Geological Sequestration -- F2a Physics-Lumped Reservoir / Trapping
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2SequestrationF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC208 F2a reservoir + trapping ODE model."""

    component_id = "EC208"
    component_name = "CO2 Geological Sequestration"
    fidelity = "F2a -- Physics-Lumped Reservoir Pressure + Saturation + Trapping ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw.update(params)
        self._model = CO2SequestrationF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped reservoir / trapping simulation.

        inputs:
            P_wellhead_bar  : float  (default from parameters.json)
            injection_years : float  (active injection duration, default 30)
            sim_years       : float  (total horizon, default max(3*inj, 200))
            n_points        : int    (output resolution, default 200)
        """
        return self._model.simulate(
            P_wellhead_bar=inputs.get("P_wellhead_bar"),
            injection_years=inputs.get("injection_years"),
            sim_years=inputs.get("sim_years"),
            n_points=inputs.get("n_points", 200),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_wellhead_bar": {"unit": "bar", "range": [40.0, 200.0]},
                "injection_years": {"unit": "yr", "range": [1.0, 100.0]},
                "sim_years": {"unit": "yr", "range": [1.0, 2000.0]},
                "n_points": {"unit": "-"},
            },
            "outputs": {
                "t_years": "yr",
                "M_mobile_t": "kg (structural/mobile)",
                "M_residual_t": "kg (residual trapping)",
                "M_dissolved_t": "kg (solubility trapping)",
                "M_mineral_t": "kg (mineral trapping)",
                "M_total_t": "kg (total stored)",
                "injected_cumulative_t": "kg (independent mass check)",
                "reservoir_pressure_bar": "bar",
                "fracture_pressure_bar": "bar",
                "saturation_avg": "-",
                "plume_radius_m": "m",
                "injection_rate_kg_s": "kg/s",
                "trapping_fraction": "dict of fraction arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"P_wellhead_bar": 80.0, "injection_years": 30.0, "sim_years": 500.0})
    Mt = 1e9  # kg per Mt
    print(f"  Injected (cum):   {r['injected_cumulative_t'][-1]/Mt:8.3f} Mt")
    print(f"  Total stored:     {r['M_total_t'][-1]/Mt:8.3f} Mt")
    print(f"  Final P_res:      {r['reservoir_pressure_bar'][-1]:8.2f} bar "
          f"(frac {r['fracture_pressure_bar']:.1f} bar)")
    print(f"  Plume radius:     {r['plume_radius_m'].max():8.1f} m")
    print("  Trapping fractions @ end:")
    for k, v in r["trapping_fraction"].items():
        print(f"     {k:11s}: {v[-1]*100:6.2f} %")
