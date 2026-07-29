"""
EC131 -- Tidal Barrage -- F2a Physics-Lumped Basin ODE
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TidalBarrageF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC131 tidal barrage basin-ODE model."""

    component_id = "EC131"
    component_name = "Tidal Barrage"
    fidelity = "F2a -- Physics-Lumped Basin Water-Level ODE (Bulb Turbine)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TidalBarrageF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the basin water-level ODE simulation over n_cycles tidal periods.

        inputs:
            n_cycles : int    number of tidal cycles to simulate (default 2)
            tidal_amplitude_m : float  override sea amplitude [m] (optional)
            z_init_m : float  initial basin level [m] (default = mean sea level)
            flood_gen : bool  enable flood (two-way) generation (default False)
            n_eval : int      number of output samples (default 2000)

        returns dict with t, z_sea, z_basin, head, flow, power, and
        energy_per_cycle_MWh, avg_power_MW, peak_power_MW, mass-balance volumes.
        """
        n_cycles = int(inputs.get("n_cycles", 2))
        amp = inputs.get("tidal_amplitude_m", None)
        z_init = inputs.get("z_init_m", None)
        flood_gen = bool(inputs.get("flood_gen", False))
        n_eval = int(inputs.get("n_eval", 2000))

        if amp is not None:
            self._model.a = float(amp)

        result = self._model.simulate(
            n_cycles=n_cycles, z_init=z_init, flood_gen=flood_gen, n_eval=n_eval
        )
        result["theoretical_energy_per_cycle_MWh"] = \
            self._model.theoretical_energy_per_cycle_MWh()
        result["max_energy_per_cycle_MWh"] = \
            self._model.max_energy_per_cycle_MWh()
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "n_cycles": {"unit": "-", "range": [1, 30]},
                "tidal_amplitude_m": {"unit": "m", "range": [0.5, 8.0]},
                "z_init_m": {"unit": "m"},
                "flood_gen": {"unit": "bool"},
                "n_eval": {"unit": "-"},
            },
            "outputs": {
                "t": "s",
                "z_sea": "m",
                "z_basin": "m",
                "head": "m",
                "flow": "m3/s",
                "power": "W",
                "energy_per_cycle_MWh": "MWh",
                "avg_power_MW": "MW",
                "peak_power_MW": "MW",
                "volume_in_m3": "m3",
                "volume_out_m3": "m3",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"n_cycles": 2, "flood_gen": False})
    print(f"Energy/cycle: {r['energy_per_cycle_MWh']:.1f} MWh "
          f"(theoretical {r['theoretical_energy_per_cycle_MWh']:.1f} MWh)")
    print(f"Avg power: {r['avg_power_MW']:.1f} MW, "
          f"Peak power: {r['peak_power_MW']:.1f} MW")
    print(f"Mass balance: V_in={r['volume_in_m3']:.3e} m3, "
          f"V_out={r['volume_out_m3']:.3e} m3, solver_ok={r['solver_success']}")
