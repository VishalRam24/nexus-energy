"""
EC115 -- Integrated Gasification Combined Cycle (IGCC) -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import IGCC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for IGCC F2a physics-lumped model."""

    component_id = "EC115"
    component_name = "Integrated Gasification Combined Cycle (IGCC)"
    fidelity = "F2a -- Physics-Lumped Gasifier + Combined Cycle with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = IGCC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run lumped IGCC simulation (combustor-metal thermal ODE + plant balances).

        inputs:
            coal_rate_kgs : float (or callable(t)->kg/s) coal feed, default = design
            T_metal_K     : float  initial combustor metal temperature [K]
            dt            : float  output time step [s] (default 2.0)
            duration_s    : float  simulation horizon [s] (default 600.0)
        """
        m = self._model
        # default coal rate = design coal flow for rated net power
        default_coal = m.Q_coal_design / m.LHV_coal
        coal = inputs.get("coal_rate_kgs", default_coal)
        T0 = inputs.get("T_metal_K", None)
        dt = inputs.get("dt", 2.0)
        dur = inputs.get("duration_s", 600.0)
        return m.simulate(coal, T0, dt, dur)

    def get_info(self) -> dict:
        m = self._model
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "coal_rate_kgs": {"unit": "kg/s", "range": [5.0, 60.0]},
                "T_metal_K": {"unit": "K", "range": [288.15, 1900.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_metal": "K (combustor metal, ODE state)",
                "T_gas": "K (combustor gas)",
                "net_power_mw": "MW_e",
                "net_efficiency": "-",
                "combined_cycle_efficiency": "-",
                "carnot_efficiency": "-",
                "syngas_rate_nm3s": "Nm3/s",
                "co2_intensity_g_per_kwh": "g/kWh",
                "tau_s": "s",
            },
            "design": {
                "net_efficiency": round(m.net_efficiency(), 4),
                "combined_cycle_efficiency": round(m.combined_cycle_efficiency(), 4),
                "cold_gas_efficiency": m.cge,
                "carnot_bound": round(m.carnot_efficiency(), 4),
                "tau_s": round(m.time_constant(), 1),
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    cm = ComponentModel()
    info = cm.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    print(f"Design net eff = {info['design']['net_efficiency']:.3f}, "
          f"combined-cycle eff = {info['design']['combined_cycle_efficiency']:.3f}, "
          f"Carnot bound = {info['design']['carnot_bound']:.3f}, "
          f"tau = {info['design']['tau_s']:.0f} s")
    r = cm.predict({"duration_s": 600.0, "dt": 5.0})
    print(f"Net power = {r['net_power_mw'][-1]:.1f} MW_e, "
          f"T_metal {r['T_metal'][0]:.0f} -> {r['T_metal'][-1]:.0f} K "
          f"(gas {r['T_gas'][-1]:.0f} K), "
          f"syngas = {r['syngas_rate_nm3s'][-1]:.1f} Nm3/s, "
          f"CO2 = {r['co2_intensity_g_per_kwh'][-1]:.0f} g/kWh")
