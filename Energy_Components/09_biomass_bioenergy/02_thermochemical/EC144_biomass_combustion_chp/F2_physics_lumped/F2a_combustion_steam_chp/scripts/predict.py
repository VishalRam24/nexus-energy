"""
EC144 -- Biomass Combustion CHP -- F2a Combustion + Steam-Cycle (Physics-Lumped)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BiomassCombustionCHP_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC144 F2a biomass combustion steam-CHP model."""

    component_id = "EC144"
    component_name = "Biomass Combustion CHP"
    fidelity = "F2a -- Combustion + Steam-Cycle CHP with Lumped Boiler Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BiomassCombustionCHP_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Compute the steady CHP operating point and (optionally) the boiler
        thermal transient.

        inputs:
            PLR : float               part-load ratio [0.25-1.0] (default 1.0)
            moisture_fraction : float wet-basis moisture [0-0.6] (default 0.20)
            transient : bool          also integrate the boiler ODE (default True)
            T0_K : float              initial boiler temperature (default ambient)
            dt : float                ODE output step [s] (default 10)
            duration_s : float        integration horizon [s] (default 3600)

        Returns the steady operating point dict plus, if transient,
        't' and 'T_boiler_K' arrays.
        """
        PLR = inputs.get("PLR", 1.0)
        M = inputs.get("moisture_fraction", 0.20)
        transient = inputs.get("transient", True)

        out = self._model.predict_steady(PLR, M)

        if transient:
            dt = inputs.get("dt", 10.0)
            dur = inputs.get("duration_s", 3600.0)
            T0 = inputs.get("T0_K", None)
            therm = self._model.simulate_thermal(PLR, M, T0_K=T0, dt=dt, duration_s=dur)
            out["t"] = therm["t"]
            out["T_boiler_K"] = therm["T_boiler_K"]

        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "PLR": {"unit": "-", "range": [0.25, 1.0]},
                "moisture_fraction": {"unit": "wet-basis", "range": [0.0, 0.60]},
                "transient": {"unit": "bool"},
                "T0_K": {"unit": "K"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "fuel_input_kw": "kW",
                "useful_heat_kw": "kW",
                "P_electrical_kw": "kW",
                "Q_thermal_kw": "kW",
                "eta_boiler": "-",
                "eta_electrical": "-",
                "eta_thermal": "-",
                "eta_total_chp": "-",
                "eta_carnot": "-",
                "power_to_heat_ratio": "-",
                "LHV_eff_MJ_kg": "MJ/kg",
                "t": "s",
                "T_boiler_K": "K",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"PLR": 1.0, "moisture_fraction": 0.20, "duration_s": 1800.0, "dt": 30.0})
    print(f"\nP_el={r['P_electrical_kw']:.1f} kW  Q_th={r['Q_thermal_kw']:.1f} kW")
    print(f"eta_el={r['eta_electrical']:.3f}  eta_th={r['eta_thermal']:.3f}  "
          f"eta_total={r['eta_total_chp']:.3f}  (Carnot={r['eta_carnot']:.3f})")
    print(f"P/H ratio={r['power_to_heat_ratio']:.3f}  LHV_eff={r['LHV_eff_MJ_kg']:.2f} MJ/kg")
    print(f"Boiler T: {r['T_boiler_K'][0]:.1f} K -> {r['T_boiler_K'][-1]:.1f} K")
