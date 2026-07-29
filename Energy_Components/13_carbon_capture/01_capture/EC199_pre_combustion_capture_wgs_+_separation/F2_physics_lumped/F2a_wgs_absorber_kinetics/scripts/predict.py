"""
EC199 -- Pre-Combustion Capture (WGS + Separation) -- F2a Physics-Lumped
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PreCombustionCaptureF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC199 pre-combustion capture F2a model."""

    component_id = "EC199"
    component_name = "Pre-Combustion Capture (WGS + Separation)"
    fidelity = "F2a -- WGS Reactor Kinetics + Physical-Solvent Absorber ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PreCombustionCaptureF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the coupled WGS + absorber simulation.

        inputs:
            syngas_flow_mol_s : float  (default 1000.0)
            co_fraction       : float  (default 0.45)
            h2_fraction       : float  (default 0.35)
            co2_fraction      : float  (default 0.0)
            inert_fraction    : float  (default 0.0)
            T_WGS_K           : float  (default nominal 523.15)
            T_abs_K           : float  (default nominal 298.15)
            P_bar             : float  (default 30.0)
            steam_co_ratio    : float  (default 3.0)
        """
        r = self._model.simulate(
            syngas_flow_mol_s=inputs.get("syngas_flow_mol_s", 1000.0),
            co_fraction=inputs.get("co_fraction", 0.45),
            h2_fraction=inputs.get("h2_fraction", 0.35),
            co2_fraction=inputs.get("co2_fraction", 0.0),
            inert_fraction=inputs.get("inert_fraction", 0.0),
            T_WGS_K=inputs.get("T_WGS_K", None),
            T_abs_K=inputs.get("T_abs_K", None),
            P_bar=inputs.get("P_bar", 30.0),
            steam_co_ratio=inputs.get("steam_co_ratio", 3.0),
        )
        # Drop heavy sub-dicts' time arrays from the top-level for a clean API,
        # but keep them accessible.
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "syngas_flow_mol_s": {"unit": "mol/s", "range": [1.0, 5000.0]},
                "co_fraction": {"unit": "mol/mol", "range": [0.1, 0.6]},
                "h2_fraction": {"unit": "mol/mol", "range": [0.1, 0.6]},
                "co2_fraction": {"unit": "mol/mol", "range": [0.0, 0.3]},
                "inert_fraction": {"unit": "mol/mol", "range": [0.0, 0.5]},
                "T_WGS_K": {"unit": "K", "range": [453.15, 723.15]},
                "T_abs_K": {"unit": "K", "range": [273.15, 313.15]},
                "P_bar": {"unit": "bar", "range": [10.0, 60.0]},
                "steam_co_ratio": {"unit": "mol/mol", "range": [2.0, 5.0]},
            },
            "outputs": {
                "capture_rate": "-",
                "wgs_conversion": "-",
                "wgs_equilibrium_conversion": "-",
                "co2_captured_kg_s": "kg/s",
                "h2_rich_fuel_mol_s": "mol/s",
                "h2_purity": "-",
                "energy_penalty_GJ_tCO2": "GJ/tCO2",
                "power_penalty_MW": "MW",
                "wgs_heat_kW": "kW",
                "carbon_residual": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"syngas_flow_mol_s": 1000.0, "co_fraction": 0.45,
                   "h2_fraction": 0.35, "P_bar": 30.0})
    print(f"  WGS conversion        : {r['wgs_conversion']*100:.1f} %  "
          f"(equilibrium {r['wgs_equilibrium_conversion']*100:.1f} %)")
    print(f"  CO2 capture rate      : {r['capture_rate']*100:.1f} %")
    print(f"  CO2 captured          : {r['co2_captured_kg_s']:.2f} kg/s")
    print(f"  H2-rich fuel          : {r['h2_rich_fuel_mol_s']:.1f} mol/s "
          f"(purity {r['h2_purity']*100:.1f} %)")
    print(f"  Energy penalty        : {r['energy_penalty_GJ_tCO2']:.2f} GJ/tCO2 "
          f"({r['power_penalty_MW']:.1f} MW)")
    print(f"  Carbon balance residual: {r['carbon_residual']:.2e}")
