"""
EC196 -- Synthetic Jet Fuel (Power-to-Liquid) -- F2a
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FTJetFuelF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the FT PtL jet-fuel F2a kinetics+thermal model."""

    component_id = "EC196"
    component_name = "Synthetic Jet Fuel (Power-to-Liquid)"
    fidelity = "F2a -- FT Chain-Growth Kinetics + Lumped Reactor Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FTJetFuelF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the dynamic FT reactor simulation.

        inputs:
            T0_C : float        initial reactor temperature [degC] (default: coolant T)
            P_CO_bar : float    CO partial pressure [bar]   (default from params)
            P_H2_bar : float    H2 partial pressure [bar]   (default from params)
            n_CO_in_mol_s : float CO feed rate [mol/s]      (default from params)
            dt : float          output step [s]             (default 10.0)
            duration_s : float  total duration [s]          (default 3600.0)
        """
        T0 = inputs.get("T0_C", None)
        P_CO = inputs.get("P_CO_bar", None)
        P_H2 = inputs.get("P_H2_bar", None)
        n_co = inputs.get("n_CO_in_mol_s", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(T0, P_CO, P_H2, n_co, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T0_C": {"unit": "degC", "range": [150, 300]},
                "P_CO_bar": {"unit": "bar", "range": [1, 20]},
                "P_H2_bar": {"unit": "bar", "range": [1, 40]},
                "n_CO_in_mol_s": {"unit": "mol/s", "range": [0.01, 100]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature": "degC",
                "co_conversion": "-",
                "selectivity_jet": "- (C8-C16 carbon-weight fraction incl. hydrocracked wax)",
                "alpha": "- (ASF chain-growth probability)",
                "jet_mol_s": "mol/s (C12 surrogate)",
                "jet_kg_s": "kg/s",
                "ptl_efficiency": "- (power-to-liquid, <1)",
                "heat_released_kW": "kW",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T0_C": 200.0, "duration_s": 3600.0, "dt": 60.0})
    print(
        f"Final T: {r['temperature'][-1]:.2f} degC | "
        f"X_CO: {r['co_conversion'][-1]:.3f} | "
        f"S_jet: {r['selectivity_jet'][-1]:.3f} | "
        f"jet: {r['jet_kg_s'][-1]*3600:.2f} kg/h | "
        f"eta_PtL: {r['ptl_efficiency'][-1]:.3f}"
    )
