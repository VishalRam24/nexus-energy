"""
EC197 -- DME Synthesis Reactor -- F2a Kinetics + Equilibrium
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DMEReactorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the DME synthesis reactor F2a kinetics model."""

    component_id = "EC197"
    component_name = "DME Synthesis Reactor"
    fidelity = "F2a -- Kinetics + Equilibrium (LHHW + Lumped Reactor Energy Balance)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DMEReactorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the reactor simulation along the residence-time coordinate.

        inputs:
            T_in_K     : float   inlet temperature [K]      (default 523.15)
            P_bar      : float   pressure [bar]             (default 40.0)
            tau_max    : float   max residence coord        (default 2.0)
            n_eval     : int     output points              (default 120)
            n_CO_in    : float   CO feed [mol/s]            (default param)
            H2_CO      : float   feed H2/CO ratio           (default param)
            CO2_frac   : float   CO2 fraction               (default param)
            adiabatic  : bool    no coolant if True         (default False)

        Returns dict of reactor profiles plus scalar exit summary.
        """
        T_in = inputs.get("T_in_K", 523.15)
        P = inputs.get("P_bar", 40.0)
        tau_max = inputs.get("tau_max", 2.0)
        n_eval = inputs.get("n_eval", 120)
        n_CO_in = inputs.get("n_CO_in", None)
        H2_CO = inputs.get("H2_CO", None)
        CO2_frac = inputs.get("CO2_frac", None)
        adiabatic = inputs.get("adiabatic", False)

        r = self._model.simulate(
            T_in, P, tau_max=tau_max, n_eval=n_eval,
            n_CO_in=n_CO_in, H2_CO=H2_CO, CO2_frac=CO2_frac, adiabatic=adiabatic,
        )
        # Append scalar exit summary
        r["exit"] = {
            "CO_conversion": float(r["CO_conversion"][-1]),
            "methanol_conversion": float(r["methanol_conversion"][-1]),
            "DME_yield": float(r["DME_yield"][-1]),
            "DME_selectivity": float(r["DME_selectivity"][-1]),
            "T_exit_K": float(r["T"][-1]),
            "n_DME_mol_s": float(r["n_DME"][-1]),
        }
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_in_K": {"unit": "K", "range": [473.15, 573.15]},
                "P_bar": {"unit": "bar", "range": [10.0, 80.0]},
                "tau_max": {"unit": "s-equiv", "range": [0.1, 10.0]},
                "n_eval": {"unit": "-"},
                "n_CO_in": {"unit": "mol/s", "range": [0.01, 100.0]},
                "H2_CO": {"unit": "mol/mol", "range": [1.0, 3.0]},
                "CO2_frac": {"unit": "mol/mol"},
                "adiabatic": {"unit": "bool"},
            },
            "outputs": {
                "tau": "s-equiv",
                "T": "K",
                "CO_conversion": "-",
                "methanol_conversion": "-",
                "DME_yield": "-",
                "DME_selectivity": "-",
                "heat_release_kW": "kW/m3",
                "n_DME": "mol/s",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_in_K": 523.15, "P_bar": 40.0, "tau_max": 2.0})
    e = r["exit"]
    print(f"Exit: X_CO={e['CO_conversion']:.3f}, DME_yield={e['DME_yield']:.3f}, "
          f"DME_sel={e['DME_selectivity']:.3f}, T_exit={e['T_exit_K']:.1f} K, "
          f"DME={e['n_DME_mol_s']:.4f} mol/s")
