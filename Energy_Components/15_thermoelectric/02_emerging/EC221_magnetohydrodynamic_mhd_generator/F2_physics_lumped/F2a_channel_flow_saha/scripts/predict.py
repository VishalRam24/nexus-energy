"""
EC221 -- Magnetohydrodynamic (MHD) Generator -- F2a Physics-Lumped Channel
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MHD_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the MHD generator F2a quasi-1D channel model."""

    component_id = "EC221"
    component_name = "Magnetohydrodynamic (MHD) Generator"
    fidelity = "F2a -- Physics-Lumped 1D Channel Flow with Saha Conductivity + Hall Effect"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MHD_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the quasi-1D channel simulation.

        inputs:
            B_field_T   : float  (default from params, magnetic field)
            K_load      : float  (default from params, load factor)
            u_inlet     : float  (default from params, inlet velocity [m/s])
            T_inlet     : float  (default from params, inlet temperature [K])
            p_inlet     : float  (default from params, inlet pressure [Pa])
            n_points    : int    (default 200, output stations)
        """
        B = inputs.get("B_field_T", None)
        K = inputs.get("K_load", None)
        u_in = inputs.get("u_inlet", None)
        T_in = inputs.get("T_inlet", None)
        p_in = inputs.get("p_inlet", None)
        n_points = int(inputs.get("n_points", 200))

        r = self._model.simulate(B=B, K=K, u_in=u_in, T_in=T_in, p_in=p_in,
                                 n_points=n_points)
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "B_field_T": {"unit": "T", "range": [0.5, 10.0]},
                "K_load": {"unit": "-", "range": [0.05, 0.95]},
                "u_inlet": {"unit": "m/s", "range": [100.0, 2000.0]},
                "T_inlet": {"unit": "K", "range": [2000.0, 4000.0]},
                "p_inlet": {"unit": "Pa", "range": [1e5, 2e6]},
                "n_points": {"unit": "-"},
            },
            "outputs": {
                "x": "m (channel coordinate)",
                "u": "m/s (velocity profile)",
                "T": "K (temperature profile)",
                "p": "Pa (pressure profile)",
                "sigma": "S/m (Saha conductivity)",
                "sigma_eff": "S/m (Hall-reduced)",
                "J": "A/m^2 (current density)",
                "power_density": "W/m^3",
                "P_elec_W": "W (total electrical power)",
                "eta_electric": "- (electrical conversion efficiency)",
                "eta_enthalpy_extraction": "-",
                "beta_hall": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"K_load": 0.5})
    print(f"P_elec = {r['P_elec_W']/1e6:.3f} MW, "
          f"eta_electric = {r['eta_electric']:.3f}, "
          f"beta_hall = {r['beta_hall']:.2f}, "
          f"sigma_in = {r['sigma'][0]:.1f} S/m, "
          f"u: {r['u'][0]:.0f} -> {r['u'][-1]:.0f} m/s, "
          f"T: {r['T'][0]:.0f} -> {r['T'][-1]:.0f} K")
