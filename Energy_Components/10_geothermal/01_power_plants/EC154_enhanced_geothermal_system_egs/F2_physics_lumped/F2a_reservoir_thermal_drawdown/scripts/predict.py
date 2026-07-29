"""
EC154 -- Enhanced Geothermal System (EGS) -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EGS_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EGS F2a reservoir-thermal-drawdown model."""

    component_id = "EC154"
    component_name = "Enhanced Geothermal System (EGS)"
    fidelity = "F2a -- Physics-Lumped Reservoir Thermal Drawdown + Binary/ORC Cycle"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = EGS_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run multi-year reservoir thermal-drawdown simulation.

        inputs:
            years            : float  horizon in years   (default 30.0)
            n_points         : int    number of samples  (default 200)
            m_dot_kg_s       : float  circulation flow   (default from params)
            T_geo_init_degC  : float  initial rock T     (default from params)
        """
        years = inputs.get("years", 30.0)
        n_points = int(inputs.get("n_points", 200))
        m_dot = inputs.get("m_dot_kg_s", None)
        T0 = inputs.get("T_geo_init_degC", None)
        return self._model.simulate(years=years, n_points=n_points,
                                    m_dot=m_dot, T_geo_init_degC=T0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "years": {"unit": "years", "range": [0.0, 60.0]},
                "n_points": {"unit": "-"},
                "m_dot_kg_s": {"unit": "kg/s", "range": [5.0, 200.0]},
                "T_geo_init_degC": {"unit": "degC", "range": [150.0, 350.0]},
            },
            "outputs": {
                "t_years": "years",
                "T_rock_degC": "degC (reservoir bulk temperature)",
                "T_prod_degC": "degC (produced fluid temperature)",
                "Q_in_kW": "kW (heat to cycle)",
                "eta_carnot": "- (Carnot bound)",
                "eta_cycle": "- (cycle efficiency)",
                "P_gross_kW": "kW",
                "P_pump_kW": "kW (parasitic)",
                "P_net_kW": "kW (net electrical)",
                "tau_res_yr": "years (drawdown time constant)",
                "effectiveness": "- (fracture HX effectiveness)",
                "energy_balance_err": "- (relative energy-conservation residual)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"years": 30.0, "n_points": 200})
    print(f"\ntau_res = {r['tau_res_yr']:.2f} yr, HX effectiveness = {r['effectiveness']:.3f}")
    print(f"Year  0: T_rock={r['T_rock_degC'][0]:.1f}C  T_prod={r['T_prod_degC'][0]:.1f}C  "
          f"P_net={r['P_net_kW'][0]:.0f} kW  eta_cycle={r['eta_cycle'][0]:.3f}")
    print(f"Year {r['t_years'][-1]:.0f}: T_rock={r['T_rock_degC'][-1]:.1f}C  T_prod={r['T_prod_degC'][-1]:.1f}C  "
          f"P_net={r['P_net_kW'][-1]:.0f} kW  eta_cycle={r['eta_cycle'][-1]:.3f}")
    print(f"Energy-balance residual: {r['energy_balance_err']:.2e}")
