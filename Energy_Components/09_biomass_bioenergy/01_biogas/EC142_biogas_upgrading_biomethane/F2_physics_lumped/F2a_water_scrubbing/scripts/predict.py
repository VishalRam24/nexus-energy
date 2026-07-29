"""
EC142 -- Biogas Upgrading to Biomethane -- F2a Physics-Lumped Water Scrubbing
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BiogasUpgradingF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC142 F2a high-pressure water-scrubbing model."""

    component_id = "EC142"
    component_name = "Biogas Upgrading to Biomethane"
    fidelity = "F2a -- Physics-Lumped High-Pressure Water Scrubbing (Henry/NTU absorption ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BiogasUpgradingF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the absorption-column transient and return performance.

        inputs:
            biogas_flow_Nm3_per_h : float   raw biogas feed [Nm3/h]
            CH4_fraction_in       : float   CH4 mole fraction of raw biogas [-]
            T_col_K               : float   column temperature [K] (optional)
            dt                    : float   output step [s]   (default 5.0)
            duration_s            : float   horizon [s]       (default 300.0)
        """
        Q = inputs.get("biogas_flow_Nm3_per_h", 500.0)
        x = inputs.get("CH4_fraction_in", 0.60)
        T = inputs.get("T_col_K", None)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 300.0)
        return self._model.simulate(Q, x, T_col_K=T, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "biogas_flow_Nm3_per_h": {"unit": "Nm3/h", "range": [1.0, 2000.0]},
                "CH4_fraction_in": {"unit": "-", "range": [0.40, 0.75]},
                "T_col_K": {"unit": "K", "range": [278.15, 313.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "purity_CH4": "- (mole fraction in product)",
                "CH4_recovery": "- (<1)",
                "CH4_slip": "- (fraction of feed CH4 lost)",
                "CO2_removal": "-",
                "biomethane_Nm3_per_h": "Nm3/h",
                "SEC_kWh_per_Nm3": "kWh/Nm3 product",
                "C_CO2_liquid": "mol/m3",
                "C_CH4_liquid": "mol/m3",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"biogas_flow_Nm3_per_h": 500.0, "CH4_fraction_in": 0.60,
                   "dt": 5.0, "duration_s": 300.0})
    print(f"Steady-state: purity={r['purity_CH4_ss']:.4f}, "
          f"recovery={r['CH4_recovery_ss']:.4f}, slip={r['CH4_slip_ss']:.4f}, "
          f"CO2_removal={r['CO2_removal_ss']:.4f}, "
          f"SEC={r['SEC_kWh_per_Nm3']:.3f} kWh/Nm3, "
          f"biomethane={r['biomethane_Nm3_per_h'][-1]:.1f} Nm3/h")
