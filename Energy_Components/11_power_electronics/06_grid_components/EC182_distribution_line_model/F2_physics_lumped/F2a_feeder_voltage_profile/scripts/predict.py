"""
EC182 -- Distribution Line Model -- F2a Physics-Lumped Feeder Voltage-Profile ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FeederVoltageProfileModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC182 F2a 1D feeder voltage-profile ODE model."""

    component_id = "EC182"
    component_name = "Distribution Line Model (Radial Feeder)"
    fidelity = "F2a -- Physics-Lumped 1D Feeder Voltage-Profile ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FeederVoltageProfileModel(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Solve the distributed-load feeder voltage profile.

        inputs:
            V_s_kV       : float  sending-end (substation) line-to-line voltage [kV]
            P_total_kW   : float  total distributed active load [kW]
            Q_total_kVAR : float  total distributed reactive load [kVAR] (+ inductive)
            length_km    : float  (optional) feeder length [km]
            n_sections   : int    (optional) spatial discretization points
            load_model   : str    (optional) 'constant_power' | 'constant_impedance'
                                   | 'constant_current'
        """
        return self._model.compute(
            V_s_kV=inputs.get("V_s_kV", self._model.V_base_kV),
            P_total_kW=inputs.get("P_total_kW", 1000.0),
            Q_total_kVAR=inputs.get("Q_total_kVAR", 400.0),
            length_km=inputs.get("length_km", None),
            n_sections=inputs.get("n_sections", None),
            load_model=inputs.get("load_model", "constant_power"),
        )

    def get_info(self) -> dict:
        u = self._raw["unit"]
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "ec_id": self.component_id,
            "fidelity": self.fidelity,
            "version": self.version,
            "description": "1D distributed-load radial feeder; dV/dx=-(r+jx)I, "
                           "dI/dx=-i_load; shooting BVP via solve_ivp + root; "
                           "no shunt (no Ferranti); high R/X; losses = int 3 I^2 r dx.",
            "inputs": {
                "V_s_kV": {"unit": "kV", "range": [4.0, 35.0]},
                "P_total_kW": {"unit": "kW", "range": [0.0, 5000.0]},
                "Q_total_kVAR": {"unit": "kVAR", "range": [-2000.0, 2000.0]},
                "length_km": {"unit": "km", "range": [0.1, 50.0], "optional": True},
                "n_sections": {"unit": "-", "range": [4, 500], "optional": True},
                "load_model": {"unit": "-", "optional": True},
            },
            "outputs": {
                "x_km": "km (spatial coordinate along feeder)",
                "V_profile_kV": "kV (L-L voltage along feeder)",
                "I_profile_A": "A (phase current along feeder)",
                "P_flow_kW": "kW (active power flow past each point)",
                "V_r_kV": "kV (far-end voltage)",
                "P_loss_kW": "kW",
                "efficiency": "-",
                "voltage_drop_pct": "%",
                "min_voltage_pu": "-",
                "ansi_compliant": "bool",
                "r_over_x": "-",
                "energy_balance_residual_kW": "kW",
            },
            "params": {
                "V_base": f"{u['V_base_kV']['value']} kV",
                "R_ohm_per_km": u["R_ohm_per_km"]["value"],
                "X_ohm_per_km": u["X_ohm_per_km"]["value"],
                "default_length_km": u["length_km"]["value"],
                "R_over_X": u["R_ohm_per_km"]["value"] / u["X_ohm_per_km"]["value"],
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"V_s_kV": 11.0, "P_total_kW": 1500.0,
                   "Q_total_kVAR": 600.0, "length_km": 8.0})
    print(f"V_r={r['V_r_kV']:.3f} kV  P_loss={r['P_loss_kW']:.2f} kW  "
          f"dV={r['voltage_drop_pct']:.2f}%  eta={r['efficiency']*100:.2f}%  "
          f"min_V={r['min_voltage_pu']*100:.1f}%  ANSI={r['ansi_compliant']}  "
          f"R/X={r['r_over_x']:.2f}")
