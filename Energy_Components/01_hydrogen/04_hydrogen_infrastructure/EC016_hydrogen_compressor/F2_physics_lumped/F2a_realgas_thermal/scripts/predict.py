"""
EC016 -- Hydrogen Compressor -- F2a Real-Gas Multistage Reciprocating
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import H2CompressorRealGasThermal

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC016 F2a real-gas thermal compressor model."""

    component_id = "EC016"
    component_name = "Hydrogen Compressor"
    fidelity = "F2a -- Real-Gas Multistage Reciprocating Compression with Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = H2CompressorRealGasThermal(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run transient compressor simulation.

        inputs:
            mass_flow_kg_s : float (or callable) -- default 0.0139 (50 kg/h)
            P_in_bar       : float -- default = parameter P_inlet
            P_out_bar      : float -- default = parameter P_outlet_max
            T_inlet_K      : float -- default = parameter T_inlet
            T_coolant_K    : float -- default = parameter T_coolant
            eps_ic         : float -- intercooler effectiveness, default param
            T_metal0_K     : float -- initial metal T, default ambient
            dt             : float -- output step [s], default 10.0
            duration_s     : float -- horizon [s], default 1800.0
        """
        m = self._model
        m_dot = inputs.get("mass_flow_kg_s", 50.0 / 3600.0)
        P_in = inputs.get("P_in_bar", m.P_inlet_default)
        P_out = inputs.get("P_out_bar", m.P_out_max)
        T_in = inputs.get("T_inlet_K", None)
        T_cool = inputs.get("T_coolant_K", None)
        eps = inputs.get("eps_ic", None)
        T0 = inputs.get("T_metal0_K", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 1800.0)

        return m.simulate(m_dot, P_in, P_out, T_in, T_cool, eps, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mass_flow_kg_s": {"unit": "kg/s", "range": [0.0, 0.05]},
                "P_in_bar": {"unit": "bar", "range": [1.0, 100.0]},
                "P_out_bar": {"unit": "bar", "range": [10.0, 1000.0]},
                "T_inlet_K": {"unit": "K", "range": [263.0, 333.0]},
                "T_coolant_K": {"unit": "K", "range": [263.0, 333.0]},
                "eps_ic": {"unit": "-", "range": [0.0, 1.0]},
                "T_metal0_K": {"unit": "K"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_metal": "K",
                "shaft_power_kW": "kW",
                "heat_rejected_kW": "kW",
                "heat_dissipated_kW": "kW",
                "SEC_kWh_kg": "kWh/kg",
                "isentropic_efficiency": "-",
                "compression_efficiency": "-",
                "T_discharge_final_K": "K",
                "stage_profile": "dict of per-stage arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    cm = ComponentModel()
    print(cm.get_info())
    r = cm.predict({"P_in_bar": 20.0, "P_out_bar": 900.0, "duration_s": 600.0, "dt": 30.0})
    print(
        f"SEC={r['SEC_kWh_kg']:.3f} kWh/kg | "
        f"shaft={r['shaft_power_kW'][0]:.2f} kW | "
        f"eta_isen={r['isentropic_efficiency']:.3f} | "
        f"T_disc_final={r['T_discharge_final_K']:.1f} K | "
        f"T_metal {r['T_metal'][0]:.1f}->{r['T_metal'][-1]:.1f} K"
    )
    print("  Stage Z:", [round(z, 3) for z in r["stage_profile"]["Z"]])
