"""
EC207 -- CO2 Compression & Pipeline -- F2a Real-Gas Compression + Pipeline
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2CompressionPipelineF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC207 F2a physics-lumped model."""

    component_id = "EC207"
    component_name = "CO2 Compression & Pipeline"
    fidelity = "F2a -- Real-Gas Multistage Compression + Dense-Phase Pipeline (Pressure-Transient ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            for k, v in params.items():
                if k in self._raw:
                    self._raw[k].update(v)
        self._model = CO2CompressionPipelineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Compute compression train + pipeline + pressure transient.

        inputs:
            P_inlet_bar     : float  (default 1.5)
            P_outlet_bar    : float  (default 150.0, supercritical target)
            T_inlet_K       : float  (default 308.15)
            mass_flow_kg_s  : float  (default 100.0)
            pipeline_length_km : float (default 100.0)
            diameter_m      : float  (default 0.508)
            P_delivery_bar  : float  (default 80.0)
            duration_s      : float  (default 300.0)
            run_transient   : bool   (default True)
        """
        P_in = inputs.get("P_inlet_bar", self._model.P_inlet_default)
        P_out = inputs.get("P_outlet_bar", self._model.P_sc)
        T_in = inputs.get("T_inlet_K", self._model.T_inlet_default)
        m_dot = inputs.get("mass_flow_kg_s", self._model.m_dot_default)
        L = inputs.get("pipeline_length_km", self._model.L_default)
        D = inputs.get("diameter_m", self._model.D_default)
        P_del = inputs.get("P_delivery_bar", 80.0)
        dur = inputs.get("duration_s", 300.0)
        run_transient = inputs.get("run_transient", True)

        comp = self._model.compress(P_in, P_out, T_in)
        shaft_kw = m_dot * comp["w_specific_J_per_kg"] / 1000.0
        dP_pipe = self._model.pipeline_pressure_drop_bar(m_dot, L, D)
        supercritical = self._model.is_supercritical(comp["T_after_intercool_K"], P_out)

        out = {
            "stage_T_in_K": comp["stage_T_in"],
            "stage_T_discharge_K": comp["stage_T_discharge"],
            "stage_P_in_bar": comp["stage_P_in"],
            "stage_P_out_bar": comp["stage_P_out"],
            "stage_work_J_per_kg": comp["stage_work"],
            "pressure_ratio": comp["pressure_ratio"],
            "specific_work_J_per_kg": comp["w_specific_J_per_kg"],
            "SEC_kWh_per_tCO2": comp["SEC_kWh_per_tCO2"],
            "shaft_power_kW": shaft_kw,
            "P_discharge_bar": comp["P_discharge_bar"],
            "Z_inlet": self._model.z_factor(T_in, P_in),
            "Z_discharge": self._model.z_factor(comp["T_after_intercool_K"], P_out),
            "rho_discharge_kg_m3": self._model.density_real(comp["T_after_intercool_K"], P_out),
            "pipeline_dP_bar": dP_pipe,
            "supercritical": bool(supercritical),
        }
        if run_transient:
            tr = self._model.simulate_pressure_transient(
                m_in_kg_s=m_dot, P_delivery_bar=P_del, length_km=L,
                diameter_m=D, duration_s=dur)
            out["transient"] = tr
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_inlet_bar": {"unit": "bar", "range": [0.1, 10.0]},
                "P_outlet_bar": {"unit": "bar", "range": [100.0, 250.0]},
                "T_inlet_K": {"unit": "K", "range": [300.0, 330.0]},
                "mass_flow_kg_s": {"unit": "kg/s", "range": [0.0, 1000.0]},
                "pipeline_length_km": {"unit": "km", "range": [0.0, 500.0]},
                "diameter_m": {"unit": "m", "range": [0.1, 1.2]},
                "P_delivery_bar": {"unit": "bar", "range": [50.0, 150.0]},
                "duration_s": {"unit": "s", "range": [1.0, 36000.0]},
            },
            "outputs": {
                "SEC_kWh_per_tCO2": "kWh/tCO2",
                "specific_work_J_per_kg": "J/kg",
                "shaft_power_kW": "kW",
                "P_discharge_bar": "bar",
                "Z_inlet": "-",
                "Z_discharge": "-",
                "pipeline_dP_bar": "bar",
                "supercritical": "bool",
                "transient": "dict of time-series arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mass_flow_kg_s": 100.0, "duration_s": 200.0})
    print(f"SEC = {r['SEC_kWh_per_tCO2']:.1f} kWh/tCO2 | "
          f"shaft = {r['shaft_power_kW']:.0f} kW | "
          f"P_disch = {r['P_discharge_bar']:.0f} bar | "
          f"Z_in = {r['Z_inlet']:.3f} Z_out = {r['Z_discharge']:.3f} | "
          f"dP_pipe = {r['pipeline_dP_bar']:.1f} bar | "
          f"supercritical = {r['supercritical']}")
    tr = r["transient"]
    print(f"Pressure transient: P0 = {tr['P_discharge_bar'][0]:.1f} -> "
          f"P_final = {tr['P_discharge_bar'][-1]:.1f} bar "
          f"(SS = {tr['P_steady_state_bar']:.1f} bar)")
