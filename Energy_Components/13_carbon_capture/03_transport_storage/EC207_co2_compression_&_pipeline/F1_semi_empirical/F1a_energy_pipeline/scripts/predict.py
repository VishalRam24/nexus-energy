"""EC207 — CO2 Compression & Pipeline — F1a — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CO2CompressionPipelineF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2CompressionPipelineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict CO2 compression and pipeline performance.

        Args:
            inputs: dict with keys:
                - P_inlet_bar     : compressor inlet pressure [bar], default 1.5
                - P_outlet_bar    : compressor outlet / pipeline inlet pressure [bar], default 150
                - T_inlet_K       : compressor inlet temperature [K], default 308.15
                - m_dot_kg_s      : CO2 mass flow [kg/s], default 100.0
                - pipeline_length_km  : pipeline length [km], optional, default 0 (no pipeline)
                - pipeline_diameter_m : pipeline diameter [m], optional, default 0.3

        Returns:
            dict with keys:
                - sec_kwh_per_tco2      : specific energy [kWh/tCO2]
                - shaft_power_kw        : compressor power [kW]
                - stage_pressure_ratio  : per-stage compression ratio [-]
                - stage_discharge_T_K   : stage discharge T (before intercooling) [K]
                - pipeline_dp_bar       : pipeline pressure drop [bar]
                - pipeline_outlet_P_bar : estimated pipeline outlet pressure [bar]
                - is_supercritical_in   : inlet is supercritical [bool]
                - is_supercritical_out  : outlet is supercritical [bool]
        """
        P_in = np.asarray(inputs.get("P_inlet_bar", 1.5), dtype=float)
        P_out = np.asarray(inputs.get("P_outlet_bar", 150.0), dtype=float)
        T_in = np.asarray(inputs.get("T_inlet_K", 308.15), dtype=float)
        m_dot = np.asarray(inputs.get("m_dot_kg_s", 100.0), dtype=float)
        L_km = np.asarray(inputs.get("pipeline_length_km", 0.0), dtype=float)
        D_m = np.asarray(inputs.get("pipeline_diameter_m", 0.3), dtype=float)

        pipeline_dp = self._model.pipeline_pressure_drop_bar(m_dot, L_km, D_m)
        P_pipeline_out = np.maximum(P_out - pipeline_dp, 0.0)

        return {
            "sec_kwh_per_tco2":      self._model.sec_kwh_per_tco2(P_in, P_out, T_in),
            "shaft_power_kw":        self._model.shaft_power_kw(m_dot, P_in, P_out, T_in),
            "stage_pressure_ratio":  self._model.stage_pressure_ratio(P_in, P_out),
            "stage_discharge_T_K":   self._model.stage_discharge_temperature(T_in, P_in, P_out),
            "pipeline_dp_bar":       pipeline_dp,
            "pipeline_outlet_P_bar": P_pipeline_out,
            "is_supercritical_in":   self._model.is_supercritical(T_in, P_in),
            "is_supercritical_out":  self._model.is_supercritical(T_in, P_out),
        }

    def get_info(self) -> dict:
        return {
            "name": "CO2 Compression & Pipeline",
            "ec_id": "EC207",
            "fidelity": "F1a",
            "description": "Polytropic CO2 compression to supercritical + Darcy-Weisbach dense-phase pipeline",
            "inputs": {
                "P_inlet_bar":        {"unit": "bar",  "range": [0.1, 10.0],   "default": 1.5},
                "P_outlet_bar":       {"unit": "bar",  "range": [100.0, 250.0], "default": 150.0},
                "T_inlet_K":          {"unit": "K",    "range": [280.0, 400.0], "default": 308.15},
                "m_dot_kg_s":         {"unit": "kg/s", "range": [0.0, 1000.0],  "default": 100.0},
                "pipeline_length_km": {"unit": "km",   "range": [0.0, 500.0],   "default": 0.0},
                "pipeline_diameter_m":{"unit": "m",    "range": [0.1, 1.0],     "default": 0.3},
            },
            "outputs": {
                "sec_kwh_per_tco2":      {"unit": "kWh/tCO2"},
                "shaft_power_kw":        {"unit": "kW"},
                "stage_pressure_ratio":  {"unit": "dimensionless"},
                "stage_discharge_T_K":   {"unit": "K"},
                "pipeline_dp_bar":       {"unit": "bar"},
                "pipeline_outlet_P_bar": {"unit": "bar"},
                "is_supercritical_in":   {"unit": "bool"},
                "is_supercritical_out":  {"unit": "bool"},
            },
            "source": "IPCC (2005) CCS SR Ch4; McCoy & Rubin (2008)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("Compression 1.5→150 bar:")
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0,
                       "T_inlet_K": 308.15, "m_dot_kg_s": 100.0,
                       "pipeline_length_km": 200.0, "pipeline_diameter_m": 0.3})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4g}")
