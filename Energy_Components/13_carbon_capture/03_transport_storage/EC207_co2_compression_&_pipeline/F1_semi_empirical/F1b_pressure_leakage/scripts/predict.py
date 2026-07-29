"""EC207 — CO2 Compression & Pipeline — F1b Injection Pressure + Leakage — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CO2CompressionPipelineF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2CompressionPipelineF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict CO2 compression and pipeline with injection pressure effects and leakage.

        Parameters
        ----------
        inputs : dict
            P_inlet_bar          : float (bar, default 1.5)
            P_outlet_bar         : float (bar, default 150)
            T_inlet_K            : float (K, default 308.15)
            m_dot_kg_s           : float (kg/s, default 100)
            pipeline_length_km   : float (km, default 100)
            pipeline_diameter_m  : float (m, default 0.3)
            operating_hours      : float (hours, default 0)
        """
        P_in = np.asarray(inputs.get("P_inlet_bar", 1.5), dtype=float)
        P_out = np.asarray(inputs.get("P_outlet_bar", 150.0), dtype=float)
        T_in = np.asarray(inputs.get("T_inlet_K", 308.15), dtype=float)
        m_dot = np.asarray(inputs.get("m_dot_kg_s", 100.0), dtype=float)
        L_km = np.asarray(inputs.get("pipeline_length_km", 100.0), dtype=float)
        D_m = np.asarray(inputs.get("pipeline_diameter_m", 0.3), dtype=float)
        hours = inputs.get("operating_hours", 0.0)

        return self._model.compute(P_in, P_out, m_dot, T_in, L_km, D_m, hours)

    def get_info(self) -> dict:
        return {
            "name": "CO2 Compression & Pipeline",
            "ec_id": "EC207",
            "fidelity": "F1b",
            "description": (
                "CO2 compression with compressor degradation (efficiency loss + seal leakage) "
                "and pipeline leakage model (0.2%/year per 100 km). "
                "Injection pressure effects on required compression level."
            ),
            "inputs": {
                "P_inlet_bar": {"unit": "bar", "range": [0.1, 10.0], "default": 1.5},
                "P_outlet_bar": {"unit": "bar", "range": [100.0, 250.0], "default": 150.0},
                "T_inlet_K": {"unit": "K", "range": [280.0, 400.0], "default": 308.15},
                "m_dot_kg_s": {"unit": "kg/s", "range": [0.0, 1000.0], "default": 100.0},
                "pipeline_length_km": {"unit": "km", "range": [0.0, 500.0], "default": 100.0},
                "pipeline_diameter_m": {"unit": "m", "range": [0.1, 1.0], "default": 0.3},
                "operating_hours": {"unit": "hours", "range": [0.0, 100000.0], "default": 0.0},
            },
            "outputs": {
                "sec_kwh_per_tco2": {"unit": "kWh/tCO2"},
                "shaft_power_kw": {"unit": "kW"},
                "pipeline_dp_bar": {"unit": "bar"},
                "pipeline_outlet_P_bar": {"unit": "bar"},
                "polytropic_efficiency": {"unit": "dimensionless"},
                "seal_leakage_fraction": {"unit": "dimensionless"},
                "pipeline_leakage_fraction": {"unit": "dimensionless"},
                "net_co2_delivered_kg_s": {"unit": "kg/s"},
            },
            "source": "IPCC (2005) CCS SR Ch4; IEA GHG (2007); McCoy & Rubin (2008)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "m_dot_kg_s": 100.0,
                        "pipeline_length_km": 200.0, "operating_hours": 0})
    print("Design point (fresh, 200 km pipeline):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
