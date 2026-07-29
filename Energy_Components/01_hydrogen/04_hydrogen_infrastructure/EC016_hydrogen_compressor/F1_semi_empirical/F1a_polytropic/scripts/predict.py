"""EC016 — Hydrogen Compressor — F1a Polytropic — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import H2CompressorF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = H2CompressorF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict H2 compressor performance.

        Args:
            inputs: dict with keys:
                - P_inlet  (bar)
                - P_outlet (bar)
                - T_inlet  (K),    optional, default 298.15
                - m_dot    (kg/s), optional, default 0.014 kg/s (~50 kg/h)

        Returns:
            dict with keys:
                - specific_work_kJ_per_kg, sec_kwh_per_kg, shaft_power_kw,
                  stage_pressure_ratio, stage_discharge_T_K, compression_efficiency
        """
        P_in = np.asarray(inputs["P_inlet"], dtype=float)
        P_out = np.asarray(inputs["P_outlet"], dtype=float)
        T_in = np.asarray(inputs.get("T_inlet", 298.15), dtype=float)
        m_dot = np.asarray(inputs.get("m_dot", 0.014), dtype=float)

        return {
            "specific_work_kJ_per_kg": self._model.specific_work_kJ_per_kg(P_in, P_out, T_in),
            "sec_kwh_per_kg":          self._model.sec_kwh_per_kg(P_in, P_out, T_in),
            "shaft_power_kw":          self._model.shaft_power_kw(m_dot, P_in, P_out, T_in),
            "stage_pressure_ratio":    self._model.stage_pressure_ratio(P_in, P_out),
            "stage_discharge_T_K":     self._model.stage_discharge_temperature(T_in, P_in, P_out),
            "compression_efficiency":  self._model.compression_efficiency(P_in, P_out, T_in),
        }

    def get_info(self) -> dict:
        return {
            "name": "Hydrogen Compressor",
            "ec_id": "EC016",
            "fidelity": "F1a",
            "description": "Multistage polytropic compression with optimum stage ratio and intercooling",
            "inputs": {
                "P_inlet":  {"unit": "bar",  "range": [1.0, 100.0]},
                "P_outlet": {"unit": "bar",  "range": [10.0, 1000.0]},
                "T_inlet":  {"unit": "K",    "range": [263.0, 333.0], "default": 298.15},
                "m_dot":    {"unit": "kg/s", "range": [0.0, 0.05],    "default": 0.014},
            },
            "outputs": {
                "specific_work_kJ_per_kg": {"unit": "kJ/kg"},
                "sec_kwh_per_kg":          {"unit": "kWh/kg"},
                "shaft_power_kw":          {"unit": "kW"},
                "stage_pressure_ratio":    {"unit": "dimensionless"},
                "stage_discharge_T_K":     {"unit": "K"},
                "compression_efficiency":  {"unit": "dimensionless"},
            },
            "source": "Sdanghi et al. (2019); Bossel (2006)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for P_out in [200, 350, 500, 700, 900]:
        r = model.predict({"P_inlet": 20.0, "P_outlet": P_out, "T_inlet": 298.15, "m_dot": 0.014})
        print(f"P_out={P_out} bar: w={float(r['specific_work_kJ_per_kg']):.1f} kJ/kg, "
              f"SEC={float(r['sec_kwh_per_kg']):.3f} kWh/kg, "
              f"P={float(r['shaft_power_kw']):.2f} kW, "
              f"T_stage={float(r['stage_discharge_T_K']):.1f} K")
