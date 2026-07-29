"""EC156 -- GHP -- F1b Ground Thermal Degradation -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GHPF1b


class ComponentModel:
    """Standardized interface for EC156 GHP -- F1b ground thermal degradation model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_sink_c":         float [25-65 degC] — load supply temperature
                "PLR":              float [0.2-1.0] (default 1.0)
                "operation_hours":  float [0-8760] (default 0)
                "TDS_ppm":          float [0-3000] (default 200)
                "heat_rate_kw":     float (optional, default rated*PLR)
                "mode":             str "heating" or "cooling" (default "heating")
            }
        """
        return self._model.predict(
            T_sink_c=float(inputs.get("T_sink_c", 35.0)),
            PLR=float(inputs.get("PLR", 1.0)),
            operation_hours=float(inputs.get("operation_hours", 0.0)),
            TDS_ppm=float(inputs.get("TDS_ppm", 200.0)),
            heat_rate_kw=inputs.get("heat_rate_kw", None),
            mode=str(inputs.get("mode", "heating")),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Geothermal Heat Pump (GHP)",
            "ec_id": "EC156",
            "fidelity": "F1b",
            "model": "Ground Thermal Saturation + Brine Fouling + Part-Load",
            "description": (
                "GHP extending F1a COP map with: "
                "ground thermal saturation over heating season (tau=800h), "
                "brine TDS-dependent fouling factor, "
                "part-load COP curve, "
                "condenser temperature sensitivity."
            ),
            "inputs": {
                "T_sink_c":        {"unit": "degC",   "range": [25.0, 65.0], "default": 35.0},
                "PLR":             {"unit": "—",      "range": [0.20, 1.0],  "default": 1.0},
                "operation_hours": {"unit": "hours",  "range": [0.0, 8760.0],"default": 0.0},
                "TDS_ppm":         {"unit": "ppm",    "range": [0.0, 3000.0],"default": 200.0},
                "mode":            {"type": "str",    "options": ["heating", "cooling"]},
            },
            "outputs": {
                "cop_heating":           {"unit": "dimensionless"},
                "cop_cooling":           {"unit": "dimensionless"},
                "cop_effective":         {"unit": "dimensionless"},
                "T_source_effective":    {"unit": "degC"},
                "ground_dT":             {"unit": "K"},
                "fouling_factor":        {"unit": "dimensionless"},
                "part_load_factor":      {"unit": "dimensionless"},
                "heating_capacity_kw":   {"unit": "kW_th"},
                "electrical_input_kw":   {"unit": "kW_e"},
                "cop_advantage_over_ashp":{"unit": "dimensionless"},
            },
            "source": "Staffell et al. (2012) E&ES; ASHRAE (2011); Kavanaugh & Rafferty (2014); Yang et al. (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for t_op in [0, 500, 2000]:
        r = model.predict({"T_sink_c": 35.0, "operation_hours": float(t_op), "TDS_ppm": 500.0})
        print(f"t_op={t_op}h: T_src_eff={r['T_source_effective']:.1f}°C, "
              f"COP_eff={r['cop_effective']:.2f}, fouling={r['fouling_factor']:.3f}")
