"""EC146 -- Torrefaction Reactor -- F1b Feedstock Variation -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TorrefactionReactorF1b


class ComponentModel:
    """Standardized interface for EC146 Torrefaction Reactor -- F1b feedstock variation model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TorrefactionReactorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "feedstock_type":      str, e.g. "wood_chips", "pine", "wheat_straw"
                "temperature_degC":    float [200-300]
                "residence_time_min":  float [5-120] (default 30)
                "moisture_fraction":   float [0-0.30] (default 0.10)
                "PLR":                 float [0.2-1.0] (default 1.0)
                "feed_rate_kg_h":      float (default 1000)
            }
        Returns:
            dict with: mass_yield, energy_densification, energy_yield,
                       torrefied_LHV_MJ_kg, LHV_eff_MJ_kg, moisture_lhv_factor,
                       thermal_efficiency, solid_rate_kg_h
        """
        return self._model.predict(
            feedstock_type=str(inputs["feedstock_type"]),
            temperature_degC=float(inputs.get("temperature_degC", 250.0)),
            residence_time_min=float(inputs.get("residence_time_min", 30.0)),
            moisture_fraction=float(inputs.get("moisture_fraction", 0.10)),
            PLR=float(inputs.get("PLR", 1.0)),
            feed_rate_kg_h=float(inputs.get("feed_rate_kg_h", 1000.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Torrefaction Reactor",
            "ec_id": "EC146",
            "fidelity": "F1b",
            "model": "Feedstock-Specific with Moisture-LHV Coupling",
            "description": (
                f"Feedstock-specific torrefaction with {len(m.feedstock_db)} feedstocks. "
                "Moisture-LHV: LHV_eff = LHV_dry*(1-M) - h_fg*M. "
                "Mass yield model: MY=exp(-k*(dT^n)*(t^p)). "
                "Energy densification increases with temperature and lignin content."
            ),
            "inputs": {
                "feedstock_type":     {"type": "str", "options": list(m.feedstock_db.keys())},
                "temperature_degC":   {"unit": "degC", "range": [200.0, 300.0]},
                "residence_time_min": {"unit": "min",  "range": [5.0, 120.0], "default": 30.0},
                "moisture_fraction":  {"unit": "—",    "range": [0.0, 0.30],  "default": 0.10},
                "PLR":                {"unit": "—",    "range": [0.20, 1.0],  "default": 1.0},
                "feed_rate_kg_h":     {"unit": "kg/h", "range": [10.0, 10000.0]},
            },
            "outputs": {
                "mass_yield":           {"unit": "kg/kg_dry"},
                "energy_densification": {"unit": "dimensionless"},
                "energy_yield":         {"unit": "dimensionless"},
                "torrefied_LHV_MJ_kg":  {"unit": "MJ/kg"},
                "LHV_eff_MJ_kg":        {"unit": "MJ/kg_wet"},
                "moisture_lhv_factor":  {"unit": "dimensionless"},
                "thermal_efficiency":   {"unit": "dimensionless"},
                "solid_rate_kg_h":      {"unit": "kg/h"},
            },
            "source": "Bach et al. (2017) Fuel; van der Stelt et al. (2011) Biomass & Bioenergy; Bergman et al. (2005) ECN",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    for fs in ["wood_chips", "wheat_straw", "pine"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 250.0, "residence_time_min": 30.0, "moisture_fraction": 0.10})
        print(f"\n{fs} (250°C, 30 min, M=10%):")
        print(f"  MY={r['mass_yield']:.3f}  EDR={r['energy_densification']:.3f}  EY={r['energy_yield']:.3f}")
        print(f"  LHV_torr={r['torrefied_LHV_MJ_kg']:.2f} MJ/kg  eta_th={r['thermal_efficiency']:.3f}")
