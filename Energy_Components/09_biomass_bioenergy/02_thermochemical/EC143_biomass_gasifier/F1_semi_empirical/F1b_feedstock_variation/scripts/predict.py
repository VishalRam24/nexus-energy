"""EC143 -- Biomass Gasifier -- F1b Feedstock Variation -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BiomassGasifierF1b


class ComponentModel:
    """Standardized interface for EC143 Biomass Gasifier -- F1b feedstock model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BiomassGasifierF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "feedstock_type":      str, e.g. "wood", "rice_husk", "pine"
                "equivalence_ratio":   float [0.15-0.45]
                "moisture_content":    float [0-0.5] (default 0.1)
                "feed_rate_kg_h":      float [10-10000] (default 100)
            }

        Returns:
            dict with: syngas_composition, syngas_yield_nm3_kg,
                       cold_gas_efficiency, tar_content_g_nm3, lhv_syngas_mj_nm3
        """
        return self._model.predict(
            feedstock_type=str(inputs["feedstock_type"]),
            equivalence_ratio=float(inputs.get("equivalence_ratio", 0.25)),
            moisture_content=float(inputs.get("moisture_content", 0.1)),
            feed_rate_kg_h=float(inputs.get("feed_rate_kg_h", 100.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Biomass Gasifier",
            "ec_id": "EC143",
            "fidelity": "F1b",
            "model": "Feedstock-Specific with Ultimate Analysis",
            "description": (
                f"Feedstock-specific gasification model with {len(m.feedstock_db)} feedstocks. "
                "Ultimate analysis (C,H,O,N,S) drives syngas composition. "
                "ER effect: higher ER -> more CO2/less CO. "
                "Moisture correction on composition and yield."
            ),
            "inputs": {
                "feedstock_type":    {"type": "str", "options": list(m.feedstock_db.keys())},
                "equivalence_ratio": {"unit": "dimensionless", "range": [0.15, 0.45]},
                "moisture_content":  {"unit": "dimensionless", "range": [0.0, 0.5], "default": 0.1},
                "feed_rate_kg_h":    {"unit": "kg/h", "range": [10.0, 10000.0], "default": 100.0},
            },
            "outputs": {
                "syngas_composition":  {"unit": "mole_fraction", "keys": ["CO", "H2", "CO2", "CH4", "N2"]},
                "syngas_yield_nm3_kg": {"unit": "Nm3/kg"},
                "cold_gas_efficiency": {"unit": "dimensionless"},
                "tar_content_g_nm3":   {"unit": "g/Nm3"},
                "lhv_syngas_mj_nm3":   {"unit": "MJ/Nm3"},
            },
            "source": "Zainal et al. (2001); Basu (2010); Li et al. (2004)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    for fs in ["wood", "rice_husk", "pine"]:
        r = model.predict({
            "feedstock_type": fs,
            "equivalence_ratio": 0.25,
            "moisture_content": 0.1,
        })
        comp = r["syngas_composition"]
        print(f"\n{fs} (ER=0.25, MC=10%):")
        print(f"  CO={comp['CO']:.3f} H2={comp['H2']:.3f} CO2={comp['CO2']:.3f} "
              f"CH4={comp['CH4']:.3f} N2={comp['N2']:.3f}")
        print(f"  LHV={r['lhv_syngas_mj_nm3']:.2f} MJ/Nm3, CGE={r['cold_gas_efficiency']:.3f}")
        print(f"  Tar={r['tar_content_g_nm3']:.1f} g/Nm3, Yield={r['syngas_yield_nm3_kg']:.2f} Nm3/kg")
