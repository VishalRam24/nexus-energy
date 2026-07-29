"""EC050 — OPV — F1a Single-Diode Model — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import OPVSingleDiodeF1a


class ComponentModel:
    component_id = "EC050"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OPVSingleDiodeF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        G = inputs.get("G", 1000.0)
        if np.ndim(G) == 0:
            return self._model.predict(G=float(G))
        # vectorized over G
        results = [self._model.predict(G=float(g)) for g in np.asarray(G, dtype=float)]
        return {k: np.array([r[k] for r in results]) for k in results[0]}

    def get_info(self) -> dict:
        return {
            "name": "Organic Photovoltaic (OPV)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "5-parameter single-diode model, I_L=8mA/cm², n=2.0, Voc≈0.8V",
            "inputs": {
                "G": {"unit": "W/m2", "range": [0.0, 1200.0]},
            },
            "outputs": {
                "Voc_V": {"unit": "V"},
                "Isc_A": {"unit": "A"},
                "Vmp_V": {"unit": "V"},
                "Imp_A": {"unit": "A"},
                "Pmp_W": {"unit": "W"},
                "FF": {"unit": "dimensionless"},
                "eta": {"unit": "dimensionless"},
            },
            "source": "De Soto et al. (2006); Brabec et al. (2001)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"G": 1000.0})
    print(f"Voc={r['Voc_V']:.3f} V, Pmp={r['Pmp_W']:.2f} W, eta={r['eta']:.3f}, FF={r['FF']:.3f}")
