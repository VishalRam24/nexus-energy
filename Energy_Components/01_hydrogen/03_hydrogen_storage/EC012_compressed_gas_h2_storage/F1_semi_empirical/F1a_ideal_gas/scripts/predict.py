"""EC012 — Compressed Gas H2 Storage — F1a Ideal Gas — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CompressedGasH2F1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CompressedGasH2F1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict compressed H2 storage state.

        Args:
            inputs: dict with keys:
                - pressure    (bar):  Tank pressure
                - temperature (K):    Tank temperature

        Returns:
            dict with keys:
                - stored_mass_kg, energy_stored_MJ, fill_fraction,
                  compression_work_kJ_per_kg, gravimetric_wt_pct,
                  volumetric_kg_per_m3, compressibility_Z
        """
        P = np.asarray(inputs["pressure"], dtype=float)
        T = np.asarray(inputs["temperature"], dtype=float)

        return {
            "stored_mass_kg":           self._model.stored_mass(P, T),
            "energy_stored_MJ":         self._model.energy_stored(P, T),
            "fill_fraction":            self._model.fill_fraction(P, T),
            "compression_work_kJ_per_kg": self._model.compression_work(
                self._model.P_inlet, P, self._model.T_inlet),
            "gravimetric_wt_pct":       self._model.gravimetric_density(P, T),
            "volumetric_kg_per_m3":     self._model.volumetric_density(P, T),
            "compressibility_Z":        self._model.compressibility_factor(P),
        }

    def get_info(self) -> dict:
        return {
            "name": "Compressed Gas H2 Storage",
            "ec_id": "EC012",
            "fidelity": "F1a",
            "description": "PV=nZRT with compressibility factor; isentropic compression work",
            "inputs": {
                "pressure":    {"unit": "bar", "range": [1.0, 900.0]},
                "temperature": {"unit": "K",   "range": [233.0, 373.0]},
            },
            "outputs": {
                "stored_mass_kg":            {"unit": "kg"},
                "energy_stored_MJ":          {"unit": "MJ"},
                "fill_fraction":             {"unit": "dimensionless"},
                "compression_work_kJ_per_kg": {"unit": "kJ/kg"},
                "gravimetric_wt_pct":        {"unit": "wt%"},
                "volumetric_kg_per_m3":      {"unit": "kg/m3"},
                "compressibility_Z":         {"unit": "dimensionless"},
            },
            "source": "Lemmon et al. (2008) NIST; Zheng et al. (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for P in [100, 350, 500, 700]:
        r = model.predict({"pressure": P, "temperature": 298.15})
        print(f"P={P} bar: m={float(r['stored_mass_kg']):.3f} kg, "
              f"E={float(r['energy_stored_MJ']):.1f} MJ, "
              f"Z={float(r['compressibility_Z']):.4f}")
