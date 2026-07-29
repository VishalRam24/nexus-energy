"""EC140 — Anaerobic Digester (Mesophilic) — F1a Yield Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AnaerobicDigesterF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AnaerobicDigesterF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            vs_loading     : float or array, volatile solids loading [kgVS/(m³·day), 1-8]
            hrt            : float or array, hydraulic retention time [days, 5-40]
            temperature    : float or array, reactor temperature [degC, 25-55] (optional, default 37)
        returns:
            methane_yield_m3kgvs : specific methane yield [m³_CH4/kgVS]
            biogas_rate_m3day    : total biogas production [m³_biogas/day]
            methane_rate_m3day   : methane production [m³_CH4/day]
            energy_output_kwh_day: equivalent thermal energy [kWh/day]
        """
        vs = np.asarray(inputs["vs_loading"], dtype=float)
        hrt = np.asarray(inputs["hrt"], dtype=float)
        T = np.asarray(inputs.get("temperature", 37.0), dtype=float)
        return {
            "methane_yield_m3kgvs": self._model.methane_yield(hrt, T),
            "biogas_rate_m3day": self._model.biogas_rate(vs, hrt, T),
            "methane_rate_m3day": self._model.methane_rate(vs, hrt, T),
            "energy_output_kwh_day": self._model.energy_output(vs, hrt, T),
        }

    def get_info(self) -> dict:
        return {
            "name": "Anaerobic Digester (Mesophilic)",
            "ec_id": "EC140",
            "fidelity": "F1a",
            "model": "Biogas Yield Model",
            "description": (
                "methane_yield = Y_max*(1-exp(-k*HRT))*f_T(temp); "
                "biogas_rate = methane_rate / methane_fraction"
            ),
            "inputs": {
                "vs_loading": {"unit": "kgVS/(m³·day)", "range": [1.0, 8.0]},
                "hrt": {"unit": "days", "range": [5.0, 40.0]},
                "temperature": {"unit": "degC", "range": [25.0, 55.0], "default": 37.0},
            },
            "outputs": {
                "methane_yield_m3kgvs": {"unit": "m³_CH4/kgVS"},
                "biogas_rate_m3day": {"unit": "m³/day"},
                "methane_rate_m3day": {"unit": "m³_CH4/day"},
                "energy_output_kwh_day": {"unit": "kWh/day"},
            },
            "source": "Buswell & Mueller (1952); Batstone et al. (2002) ADM1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("At design conditions (VS=3 kgVS/m³/day, HRT=20 days, T=37 degC):")
    r = model.predict({"vs_loading": 3.0, "hrt": 20.0, "temperature": 37.0})
    print(f"  Methane yield  : {float(r['methane_yield_m3kgvs']):.4f} m³_CH4/kgVS")
    print(f"  Methane rate   : {float(r['methane_rate_m3day']):.1f} m³_CH4/day")
    print(f"  Biogas rate    : {float(r['biogas_rate_m3day']):.1f} m³/day")
    print(f"  Energy output  : {float(r['energy_output_kwh_day']):.0f} kWh/day")
