"""EC198 — Post-Combustion Capture (Amine Scrubbing) — F1a Energy Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import AmineCaptureF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AmineCaptureF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict MEA post-combustion capture system performance.

        Parameters
        ----------
        inputs : dict
            flue_gas_rate : float or array  (kg/s, 100–1000)
            co2_fraction  : float or array  (mol/mol, 0.04–0.15)
            capture_rate  : float or array  (0.80–0.95), default 0.90

        Returns
        -------
        dict
            co2_captured_kgs    : CO2 captured (kg/s)
            reboiler_duty_mw    : thermal power to reboiler (MW)
            electricity_mw      : electrical power demand (MW)
            specific_energy_gjt : total specific energy (GJ/tCO2)
        """
        flue = np.asarray(inputs["flue_gas_rate"], dtype=float)
        xCO2 = np.asarray(inputs["co2_fraction"],  dtype=float)
        cr   = np.asarray(inputs.get("capture_rate", 0.90), dtype=float)

        co2  = self._model.co2_captured(flue, xCO2, cr)
        q_r  = self._model.reboiler_power(flue, xCO2, cr)
        elec = self._model.electricity_power(flue, xCO2, cr)
        E_sp = self._model.specific_energy(cr)

        return {
            "co2_captured_kgs":    co2,
            "reboiler_duty_mw":    q_r,
            "electricity_mw":      elec,
            "specific_energy_gjt": E_sp,
        }

    def get_info(self) -> dict:
        return {
            "name": "Post-Combustion Capture (Amine Scrubbing)",
            "ec_id": "EC198",
            "fidelity": "F1a",
            "description": (
                "MEA scrubbing energy model. "
                "q_reboiler = q_base / (1 - exp(-k_LG*(LG-LG_min))). "
                "capture_rate = 0.90 (design), electricity = 0.25 GJ/tCO2."
            ),
            "inputs": {
                "flue_gas_rate": {"unit": "kg/s",    "range": [100.0, 1000.0]},
                "co2_fraction":  {"unit": "mol/mol", "range": [0.04, 0.15]},
                "capture_rate":  {"unit": "-",       "range": [0.80, 0.95], "default": 0.90},
            },
            "outputs": {
                "co2_captured_kgs":    {"unit": "kg/s"},
                "reboiler_duty_mw":    {"unit": "MW"},
                "electricity_mw":      {"unit": "MW"},
                "specific_energy_gjt": {"unit": "GJ/tCO2"},
            },
            "source": "Abu-Zahra et al. (2007), Int. J. GHG Control, 1(1), 37-46",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": 0.90})
    print(f"Flue gas=500 kg/s, CO2=12%, capture=90%:")
    print(f"  CO2 captured       = {float(r['co2_captured_kgs']):.2f} kg/s")
    print(f"  Reboiler duty      = {float(r['reboiler_duty_mw']):.1f} MW")
    print(f"  Electricity        = {float(r['electricity_mw']):.1f} MW")
    print(f"  Specific energy    = {float(r['specific_energy_gjt']):.2f} GJ/tCO2")
    info = model.get_info()
    print(f"  EC ID: {info['ec_id']}, Fidelity: {info['fidelity']}")
