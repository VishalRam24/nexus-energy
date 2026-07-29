"""
EC202 -- Direct Air Capture (DAC), Liquid Solvent -- F2a Contactor + Calciner
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DAC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for DAC liquid-solvent F2a lumped contactor+calciner."""

    component_id = "EC202"
    component_name = "Direct Air Capture (DAC) — Liquid Solvent"
    fidelity = "F2a -- Lumped Contactor + Calciner ODE (KOH/Ca caustic loop)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DAC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a lumped dynamic DAC simulation.

        inputs:
            CO2_ppm_air   : float  ambient CO2 (default param, ~420 ppm)
            air_velocity_m_s : float air face velocity (default param)
            T_calciner_K  : float  initial calciner temperature (default setpoint)
            dt            : float  output sampling interval [s] (default 600)
            duration_s    : float  simulated duration [s] (default 86400 = 1 day)
            n_CaCO3_0_mol : float  initial CaCO3 inventory (default 0)

        Returns dict with time-series and steady-state energy/capture metrics.
        """
        ppm = inputs.get("CO2_ppm_air", None)
        u_air = inputs.get("air_velocity_m_s", None)
        T0 = inputs.get("T_calciner_K", None)
        dt = inputs.get("dt", 600.0)
        dur = inputs.get("duration_s", 86400.0)
        n0 = inputs.get("n_CaCO3_0_mol", 0.0)

        r = self._model.simulate(ppm=ppm, u_air=u_air, T_calc0=T0,
                                 dt=dt, duration_s=dur, n_CaCO3_0=n0)

        # convenience scalar summaries
        r["sec_thermal_final_GJ_tCO2"] = float(r["sec_thermal_GJ_tCO2"][-1])
        r["co2_captured_total_t"] = float(r["co2_captured_kg"][-1] / 1000.0)
        r["co2_product_total_t"] = float(r["co2_product_kg"][-1] / 1000.0)
        r["T_calciner_final_C"] = float(r["T_calciner_K"][-1] - 273.15)
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "CO2_ppm_air": {"unit": "ppm_v", "range": [350, 1000]},
                "air_velocity_m_s": {"unit": "m/s", "range": [0.5, 4.0]},
                "T_calciner_K": {"unit": "K", "range": [1023.15, 1223.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "n_CaCO3_0_mol": {"unit": "mol"},
            },
            "outputs": {
                "t": "s",
                "n_CO2_absorbed_mol": "mol",
                "n_CaCO3_mol": "mol",
                "T_calciner_K": "K",
                "R_absorption_mol_s": "mol/s",
                "R_calcination_mol_s": "mol/s",
                "co2_captured_kg": "kg",
                "co2_product_kg": "kg",
                "Q_thermal_W": "W",
                "sec_thermal_GJ_tCO2": "GJ/tCO2",
                "single_pass_capture": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"dt": 3600.0, "duration_s": 3600.0 * 24 * 30})
    print(f"Single-pass capture: {r['single_pass_capture']:.3f}")
    print(f"Calciner T (final): {r['T_calciner_final_C']:.1f} C")
    print(f"SEC thermal (final): {r['sec_thermal_final_GJ_tCO2']:.2f} GJ/tCO2")
    print(f"CO2 captured: {r['co2_captured_total_t']:.0f} t over 30 days")
