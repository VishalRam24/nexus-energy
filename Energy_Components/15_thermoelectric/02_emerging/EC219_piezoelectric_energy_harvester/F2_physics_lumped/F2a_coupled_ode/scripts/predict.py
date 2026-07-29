"""
EC219 -- Piezoelectric Energy Harvester -- F2a Coupled Electromechanical ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PiezoF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC219 F2a coupled-ODE harvester model."""

    component_id = "EC219"
    component_name = "Piezoelectric Energy Harvester"
    fidelity = "F2a -- Coupled Electromechanical ODE (time-domain, lumped 1-DOF)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PiezoF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a time-domain coupled-ODE simulation under harmonic base
        excitation and return the steady-state harvested power.

        inputs:
            acceleration_ms2 : float -- base acceleration amplitude [m/s^2]
            frequency_hz     : float -- excitation frequency [Hz]
            R_load_ohm       : float -- load resistance [ohm]; default = R_opt
            duration_s       : float -- integration time [s]; default auto
        """
        A = inputs.get("acceleration_ms2", self._raw["unit"]["base_acceleration_default"]["value"])
        f = inputs.get("frequency_hz", self._model.f_n)
        R_L = inputs.get("R_load_ohm", self._model.optimal_load(f))
        dur = inputs.get("duration_s", None)

        result = self._model.simulate(A, f, R_L, duration_s=dur)
        # Attach scalar conveniences
        result["optimal_R_ohm"] = self._model.optimal_load(f)
        result["power_uw"] = result["P_avg"] * 1e6
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "acceleration_ms2": {"unit": "m/s^2", "range": [0.01, 100.0]},
                "frequency_hz": {"unit": "Hz", "range": [10.0, 1000.0]},
                "R_load_ohm": {"unit": "ohm", "range": [100.0, 1e7]},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "x": "m (relative tip displacement)",
                "voltage": "V (load voltage time series)",
                "power_inst": "W (instantaneous load power)",
                "P_avg": "W (steady-state average harvested power)",
                "V_rms": "V",
                "optimal_R_ohm": "ohm",
                "frequency_ratio": "-",
            },
            "derived_parameters": {
                "m_eff_kg": self._model.m_eff,
                "k_eff_N_per_m": self._model.k_eff,
                "c_mech_Ns_per_m": self._model.c_mech,
                "theta_N_per_V": self._model.theta,
                "C_p_F": self._model.C_p,
                "f_n_Hz": self._model.f_n,
                "R_opt_ohm": self._model.optimal_load(),
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(info)
    r = m.predict({"acceleration_ms2": 9.81, "frequency_hz": 100.0})
    print(f"At resonance (1 g, R_opt={r['optimal_R_ohm']:.1f} ohm): "
          f"P_avg={r['power_uw']:.2f} uW, V_rms={r['V_rms']:.3f} V")
