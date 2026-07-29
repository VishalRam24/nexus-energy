"""
EC120 -- Fast Breeder Reactor (FBR), sodium-cooled -- F2a Point Kinetics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FBRPointKineticsF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for FBR F2a fast-spectrum point-kinetics model."""

    component_id = "EC120"
    component_name = "Fast Breeder Reactor (FBR), sodium-cooled"
    fidelity = "F2a -- Fast-Spectrum Point Kinetics + Lumped Thermal + Breeding"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FBRPointKineticsF2a(self._raw)

    # ------------------------------------------------------------------ #
    def predict(self, inputs: dict) -> dict:
        """
        Run a fast-spectrum point-kinetics transient.

        inputs:
            rho_ext     : float or callable(t) -- external reactivity [dk/k]
            dt          : float  output time step [s]   (default 0.1)
            duration_s  : float  simulation duration [s] (default 50.0)
            method      : str    'Radau' or 'BDF'        (default 'Radau')
        Returns time-series dict: t, n, C, T_f, T_Na, P_thermal_W, P_elec_W,
        rho, breeding_ratio, void_fraction, fissile_*_kg, net_fissile_bred_kg.
        """
        rho_ext = inputs.get("rho_ext", 0.0)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 50.0)
        method = inputs.get("method", "Radau")
        return self._model.simulate(rho_ext, dt, dur, method=method)

    def predict_step(self, inputs: dict) -> dict:
        """Step reactivity insertion. inputs: rho_step, dt, duration_s, t_insert, method."""
        return self._model.step_reactivity_insertion(
            inputs["rho_step"],
            dt=inputs.get("dt", 0.05),
            duration_s=inputs.get("duration_s", 50.0),
            t_insert=inputs.get("t_insert", 1.0),
            method=inputs.get("method", "Radau"),
        )

    def predict_ramp(self, inputs: dict) -> dict:
        """Ramp reactivity insertion. inputs: rho_rate, rho_max, dt, duration_s, t_start, method."""
        return self._model.ramp_reactivity_insertion(
            inputs["rho_rate"], inputs["rho_max"],
            dt=inputs.get("dt", 0.1),
            duration_s=inputs.get("duration_s", 100.0),
            t_start=inputs.get("t_start", 1.0),
            method=inputs.get("method", "Radau"),
        )

    # ------------------------------------------------------------------ #
    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "ec_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": "F2a",
            "fidelity_long": self.fidelity,
            "version": self.version,
            "inputs": {
                "rho_ext": {"unit": "dk/k", "range": [-0.005, 0.002]},
                "dt": {"unit": "s", "range": [0.001, 1.0]},
                "duration_s": {"unit": "s", "range": [0.001, 1000.0]},
                "method": {"unit": "-", "options": ["Radau", "BDF"]},
            },
            "outputs": {
                "t": "s",
                "n": "- (normalized neutron population / power)",
                "C": "- (6 delayed-neutron precursor groups)",
                "T_f": "K (lumped fuel temperature)",
                "T_Na": "K (lumped sodium-coolant temperature)",
                "P_thermal_W": "W",
                "P_elec_W": "W",
                "rho": "dk/k (total reactivity incl. feedback)",
                "breeding_ratio": "- (>1 for a breeder)",
                "net_fissile_bred_kg": "kg (cumulative net fissile bred)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict_step({"rho_step": 0.001, "dt": 0.05, "duration_s": 30.0})
    print(f"Lambda = {m._model.Lambda:.2e} s (fast spectrum), "
          f"beta_eff = {m._model.beta_total:.5f}")
    print(f"Net power coefficient = {m._model.power_coefficient():.3e} dk/k per K "
          f"({'stable' if m._model.power_coefficient() < 0 else 'UNSTABLE'})")
    print(f"After +100 pcm step: n_final={r['n'][-1]:.4f}, "
          f"T_f_final={r['T_f'][-1]:.1f} K, T_Na_final={r['T_Na'][-1]:.1f} K, "
          f"BR={r['breeding_ratio'][-1]:.3f}, "
          f"net bred={r['net_fissile_bred_kg'][-1]*1e3:.3f} g")
