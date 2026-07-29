"""EC116 -- PWR Nuclear Reactor -- F2a Point Kinetics -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PWRPointKineticsF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PWRPointKineticsF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run point kinetics transient simulation.

        inputs:
            rho_ext     : float or callable(t) [dk/k] -- external reactivity
            dt          : float [s] (default 0.01)
            duration_s  : float [s] (default 100.0)
            method      : str ('Radau' or 'BDF', default 'Radau')
            rtol        : float (default 1e-8)

        returns:
            dict with t, n, C, T_f, T_m, P_thermal_W, P_elec_W, rho
        """
        rho_ext = inputs.get("rho_ext", 0.0)
        dt = inputs.get("dt", 0.01)
        duration_s = inputs.get("duration_s", 100.0)
        method = inputs.get("method", "Radau")
        rtol = inputs.get("rtol", 1e-8)
        x0 = inputs.get("x0", None)

        return self._model.simulate(rho_ext, dt, duration_s, x0=x0,
                                     method=method, rtol=rtol)

    def predict_step(self, inputs: dict) -> dict:
        """Simulate step reactivity insertion."""
        rho_step = inputs.get("rho_step", 0.001)
        dt = inputs.get("dt", 0.01)
        duration_s = inputs.get("duration_s", 100.0)
        t_insert = inputs.get("t_insert", 1.0)
        return self._model.step_reactivity_insertion(rho_step, dt, duration_s, t_insert)

    def predict_ramp(self, inputs: dict) -> dict:
        """Simulate ramp reactivity insertion."""
        rho_rate = inputs.get("rho_rate", 1e-4)
        rho_max = inputs.get("rho_max", 0.003)
        dt = inputs.get("dt", 0.01)
        duration_s = inputs.get("duration_s", 100.0)
        t_start = inputs.get("t_start", 1.0)
        return self._model.ramp_reactivity_insertion(rho_rate, rho_max, dt, duration_s, t_start)

    def get_info(self) -> dict:
        return {
            "name": "Pressurized Water Reactor (PWR)",
            "ec_id": "EC116",
            "fidelity": "F2a",
            "sub_fidelity": "point_kinetics",
            "description": (
                "Six-group delayed neutron point kinetics with lumped fuel and "
                "moderator thermal feedback. Stiff ODE system solved with Radau "
                "(implicit Runge-Kutta) or BDF method. Includes Doppler and "
                "moderator temperature reactivity coefficients."
            ),
            "inputs": {
                "rho_ext": {"unit": "dk/k", "range": [-0.01, 0.005]},
                "dt": {"unit": "s", "default": 0.01},
                "duration_s": {"unit": "s", "default": 100.0},
                "method": {"type": "str", "default": "Radau", "options": ["Radau", "BDF"]},
            },
            "outputs": {
                "t": {"unit": "s"},
                "n": {"unit": "dimensionless", "note": "normalized neutron population"},
                "C": {"unit": "dimensionless", "note": "6 precursor group concentrations"},
                "T_f": {"unit": "K", "note": "fuel temperature"},
                "T_m": {"unit": "K", "note": "moderator temperature"},
                "P_thermal_W": {"unit": "W"},
                "P_elec_W": {"unit": "W"},
                "rho": {"unit": "dk/k", "note": "total reactivity"},
            },
            "solver": "scipy.integrate.solve_ivp with method='Radau', rtol=1e-8",
            "source": "Duderstadt & Hamilton (1976); Stacey (2007); Keepin (1965)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== Step Reactivity Insertion (+100 pcm) ===")
    r = model.predict_step({"rho_step": 0.001, "dt": 0.1, "duration_s": 50.0})
    print(f"  t range: {r['t'][0]:.2f} -- {r['t'][-1]:.2f} s")
    print(f"  n final: {r['n'][-1]:.4f} (normalized)")
    print(f"  T_f final: {r['T_f'][-1]:.1f} K")
    print(f"  T_m final: {r['T_m'][-1]:.1f} K")
    print(f"  P_thermal final: {r['P_thermal_W'][-1] / 1e6:.1f} MW")
