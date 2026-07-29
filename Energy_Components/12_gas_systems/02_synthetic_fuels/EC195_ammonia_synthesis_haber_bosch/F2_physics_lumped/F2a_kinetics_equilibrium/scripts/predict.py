"""
EC195 -- Ammonia Synthesis (Haber-Bosch) -- F2a Kinetics + Equilibrium
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AmmoniaSynthesis_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Ammonia Synthesis F2a Temkin-Pyzhev CSTR model."""

    component_id = "EC195"
    component_name = "Ammonia Synthesis (Haber-Bosch)"
    fidelity = "F2a -- Temkin-Pyzhev Kinetics + CSTR with Recycle"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AmmoniaSynthesis_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run CSTR dynamic simulation (single pass or with recycle).

        inputs:
            T0_K : float          Initial temperature [K] (default 673.15)
            duration_s : float    Simulation duration [s] (default 600)
            dt : float            Output time step [s] (default 1.0)
            P_atm : float         Total pressure [atm] (default 200)
            GHSV : float          Gas hourly space velocity [1/h] (default 10000)
            T_cool_K : float      Coolant temperature [K] (default 673.15)
            with_recycle : bool   Simulate full recycle loop (default False)
        """
        with_recycle = inputs.get("with_recycle", False)

        if with_recycle:
            T0 = inputs.get("T0_K", None)
            P = inputs.get("P_atm", None)
            GHSV = inputs.get("GHSV", None)
            result = self._model.simulate_with_recycle(
                T0=T0, P=P, GHSV=GHSV,
            )
        else:
            T0 = inputs.get("T0_K", self._model.T_in)
            dur = inputs.get("duration_s", 600.0)
            dt = inputs.get("dt", 1.0)
            P = inputs.get("P_atm", None)
            GHSV = inputs.get("GHSV", None)
            T_cool = inputs.get("T_cool_K", None)

            result = self._model.simulate(
                T0=T0, duration_s=dur, dt=dt, P=P, GHSV=GHSV, T_cool=T_cool,
            )
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T0_K": {"unit": "K", "range": [573.15, 873.15]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
                "P_atm": {"unit": "atm", "range": [100, 350]},
                "GHSV": {"unit": "1/h", "range": [5000, 30000]},
                "T_cool_K": {"unit": "K", "range": [573.15, 773.15]},
                "with_recycle": {"type": "bool", "note": "Full recycle loop simulation"},
            },
            "outputs": {
                "t": "s",
                "T": "K",
                "X_N2": "- (N2 conversion)",
                "y_NH3": "- (NH3 mole fraction)",
                "C_N2": "mol/m3",
                "C_H2": "mol/m3",
                "C_NH3": "mol/m3",
                "X_eq_final": "- (equilibrium conversion at final T)",
                "overall_conversion": "- (with recycle only)",
                "energy_per_ton_NH3_GJ": "GJ/ton (with recycle only)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())

    # Single pass
    r = m.predict({"T0_K": 723.15, "duration_s": 300.0, "dt": 5.0})
    print(f"\nSingle pass:")
    print(f"  Final T: {r['T'][-1]:.1f} K")
    print(f"  N2 conversion: {r['X_N2'][-1]:.4f}")
    print(f"  NH3 mole fraction: {r['y_NH3'][-1]:.4f}")

    # With recycle
    r2 = m.predict({"with_recycle": True})
    print(f"\nWith recycle:")
    print(f"  Overall conversion: {r2['overall_conversion']:.4f}")
    print(f"  Energy per ton NH3: {r2['energy_per_ton_NH3_GJ']:.1f} GJ/ton")
