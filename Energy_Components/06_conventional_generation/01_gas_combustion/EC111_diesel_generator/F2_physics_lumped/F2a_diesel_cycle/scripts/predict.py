"""EC111 -- Diesel Generator -- F2a Diesel Cycle -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DieselGeneratorF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DieselGeneratorF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of diesel generator.

        inputs:
            P_load       : float or callable(t) [W] -- electrical load demand
            dt           : float [s]  (default 0.01)
            duration_s   : float [s]  (default 30.0)
            omega_ref_rpm: float [rpm] (default: nominal)

        returns:
            dict with time-series arrays
        """
        P_load = inputs["P_load"]
        dt = inputs.get("dt", 0.01)
        duration_s = inputs.get("duration_s", 30.0)
        omega_ref = inputs.get("omega_ref_rpm", None)
        x0 = inputs.get("x0", None)

        return self._model.simulate(P_load, dt, duration_s, x0=x0, omega_ref_rpm=omega_ref)

    def predict_steady_state(self, inputs: dict) -> dict:
        """Return steady-state performance at given load fraction."""
        load_frac = inputs.get("load_fraction", inputs.get("P_load", 250000.0) / self._model.P_rated)
        return self._model.steady_state(load_frac)

    def predict_cycle(self, inputs: dict = None) -> dict:
        """Return diesel cycle state points and efficiency."""
        r_c = inputs.get("r_c", self._model.r_c) if inputs else self._model.r_c
        r_co = inputs.get("r_co", self._model._design_cutoff_ratio()) if inputs else self._model._design_cutoff_ratio()
        sp = self._model.cycle_state_points(r_c, r_co)
        sp["eta_thermal"] = self._model.diesel_efficiency(r_c, r_co)
        sp["w_net_per_kg"] = self._model.net_work(r_c, r_co)
        sp["q_add_per_kg"] = self._model.heat_added(r_c, r_co)
        sp["q_rej_per_kg"] = self._model.heat_rejected(r_c, r_co)
        return sp

    def get_info(self) -> dict:
        return {
            "name": "Diesel Generator",
            "ec_id": "EC111",
            "fidelity": "F2a",
            "sub_fidelity": "diesel_cycle",
            "description": (
                "Air-standard diesel cycle thermodynamic model with ODE-based "
                "dynamic governor, generator efficiency curve, and BSFC model. "
                "Processes: 1-2 isentropic compression, 2-3 constant-P heat addition, "
                "3-4 isentropic expansion, 4-1 constant-V heat rejection."
            ),
            "inputs": {
                "P_load": {"unit": "W", "range": [0.0, 550000.0]},
                "load_fraction": {"unit": "dimensionless", "range": [0.0, 1.1]},
                "dt": {"unit": "s", "default": 0.01},
                "duration_s": {"unit": "s", "default": 30.0},
            },
            "outputs": {
                "t": {"unit": "s"},
                "omega_rpm": {"unit": "rpm"},
                "frequency_Hz": {"unit": "Hz"},
                "P_elec_W": {"unit": "W"},
                "P_engine_W": {"unit": "W"},
                "fuel_rate_kg_s": {"unit": "kg/s"},
                "eta_overall": {"unit": "dimensionless"},
                "eta_thermal": {"unit": "dimensionless"},
                "BSFC_g_per_kWh": {"unit": "g/kWh"},
            },
            "source": "Heywood (2018), Internal Combustion Engine Fundamentals, 2nd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== Steady State at 75% Load ===")
    ss = model.predict_steady_state({"load_fraction": 0.75})
    for k, v in ss.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Diesel Cycle State Points ===")
    cyc = model.predict_cycle()
    for k, v in cyc.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Dynamic Simulation (load step 50% -> 100%) ===")
    def load_step(t):
        return 250000.0 if t < 10.0 else 500000.0
    r = model.predict({"P_load": load_step, "dt": 0.05, "duration_s": 30.0})
    print(f"  Final rpm: {r['omega_rpm'][-1]:.1f}")
    print(f"  Final freq: {r['frequency_Hz'][-1]:.2f} Hz")
