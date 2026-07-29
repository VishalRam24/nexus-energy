"""EC140 -- Anaerobic Digester -- F2a Monod Kinetics -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AnaerobicDigesterF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AnaerobicDigesterF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of anaerobic digester.

        inputs:
            S_in       : float or callable(t) [gCOD/L] (default: 40)
            T          : float or callable(t) [K] (default: 308.15)
            pH         : float or callable(t) (default: None = optimal)
            dt         : float [days] (default: 0.1)
            duration_d : float [days] (default: 60)
            HRT        : float [days] (default: 20)
            x0         : [S0, X0] initial state

        returns:
            dict with time-series
        """
        S_in = inputs.get("S_in", None)
        T = inputs.get("T", None)
        pH = inputs.get("pH", None)
        dt = inputs.get("dt", 0.1)
        duration_d = inputs.get("duration_d", 60.0)
        HRT = inputs.get("HRT", None)
        x0 = inputs.get("x0", None)

        return self._model.simulate(S_in, T, pH, dt, duration_d, x0, HRT)

    def predict_steady_state(self, inputs: dict = None) -> dict:
        """Return analytical steady-state performance."""
        if inputs is None:
            inputs = {}
        S_in = inputs.get("S_in", self._model.S_in)
        HRT = inputs.get("HRT", self._model.HRT)
        T = inputs.get("T", self._model.T_op)
        pH = inputs.get("pH", None)
        return self._model.steady_state(S_in, HRT, T, pH)

    def get_info(self) -> dict:
        return {
            "name": "Anaerobic Digester (Mesophilic)",
            "ec_id": "EC140",
            "fidelity": "F2a",
            "sub_fidelity": "kinetics",
            "description": (
                "Simplified ADM1 with Monod kinetics for substrate and biomass. "
                "dS/dt = (S_in - S)/HRT - mu*X/Y_xs; "
                "dX/dt = mu*X - k_d*X - X/HRT. "
                "Includes Arrhenius temperature correction and pH inhibition."
            ),
            "inputs": {
                "S_in": {"unit": "gCOD/L", "range": [0, 100], "default": 40},
                "HRT": {"unit": "d", "range": [5, 60], "default": 20},
                "T": {"unit": "K", "range": [293, 328], "default": 308.15},
                "pH": {"unit": "dimensionless", "range": [5.5, 9.0], "default": 7.0},
                "dt": {"unit": "d", "default": 0.1},
                "duration_d": {"unit": "d", "default": 60},
            },
            "outputs": {
                "t": {"unit": "d"},
                "S": {"unit": "gCOD/L", "note": "substrate concentration"},
                "X": {"unit": "gVSS/L", "note": "biomass concentration"},
                "V_ch4_rate_L_d": {"unit": "L/d", "note": "methane production rate"},
                "COD_removal_pct": {"unit": "%"},
                "mu_eff": {"unit": "1/d", "note": "effective growth rate"},
            },
            "source": "Batstone et al. (2002), IWA ADM1; Rittmann & McCarty (2001)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== Steady State (default params) ===")
    ss = model.predict_steady_state()
    for k, v in ss.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Startup Simulation (60 days) ===")
    r = model.predict({"dt": 0.5, "duration_d": 60.0})
    print(f"  S final: {r['S'][-1]:.2f} gCOD/L")
    print(f"  X final: {r['X'][-1]:.2f} gVSS/L")
    print(f"  CH4 rate final: {r['V_ch4_rate_L_d'][-1]:.0f} L/d")
    print(f"  COD removal: {r['COD_removal_pct'][-1]:.1f}%")
