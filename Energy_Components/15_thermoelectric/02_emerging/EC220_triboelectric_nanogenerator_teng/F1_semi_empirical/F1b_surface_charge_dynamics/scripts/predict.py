"""EC220 — TENG — F1b Surface Charge Dynamics — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import TENGF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TENGF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict TENG output with surface charge dynamics.

        Parameters
        ----------
        inputs : dict
            frequency_hz : float or array (Hz, 0.1-100)
            R_load_ohm   : float or array (ohm, 1e4-1e10)
            t_s          : float (s, 0-1e5) — elapsed time for charge decay

        Returns
        -------
        dict with sigma_Cm2, V_oc_peak_V, C_avg_F, R_internal_ohm,
                  power_avg_w, power_net_w, power_density_mwcm2, efficiency, dielectric_loss_w
        """
        f = inputs.get("frequency_hz", 3.0)
        R = inputs.get("R_load_ohm", 1e7)
        t = inputs.get("t_s", 0.0)
        return self._model.compute(f, R, t)

    def get_info(self) -> dict:
        return {
            "name": "Triboelectric Nanogenerator (TENG)",
            "ec_id": "EC220",
            "fidelity": "F1b",
            "description": (
                "TENG contact-separation mode with: "
                "(1) surface charge decay sigma(t) = sigma0*exp(-t/tau); "
                "(2) two-layer dielectric stack (d_eff = d1/eps_r1 + d2/eps_r2); "
                "(3) dielectric loss correction (tan_delta); "
                "(4) frequency-dependent RC output from capacitive internal impedance."
            ),
            "inputs": {
                "frequency_hz": {"unit": "Hz", "range": [0.1, 100], "default": 3.0},
                "R_load_ohm": {"unit": "ohm", "range": [1e4, 1e10], "default": 1e7},
                "t_s": {"unit": "s", "range": [0, 1e5], "default": 0.0, "note": "Time for charge decay"},
            },
            "outputs": {
                "sigma_Cm2": {"unit": "C/m^2"},
                "V_oc_peak_V": {"unit": "V"},
                "C_avg_F": {"unit": "F"},
                "R_internal_ohm": {"unit": "ohm"},
                "power_avg_w": {"unit": "W"},
                "power_net_w": {"unit": "W", "note": "After dielectric loss"},
                "power_density_mwcm2": {"unit": "mW/cm^2"},
                "efficiency": {"unit": "-"},
                "dielectric_loss_w": {"unit": "W"},
            },
            "source": "Niu & Wang (2015); Niu et al. (2013); Zi et al. (2015)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})
    print("Design point (f=3 Hz, R=10 MOhm, t=0s):")
    for k, v in r.items():
        val = v if isinstance(v, (int, float)) else float(np.atleast_1d(v)[0])
        print(f"  {k} = {val:.4e}")
