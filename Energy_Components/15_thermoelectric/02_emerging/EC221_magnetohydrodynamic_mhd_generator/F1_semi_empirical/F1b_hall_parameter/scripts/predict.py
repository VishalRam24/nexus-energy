"""EC221 — MHD Generator — F1b Hall Parameter — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import MHDF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MHDF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict MHD generator performance with Hall parameter and T-dependent sigma.

        Parameters
        ----------
        inputs : dict
            sigma     : float or array — plasma conductivity at T_ref [S/m]
            u         : float or array — plasma velocity [m/s]
            B         : float or array — magnetic flux density [T]
            K         : float or array — load factor [-] (0.5 = max power)
            beta      : float or array — Hall parameter omega_e*tau_e (default from params)
            T_plasma_K: float or array — plasma static temperature [K] (default from params)

        Returns
        -------
        dict with EMF_V, J_Am2, J_hall_Am2, sigma_actual_Sm, sigma_eff_Sm,
                  power_density_Wm3, power_elec_W, heat_input_stag_W,
                  eta_mhd, eta_hall, eta_electric, K_optimal
        """
        u = self.params["unit"]
        sigma = inputs.get("sigma", u["sigma_plasma_0"]["value"])
        vel = inputs.get("u", u["u_plasma"]["value"])
        B = inputs.get("B", u["B_field"]["value"])
        K = inputs.get("K", u["K_load"]["value"])
        beta = inputs.get("beta", u["beta_hall"]["value"])
        T_K = inputs.get("T_plasma_K", None)

        return self._model.compute(sigma, vel, B, K, beta, T_K)

    def get_info(self) -> dict:
        return {
            "name": "Magnetohydrodynamic (MHD) Generator",
            "ec_id": "EC221",
            "fidelity": "F1b",
            "description": (
                "Hall parameter correction (sigma_eff = sigma/(1+beta^2)), "
                "temperature-dependent conductivity (sigma ~ T^1.5), and "
                "stagnation-enthalpy heat input Q_in = rho*u*(cp*T+0.5*u^2)*A "
                "(first-law correct — not kinetic-energy-only Q_in)."
            ),
            "inputs": {
                "sigma": {"unit": "S/m", "range": [1, 100], "default": 10.0},
                "u": {"unit": "m/s", "range": [100, 2000], "default": 800.0},
                "B": {"unit": "T", "range": [0.5, 10], "default": 5.0},
                "K": {"unit": "-", "range": [0, 1], "default": 0.5},
                "beta": {"unit": "-", "range": [0, 10], "default": 3.0},
                "T_plasma_K": {"unit": "K", "range": [1500, 4000], "default": 2500.0},
            },
            "outputs": {
                "EMF_V": {"unit": "V"},
                "J_Am2": {"unit": "A/m2"},
                "J_hall_Am2": {"unit": "A/m2"},
                "sigma_eff_Sm": {"unit": "S/m"},
                "power_elec_W": {"unit": "W"},
                "heat_input_stag_W": {"unit": "W"},
                "eta_mhd": {"unit": "-"},
                "eta_hall": {"unit": "-"},
                "eta_electric": {"unit": "-"},
            },
            "source": "Rosa (1987); Messerle (1995); Veefkind (1977)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({
        "sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5,
        "beta": 3.0, "T_plasma_K": 2500.0
    })
    print("MHD F1b design point (sigma=10 S/m, u=800 m/s, B=5T, K=0.5, beta=3):")
    for k, v in r.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k} = {val:.4g}")
