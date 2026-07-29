"""
EC223 — Radioisotope Thermoelectric Generator (RTG) — F1a Decay + TEG Model

Two coupled mechanisms:

1. Radioactive decay heat (Pu-238 alpha decay):
    P_thermal(t) = P0 * exp(-ln(2) * t / t_half)   [W]
    Pu-238 t_half = 87.7 years — very long for space missions

2. TEG conversion (SiGe thermoelectrics):
    eta_TEG(t) = eta_0 * (1 - gamma * t)   [degradation over time]
    P_electric(t) = P_thermal(t) * eta_TEG(t)   [W]

Hot-side temperature drops as thermal power decays:
    T_hot(t) ≈ T_hot_0 * (P_thermal(t) / P_thermal_0)^0.25  [K]
    (radiative cooling: P ~ T^4, so T ~ P^0.25)

Efficiency from ZT (approximate, Carnot-scaled):
    eta_Carnot(t) = 1 - T_cold / T_hot(t)
    ZT_SiGe ~ 0.5 (at operating T)
    eta_TEG_ideal(t) = eta_Carnot(t) * (sqrt(1+ZT)-1)/(sqrt(1+ZT)+T_cold/T_hot)

This model uses the simpler parametric degradation model for F1a.

References:
    Bennett, G.L. (2006). Space nuclear power: Opening the final frontier. AIAA 2006-4191.
    El-Genk, M.S. & Saber, H.H. (2005). Energy Convers. Mgmt. 46(7-8), 1083.
    NASA GPHS-RTG: https://rps.nasa.gov/power-and-thermal-systems/power-systems/
"""

import numpy as np

ln2 = np.log(2.0)


class RTGF1a:
    """RTG — radioisotope decay heat + TEG conversion model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P0 = u["P_thermal_0_W"]["value"]          # W thermal
        self.t_half = u["t_half_years"]["value"]        # years
        self.eta0 = u["eta_teg_0"]["value"]            # -
        self.gamma = u["eta_teg_degradation_rate"]["value"]  # 1/year
        self.T_hot_0 = u["T_hot_0_K"]["value"]         # K
        self.T_cold = u["T_cold_K"]["value"]            # K
        self.ZT = 0.5   # SiGe thermoelectrics at high temperature

    def compute(self, t_years):
        """
        Parameters
        ----------
        t_years : float or array — time since launch/deployment [years]

        Returns
        -------
        dict: P_thermal_W, eta_teg, P_electric_W, T_hot_K, eta_carnot,
              fraction_thermal_remaining, power_fraction
        """
        t = np.asarray(t_years, dtype=float)
        t = np.maximum(t, 0.0)

        # Thermal power from decay
        P_thermal = self.P0 * np.exp(-ln2 * t / self.t_half)

        # Hot-side temperature (radiative cooling: P ~ T^4)
        T_hot = self.T_hot_0 * (P_thermal / self.P0)**0.25
        T_hot = np.maximum(T_hot, self.T_cold + 1.0)

        # Carnot efficiency at current temperatures
        eta_carnot = 1.0 - self.T_cold / T_hot

        # TEG efficiency from ZT (Angist formula)
        sqrt_ZT = np.sqrt(1.0 + self.ZT)
        eta_teg_thermo = eta_carnot * (sqrt_ZT - 1.0) / (sqrt_ZT + self.T_cold / T_hot)

        # Apply parametric degradation (TEG contacts/interfaces degrade)
        deg_factor = np.maximum(1.0 - self.gamma * t, 0.5)  # floor at 50% of initial
        eta_teg = eta_teg_thermo * deg_factor

        # Electric power output
        P_electric = P_thermal * eta_teg

        fraction_thermal = P_thermal / self.P0
        power_fraction = P_electric / (self.P0 * self.eta0 + 1e-12)

        return {
            "P_thermal_W": P_thermal,
            "eta_teg": eta_teg,
            "P_electric_W": P_electric,
            "T_hot_K": T_hot,
            "eta_carnot": eta_carnot,
            "fraction_thermal_remaining": fraction_thermal,
            "power_fraction": power_fraction,
        }
