"""
EC193 — Methanation Reactor (Power-to-Gas) — F1a Sabatier Equilibrium Model

Sabatier reaction: CO2 + 4H2 -> CH4 + 2H2O  (exothermic, DH = -165 kJ/mol)
Equilibrium conversion: Gaussian fit calibrated against Gao et al. (2012) data.

X = X_max * exp(-k*(T-T_opt)^2 / T_opt^2) * (P/P_ref)^P_exp

Reference:
    Gao, J., Wang, Y., Ping, Y., Hu, D., Xu, G., Gu, F., Su, F. (2012).
    A thermodynamic analysis of methanation reactions of carbon oxides for the
    production of synthetic natural gas. RSC Advances, 2, 2358-2368.
"""

import numpy as np


class MethanationF1a:
    """
    Catalytic methanation reactor — simplified equilibrium conversion model.
    Models CO2 conversion to CH4 as a function of temperature and pressure.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_opt   = u["T_opt"]["value"]           # degC
        self.X_max   = u["X_max"]["value"]            # dimensionless
        self.k_T     = u["k_T"]["value"]              # Gaussian width
        self.P_ref   = u["P_ref"]["value"]            # bar
        self.P_exp   = u["P_exp"]["value"]            # pressure exponent
        self.n_CO2   = u["n_CO2_in"]["value"]         # mol/s
        self.LHV_CH4 = u["LHV_CH4"]["value"]          # kJ/mol
        self.LHV_H2  = u["LHV_H2"]["value"]           # kJ/mol
        self.DH      = abs(u["DH_reaction"]["value"]) # kJ/mol (magnitude, exothermic)

    def conversion(self, temperature_c, pressure_bar, h2_co2_ratio=4.0):
        """
        CO2-to-CH4 equilibrium conversion fraction.

        X = X_max * exp(-k_T * ((T - T_opt)/T_opt)^2) * (P/P_ref)^P_exp

        Parameters
        ----------
        temperature_c  : float or array  (degC)
        pressure_bar   : float or array  (bar)
        h2_co2_ratio   : float or array  (mol/mol)  — modifies X slightly for non-stoichiometric feeds

        Returns
        -------
        X : float or array  — conversion [0, 1]
        """
        T = np.asarray(temperature_c, dtype=float)
        P = np.asarray(pressure_bar, dtype=float)
        r = np.asarray(h2_co2_ratio, dtype=float)

        # Gaussian temperature response (exothermic: optimal at T_opt, drops at high T)
        T_norm = (T - self.T_opt) / self.T_opt
        X_T    = self.X_max * np.exp(-self.k_T * T_norm ** 2)

        # Pressure effect (Le Chatelier: more moles on reactant side → P increases conversion)
        X_P    = (P / self.P_ref) ** self.P_exp

        X      = X_T * X_P

        # H2/CO2 ratio effect: sub-stoichiometric H2 limits conversion
        # X is limited by H2 availability: X <= r/4
        X_stoich = np.clip(r / 4.0, 0.0, 1.0)
        X        = np.minimum(X, X_stoich)

        return np.clip(X, 0.0, 1.0)

    def ch4_rate(self, temperature_c, pressure_bar, h2_co2_ratio=4.0, n_co2_in=None):
        """CH4 production rate (mol/s) for given CO2 feed rate."""
        if n_co2_in is None:
            n_co2_in = self.n_CO2
        X = self.conversion(temperature_c, pressure_bar, h2_co2_ratio)
        return X * np.asarray(n_co2_in, dtype=float)

    def h2_consumed(self, temperature_c, pressure_bar, h2_co2_ratio=4.0, n_co2_in=None):
        """H2 consumption rate (mol/s) per stoichiometry: 4 mol H2 per mol CO2 converted."""
        if n_co2_in is None:
            n_co2_in = self.n_CO2
        X = self.conversion(temperature_c, pressure_bar, h2_co2_ratio)
        return 4.0 * X * np.asarray(n_co2_in, dtype=float)

    def efficiency(self, temperature_c, pressure_bar, h2_co2_ratio=4.0):
        """
        Methanation energy efficiency.
        eta = X * LHV_CH4 / (4 * LHV_H2)
        Represents the fraction of hydrogen chemical energy captured as methane.
        """
        X   = self.conversion(temperature_c, pressure_bar, h2_co2_ratio)
        return X * self.LHV_CH4 / (4.0 * self.LHV_H2)

    def heat_released(self, temperature_c, pressure_bar, h2_co2_ratio=4.0, n_co2_in=None):
        """
        Exothermic heat released by the Sabatier reaction (kW).
        Q = X * n_CO2 * DH_reaction
        """
        if n_co2_in is None:
            n_co2_in = self.n_CO2
        X = self.conversion(temperature_c, pressure_bar, h2_co2_ratio)
        return X * np.asarray(n_co2_in, dtype=float) * self.DH  # kJ/s = kW
