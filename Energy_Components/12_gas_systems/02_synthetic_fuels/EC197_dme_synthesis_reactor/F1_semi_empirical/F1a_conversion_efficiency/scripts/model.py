"""
EC197 — DME Synthesis Reactor — F1a Equilibrium Conversion Model

Dimethyl Ether (DME) synthesis via:
  Direct route (one-step, syngas): CO + 2H2 → CH3OH  then  2CH3OH → CH3OCH3 + H2O
  Combined reaction: 2CO + 4H2 → CH3OCH3 + H2O   (ΔH = -204.8 kJ/mol DME)
  Also: 3CO2 + 3H2 → CH3OCH3 + CO2 + H2O

F1a models overall CO/CO2 conversion to DME using equilibrium-fit approach.
Selectivity between DME and MeOH by-product included.

Equilibrium conversion fit (Ereña et al. 2005 / García-Trenco et al. 2018):
    X_CO = X_max * exp(-k_T * ((T-T_opt)/T_opt)^2) * (P/P_ref)^P_exp
    Selectivity to DME: S_DME = S_max * exp(-k_S * ((T-T_S_opt)/T_S_opt)^2)

References:
    Ereña, J. et al. (2005). Effect of operating conditions on the synthesis of
        dimethyl ether over a CuO-ZnO-Al2O3/NaHZSM-5 bifunctional catalyst.
        Catal. Today, 107-108, 467-473.
    García-Trenco, A. et al. (2018). CO2 to DME. ACS Catal., 8, 4660-4671.
    Naik, S.P. et al. (2011). DME synthesis. Fuel, 90, 3266-3273.
    Lee, S. (2007). Methanol synthesis technology. CRC Press.
"""

import numpy as np


class DMEReactorF1a:
    """
    DME synthesis reactor — F1a equilibrium conversion + selectivity model.
    Single-step synthesis over bifunctional catalyst (methanol synthesis + dehydration).
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_opt     = u["T_opt"]["value"]           # degC
        self.X_max     = u["X_max"]["value"]
        self.k_T       = u["k_T"]["value"]
        self.P_ref     = u["P_ref"]["value"]           # bar
        self.P_exp     = u["P_exp"]["value"]
        self.S_DME_max = u["S_DME_max"]["value"]       # max DME selectivity
        self.k_S       = u["k_S"]["value"]             # selectivity Gaussian width
        self.T_S_opt   = u["T_S_opt"]["value"]         # degC (optimal T for DME selectivity)
        self.n_CO_in   = u["n_CO_in"]["value"]         # mol/s
        self.h2_co_ratio = u["H2_CO_ratio"]["value"]
        self.LHV_DME   = u["LHV_DME"]["value"]         # kJ/mol
        self.LHV_H2    = u["LHV_H2"]["value"]
        self.DH        = abs(u["DH_reaction"]["value"])  # kJ/mol CO

    def conversion(self, temperature_C, pressure_bar, h2_co_ratio=None):
        """
        CO conversion to DME (plus MeOH by-product).
        X = X_max * exp(-k_T*((T-T_opt)/T_opt)^2) * (P/P_ref)^P_exp
        Stoichiometric limit: CO + 2H2 → MeOH requires H2/CO >= 2.
        """
        T = np.asarray(temperature_C, dtype=float)
        P = np.asarray(pressure_bar, dtype=float)
        r = self.h2_co_ratio if h2_co_ratio is None else np.asarray(h2_co_ratio, dtype=float)

        T_norm = (T - self.T_opt) / self.T_opt
        X_T = self.X_max * np.exp(-self.k_T * T_norm ** 2)
        X_P = (P / self.P_ref) ** self.P_exp

        X = X_T * X_P
        # Stoichiometric limit: 4 H2 per 2 CO for direct DME
        X_stoich = np.clip(r / 2.0, 0.0, 1.0)
        return np.clip(np.minimum(X, X_stoich), 0.0, 1.0)

    def selectivity_dme(self, temperature_C):
        """
        DME selectivity from total converted CO.
        Dehydration of MeOH to DME is favored at 250-280 degC.
        S_DME = S_max * exp(-k_S * ((T-T_S_opt)/T_S_opt)^2)
        """
        T = np.asarray(temperature_C, dtype=float)
        T_norm = (T - self.T_S_opt) / self.T_S_opt
        return np.clip(self.S_DME_max * np.exp(-self.k_S * T_norm ** 2), 0.0, 1.0)

    def dme_production_mol_s(self, temperature_C, pressure_bar, n_co_in=None):
        """DME production [mol/s]. 2 mol CO → 1 mol DME."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(temperature_C, pressure_bar)
        S = self.selectivity_dme(temperature_C)
        return X * n * S / 2.0   # factor 2: 2 CO → 1 DME

    def energy_efficiency(self, temperature_C, pressure_bar):
        """
        Efficiency = X * S_DME * LHV_DME/2 / (H2_CO * LHV_H2 + LHV_CO)
        """
        X = self.conversion(temperature_C, pressure_bar)
        S = self.selectivity_dme(temperature_C)
        eta = X * S * (self.LHV_DME / 2.0) / (self.h2_co_ratio * self.LHV_H2 + 1e-12)
        return np.clip(eta, 0.0, 1.0)

    def heat_released_kW(self, temperature_C, pressure_bar, n_co_in=None):
        """Exothermic heat [kW]."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(temperature_C, pressure_bar)
        return X * n * self.DH

    def compute(self, n_co_in, temperature_C, pressure_bar):
        """Full computation."""
        X    = self.conversion(temperature_C, pressure_bar)
        S    = self.selectivity_dme(temperature_C)
        dme  = self.dme_production_mol_s(temperature_C, pressure_bar, n_co_in)
        eta  = self.energy_efficiency(temperature_C, pressure_bar)
        Q    = self.heat_released_kW(temperature_C, pressure_bar, n_co_in)
        return {
            "co_conversion": X,
            "selectivity_dme": S,
            "dme_production_mol_s": dme,
            "energy_efficiency": eta,
            "heat_released_kW": Q,
        }
