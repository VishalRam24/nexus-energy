"""
EC196 — Synthetic Jet Fuel (Power-to-Liquid) — F1a Fischer-Tropsch Conversion Model

Power-to-Liquid chain: CO2/H2 → reverse WGS → CO/H2 (syngas) → FT reactor → C8-C16 jet fuel.

F1a models the Fischer-Tropsch (FT) reactor conversion step:
  - Anderson-Schulz-Flory (ASF) distribution to estimate C8+ selectivity:
        W_n = n * (1 - alpha)^2 * alpha^(n-1)   (weight fraction of Cn product)
        S_jet = sum(W_n, n=8..16)  [fraction of total FT product going to jet-fuel cut]
  - Syngas conversion:
        X_CO = X_max * exp(-k_T * ((T-T_opt)/T_opt)^2) * (P/P_ref)^P_exp
  - Overall plant energy efficiency (power-to-jet):
        eta = X_CO * S_jet * LHV_FT / (LHV_H2 * H2_per_CO + LHV_CO * 1)
  - CO consumption and jet fuel production rate.

References:
    Anderson, R.B. (1956). Catalysts for the Fischer-Tropsch Synthesis, Vol. 4.
    Schulz, H. (1999). Applied Catalysis A: General, 186(1-2), 3-12.
    De Klerk, A. (2011). Fischer-Tropsch Refining. Wiley-VCH.
    Dry, M.E. (2002). The Fischer-Tropsch process: 1950-2000. Catalysis Today, 71, 227-241.
    Hillestad, M. et al. (2018). Improving carbon efficiency and profitability of the
        power to liquid process with a sorption enhanced Fischer-Tropsch reactor.
        Chemical Engineering Journal, 364, 520-531.
"""

import numpy as np


class FTJetFuelF1a:
    """
    Fischer-Tropsch power-to-liquid reactor — F1a equilibrium conversion model.
    Predicts CO conversion, C8-C16 selectivity, jet fuel yield, and efficiency.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_opt     = u["T_opt"]["value"]           # degC (FT catalyst optimal)
        self.X_max     = u["X_max"]["value"]           # max CO conversion fraction
        self.k_T       = u["k_T"]["value"]             # Gaussian temperature width
        self.P_ref     = u["P_ref"]["value"]           # bar
        self.P_exp     = u["P_exp"]["value"]           # pressure exponent
        self.alpha_ASF = u["alpha_ASF"]["value"]       # chain-growth probability (0.85-0.92 LTFT)
        self.n_CO_in   = u["n_CO_in"]["value"]         # mol/s design CO feed
        self.h2_co_ratio = u["H2_CO_ratio"]["value"]   # stoichiometric ratio (typically 2.0-2.15)
        self.LHV_FT    = u["LHV_FT_kJ_mol"]["value"]  # kJ/mol average for C12 surrogate
        self.LHV_H2    = u["LHV_H2"]["value"]          # kJ/mol
        self.LHV_CO    = u["LHV_CO"]["value"]          # kJ/mol (HHV basis vs LHV: ~283 kJ/mol)
        self.DH        = abs(u["DH_reaction"]["value"])  # kJ/mol CO (FT exotherm ~165 kJ/mol)

    # ------------------------------------------------------------------
    # ASF distribution → C8-C16 selectivity
    # ------------------------------------------------------------------
    def asf_selectivity_jet(self, alpha=None):
        """
        Anderson-Schulz-Flory selectivity for C8-C16 (jet fuel cut).

        W_n = n * (1 - alpha)^2 * alpha^(n-1)
        S_jet = sum W_n for n=8..16
        """
        a = self.alpha_ASF if alpha is None else np.asarray(alpha, dtype=float)
        n_range = np.arange(8, 17)   # C8 to C16
        W = n_range * (1.0 - a) ** 2 * a ** (n_range - 1)
        return float(np.sum(W))

    # ------------------------------------------------------------------
    # CO conversion
    # ------------------------------------------------------------------
    def conversion(self, temperature_C, pressure_bar, h2_co_ratio=None):
        """
        CO conversion fraction based on Gaussian equilibrium fit.
        LTFT (200-230 degC, Co catalyst): high alpha (~0.90), high C5+ selectivity.
        HTFT (320-350 degC, Fe catalyst): lower alpha (~0.70).

        X = X_max * exp(-k_T * ((T-T_opt)/T_opt)^2) * (P/P_ref)^P_exp
        Limited by stoichiometry: X <= h2_co / (H2_CO_stoichiometric)
        """
        T = np.asarray(temperature_C, dtype=float)
        P = np.asarray(pressure_bar, dtype=float)
        r = self.h2_co_ratio if h2_co_ratio is None else np.asarray(h2_co_ratio, dtype=float)

        T_norm = (T - self.T_opt) / self.T_opt
        X_T = self.X_max * np.exp(-self.k_T * T_norm ** 2)
        X_P = (P / self.P_ref) ** self.P_exp

        X = X_T * X_P
        # Stoichiometric limit: each CO needs ~2 H2
        X_stoich = np.clip(r / 2.1, 0.0, 1.0)
        return np.clip(np.minimum(X, X_stoich), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Jet fuel yield and energy efficiency
    # ------------------------------------------------------------------
    def jet_fuel_yield_mol_s(self, temperature_C, pressure_bar, n_co_in=None):
        """Jet fuel (C8-C16 cut) production [mol/s equivalent as C12 surrogate]."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(temperature_C, pressure_bar)
        S_jet = self.asf_selectivity_jet()
        # FT products: each CO → ~1/n_C carbon atoms in Cn chain
        # Simplified: n_jet = (X * n_CO / 12) * S_jet  (per C12 average)
        return X * n * S_jet / 12.0

    def energy_efficiency(self, temperature_C, pressure_bar):
        """
        Power-to-jet energy efficiency:
        eta = X * S_jet * LHV_FT / (H2_CO_ratio * LHV_H2 + LHV_CO)
        """
        X = self.conversion(temperature_C, pressure_bar)
        S_jet = self.asf_selectivity_jet()
        eta_num = X * S_jet * self.LHV_FT
        eta_den = self.h2_co_ratio * self.LHV_H2 + self.LHV_CO
        return np.clip(eta_num / (eta_den + 1e-12), 0.0, 1.0)

    def heat_released_kW(self, temperature_C, pressure_bar, n_co_in=None):
        """Exothermic FT heat [kW]."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(temperature_C, pressure_bar)
        return X * n * self.DH   # kJ/s = kW

    def compute(self, n_co_in, temperature_C, pressure_bar):
        """Full computation."""
        X   = self.conversion(temperature_C, pressure_bar)
        S   = self.asf_selectivity_jet()
        jet = self.jet_fuel_yield_mol_s(temperature_C, pressure_bar, n_co_in)
        eta = self.energy_efficiency(temperature_C, pressure_bar)
        Q   = self.heat_released_kW(temperature_C, pressure_bar, n_co_in)
        return {
            "co_conversion": X,
            "selectivity_jet_C8_C16": S,
            "jet_fuel_mol_s": jet,
            "energy_efficiency": eta,
            "heat_released_kW": Q,
        }
