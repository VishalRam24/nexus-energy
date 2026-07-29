"""
EC196 — Synthetic Jet Fuel (Power-to-Liquid) — F1b Part-Load + Thermal Integration Model

Extends F1a with:
  1. Part-load ratio (PLR) correction on CO conversion — at reduced feed rates,
     the catalyst bed cools (less exotherm) → conversion drops slightly:
         T_eff(PLR) = T_set - dT_cold * (1 - PLR)
         X_PLR = X(T_eff, P) * PLR_factor(PLR)
     PLR_factor = quadratic polynomial (normalized to 1 at PLR=1).
  2. ASF alpha correction at lower temperature:
         alpha(T) = alpha_ref + alpha_k * (T_opt - T_eff)   [temperature sensitivity]
     Lower T → higher alpha → more wax, less jet fraction (trade-off).
  3. Exothermic heat integration credit:
         Q_recovered = X * n_CO * DH * f_recovery  [kW]
     Used for pre-heating feed or driving rWGS section.
  4. Catalyst deactivation proxy (sintering/coking):
         X_deact = X_PLR * (1 - k_deact * t_operating_h / 1000)  [limited to 20% loss]
     References typical Co-catalyst decline of 0.5-1% per 1000 h (Dry 2002).

References:
    Dry, M.E. (2002). The Fischer-Tropsch process: 1950-2000. Catalysis Today, 71, 227-241.
    Schulz, H. (1999). Short history and present trends of FT synthesis. Appl. Cat. A, 186, 3-12.
    De Klerk, A. (2011). Fischer-Tropsch Refining. Wiley-VCH. Ch. 5.
    Hillestad, M. et al. (2018). Chemical Engineering Journal, 364, 520-531.
"""

import numpy as np


class FTJetFuelF1b:
    """
    Fischer-Tropsch Power-to-Liquid F1b: part-load, thermal integration,
    ASF alpha temperature dependence, catalyst deactivation.
    """

    def __init__(self, params: dict):
        u = params["unit"]

        self.T_opt       = u["T_opt"]["value"]
        self.X_max       = u["X_max"]["value"]
        self.k_T         = u["k_T"]["value"]
        self.P_ref       = u["P_ref"]["value"]
        self.P_exp       = u["P_exp"]["value"]
        self.alpha_ref   = u["alpha_ASF"]["value"]
        self.alpha_k     = u["alpha_k_per_K"]["value"]   # alpha change per K (negative = alpha increases as T drops)
        self.n_CO_in     = u["n_CO_in"]["value"]
        self.h2_co_ratio = u["H2_CO_ratio"]["value"]
        self.LHV_FT      = u["LHV_FT_kJ_mol"]["value"]
        self.LHV_H2      = u["LHV_H2"]["value"]
        self.LHV_CO      = u["LHV_CO"]["value"]
        self.DH          = abs(u["DH_reaction"]["value"])
        self.f_recovery  = u["f_recovery"]["value"]
        self.PLR_coeffs  = u["PLR_coeffs"]["value"]
        self.dT_cold     = u["dT_cold_K"]["value"]       # K bed temperature drop at PLR=0
        self.k_deact     = u["k_deact_per_1000h"]["value"]  # fraction/1000h

    # ------------------------------------------------------------------
    # Part-load helpers
    # ------------------------------------------------------------------
    def _plr_factor(self, plr):
        """Polynomial PLR correction, normalized so PLR=1 → factor=1."""
        plr = np.asarray(plr, dtype=float)
        a0, a1, a2 = self.PLR_coeffs
        f_raw = a0 + a1 * plr + a2 * plr ** 2
        f_at_1 = a0 + a1 + a2
        return np.clip(f_raw / f_at_1, 0.0, 1.0)

    def _effective_temperature(self, T_set, plr):
        """Effective bed temperature at part-load (less exotherm → cooler)."""
        return np.asarray(T_set, dtype=float) - self.dT_cold * (1.0 - np.asarray(plr, dtype=float))

    def _alpha_at_T(self, T_eff):
        """ASF alpha corrected for temperature. Lower T → higher alpha (more wax)."""
        return np.clip(
            self.alpha_ref + self.alpha_k * (self.T_opt - np.asarray(T_eff, dtype=float)),
            0.70, 0.98
        )

    # ------------------------------------------------------------------
    # ASF selectivity
    # ------------------------------------------------------------------
    def asf_selectivity_jet(self, alpha=None):
        """C8-C16 selectivity from ASF distribution (sum of W_n, n=8..16)."""
        a = self.alpha_ref if alpha is None else np.asarray(alpha, dtype=float)
        n_range = np.arange(8, 17)
        W = n_range * (1.0 - a) ** 2 * a ** (n_range - 1)
        if np.ndim(a) == 0:
            return float(np.sum(W))
        return np.sum(W, axis=-1) if W.ndim > 1 else float(np.sum(W))

    # ------------------------------------------------------------------
    # Conversion with part-load and deactivation
    # ------------------------------------------------------------------
    def conversion(self, T_set, pressure_bar, plr=1.0,
                   operating_hours=0.0, h2_co_ratio=None):
        """
        CO conversion with part-load, temperature, and catalyst deactivation.

        Parameters
        ----------
        T_set          : Setpoint temperature [degC]
        pressure_bar   : Reactor pressure [bar]
        plr            : Part-load ratio [0-1]
        operating_hours: Cumulative catalyst hours [h]
        h2_co_ratio    : H2/CO ratio (optional override)
        """
        T_set = np.asarray(T_set, dtype=float)
        P = np.asarray(pressure_bar, dtype=float)
        plr_arr = np.asarray(plr, dtype=float)
        r = self.h2_co_ratio if h2_co_ratio is None else np.asarray(h2_co_ratio, dtype=float)

        T_eff = self._effective_temperature(T_set, plr_arr)
        T_norm = (T_eff - self.T_opt) / self.T_opt
        X_T = self.X_max * np.exp(-self.k_T * T_norm ** 2)
        X_P = (P / self.P_ref) ** self.P_exp

        plr_f = self._plr_factor(plr_arr)
        X_base = X_T * X_P * plr_f

        # Stoichiometric limit
        X_stoich = np.clip(r / 2.1, 0.0, 1.0)
        X_base = np.minimum(X_base, X_stoich)

        # Catalyst deactivation (sintering, coking)
        hours = np.asarray(operating_hours, dtype=float)
        deact = np.clip(self.k_deact * hours / 1000.0, 0.0, 0.20)
        return np.clip(X_base * (1.0 - deact), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Jet fuel yield and efficiency
    # ------------------------------------------------------------------
    def jet_fuel_yield_mol_s(self, T_set, pressure_bar, plr=1.0,
                              operating_hours=0.0, n_co_in=None):
        """Jet fuel (C8-C16) production [mol/s as C12 equivalent]."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(T_set, pressure_bar, plr, operating_hours)
        T_eff = self._effective_temperature(T_set, plr)
        alpha = self._alpha_at_T(T_eff)
        S = self.asf_selectivity_jet(alpha)
        return X * n * S / 12.0

    def heat_recovery_kW(self, T_set, pressure_bar, plr=1.0,
                          operating_hours=0.0, n_co_in=None):
        """Recovered exothermic heat [kW]."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(T_set, pressure_bar, plr, operating_hours)
        return X * n * self.DH * self.f_recovery

    def energy_efficiency(self, T_set, pressure_bar, plr=1.0, operating_hours=0.0):
        """Power-to-jet efficiency including heat integration credit."""
        X = self.conversion(T_set, pressure_bar, plr, operating_hours)
        T_eff = self._effective_temperature(T_set, plr)
        alpha = self._alpha_at_T(T_eff)
        S = self.asf_selectivity_jet(alpha)
        plr_arr = np.asarray(plr, dtype=float)
        numerator = X * S * self.LHV_FT + X * self.DH * self.f_recovery
        denominator = (self.h2_co_ratio * self.LHV_H2 + self.LHV_CO) * plr_arr + 1e-12
        return np.clip(numerator / denominator, 0.0, 1.0)

    def deactivation_factor(self, operating_hours):
        """Fractional conversion remaining after deactivation."""
        hours = np.asarray(operating_hours, dtype=float)
        return 1.0 - np.clip(self.k_deact * hours / 1000.0, 0.0, 0.20)

    # ------------------------------------------------------------------
    # Full compute
    # ------------------------------------------------------------------
    def compute(self, n_co_in, T_set, pressure_bar, plr=1.0, operating_hours=0.0):
        """Full computation returning all F1b outputs."""
        X = self.conversion(T_set, pressure_bar, plr, operating_hours, None)
        T_eff = self._effective_temperature(T_set, plr)
        alpha = self._alpha_at_T(T_eff)
        S = self.asf_selectivity_jet(alpha)
        jet = self.jet_fuel_yield_mol_s(T_set, pressure_bar, plr, operating_hours, n_co_in)
        Q = self.heat_recovery_kW(T_set, pressure_bar, plr, operating_hours, n_co_in)
        eta = self.energy_efficiency(T_set, pressure_bar, plr, operating_hours)
        deact = self.deactivation_factor(operating_hours)

        return {
            "co_conversion": X,
            "effective_temperature_C": T_eff,
            "alpha_ASF": alpha,
            "selectivity_jet_C8_C16": S,
            "jet_fuel_mol_s": jet,
            "heat_recovery_kW": Q,
            "energy_efficiency": eta,
            "deactivation_factor": deact,
        }
