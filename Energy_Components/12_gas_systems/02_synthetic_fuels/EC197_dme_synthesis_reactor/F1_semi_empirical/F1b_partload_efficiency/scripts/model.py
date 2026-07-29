"""
EC197 — DME Synthesis Reactor — F1b Part-Load + Thermal Integration Model

Extends F1a with:
  1. Part-load ratio correction on conversion and selectivity.
  2. Bed temperature drop at part-load (reduced exotherm).
  3. Catalyst deactivation (coking on HZSM-5 component).
  4. Heat recovery credit for pre-heating feed gas.
  5. Methanol slip fraction (by-product fraction = 1 - S_DME) at part-load.

References:
    Ereña, J. et al. (2005). Catal. Today, 107-108, 467-473.
    García-Trenco, A. et al. (2018). ACS Catal., 8, 4660-4671.
    Naik, S.P. et al. (2011). Fuel, 90, 3266-3273.
"""

import numpy as np


class DMEReactorF1b:
    """DME synthesis reactor F1b: part-load, thermal, catalyst deactivation."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_opt       = u["T_opt"]["value"]
        self.X_max       = u["X_max"]["value"]
        self.k_T         = u["k_T"]["value"]
        self.P_ref       = u["P_ref"]["value"]
        self.P_exp       = u["P_exp"]["value"]
        self.S_DME_max   = u["S_DME_max"]["value"]
        self.k_S         = u["k_S"]["value"]
        self.T_S_opt     = u["T_S_opt"]["value"]
        self.n_CO_in     = u["n_CO_in"]["value"]
        self.h2_co_ratio = u["H2_CO_ratio"]["value"]
        self.LHV_DME     = u["LHV_DME"]["value"]
        self.LHV_H2      = u["LHV_H2"]["value"]
        self.DH          = abs(u["DH_reaction"]["value"])
        self.f_recovery  = u["f_recovery"]["value"]
        self.PLR_coeffs  = u["PLR_coeffs"]["value"]
        self.dT_cold     = u["dT_cold_K"]["value"]
        self.k_deact     = u["k_deact_per_1000h"]["value"]

    def _plr_factor(self, plr):
        plr = np.asarray(plr, dtype=float)
        a0, a1, a2 = self.PLR_coeffs
        f_raw = a0 + a1 * plr + a2 * plr ** 2
        f_at_1 = a0 + a1 + a2
        return np.clip(f_raw / f_at_1, 0.0, 1.0)

    def _effective_temperature(self, T_set, plr):
        return np.asarray(T_set, dtype=float) - self.dT_cold * (1.0 - np.asarray(plr, dtype=float))

    def conversion(self, T_set, pressure_bar, plr=1.0,
                   operating_hours=0.0, h2_co_ratio=None):
        """CO conversion with part-load and catalyst deactivation."""
        T_set = np.asarray(T_set, dtype=float)
        P = np.asarray(pressure_bar, dtype=float)
        plr_arr = np.asarray(plr, dtype=float)
        r = self.h2_co_ratio if h2_co_ratio is None else np.asarray(h2_co_ratio, dtype=float)

        T_eff = self._effective_temperature(T_set, plr_arr)
        T_norm = (T_eff - self.T_opt) / self.T_opt
        X_T = self.X_max * np.exp(-self.k_T * T_norm ** 2)
        X_P = (P / self.P_ref) ** self.P_exp
        plr_f = self._plr_factor(plr_arr)

        X = X_T * X_P * plr_f
        X_stoich = np.clip(r / 2.0, 0.0, 1.0)
        X = np.minimum(X, X_stoich)

        # Catalyst deactivation (coking on HZSM-5 acid sites, faster than FT)
        hours = np.asarray(operating_hours, dtype=float)
        deact = np.clip(self.k_deact * hours / 1000.0, 0.0, 0.30)
        return np.clip(X * (1.0 - deact), 0.0, 1.0)

    def selectivity_dme(self, T_set, plr=1.0):
        """DME selectivity with temperature and PLR correction.
        At part-load (lower T), dehydration activity of HZSM-5 drops → more MeOH slip.
        """
        T_eff = self._effective_temperature(T_set, plr)
        T_norm = (T_eff - self.T_S_opt) / self.T_S_opt
        # Base selectivity at effective temperature
        S_base = self.S_DME_max * np.exp(-self.k_S * T_norm ** 2)
        # PLR penalty: at low PLR, contact time changes → reduced dehydration
        plr_arr = np.asarray(plr, dtype=float)
        S = S_base * (0.85 + 0.15 * plr_arr)
        return np.clip(S, 0.0, 1.0)

    def dme_production_mol_s(self, T_set, pressure_bar, plr=1.0,
                              operating_hours=0.0, n_co_in=None):
        """DME production [mol/s]."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(T_set, pressure_bar, plr, operating_hours)
        S = self.selectivity_dme(T_set, plr)
        return X * n * S / 2.0

    def meoh_slip_mol_s(self, T_set, pressure_bar, plr=1.0,
                         operating_hours=0.0, n_co_in=None):
        """Methanol by-product [mol/s] (unreacted MeOH not converted to DME)."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(T_set, pressure_bar, plr, operating_hours)
        S = self.selectivity_dme(T_set, plr)
        return X * n * (1.0 - S)

    def heat_recovery_kW(self, T_set, pressure_bar, plr=1.0,
                          operating_hours=0.0, n_co_in=None):
        """Recovered exothermic heat [kW]."""
        n = self.n_CO_in if n_co_in is None else np.asarray(n_co_in, dtype=float)
        X = self.conversion(T_set, pressure_bar, plr, operating_hours)
        return X * n * self.DH * self.f_recovery

    def energy_efficiency(self, T_set, pressure_bar, plr=1.0, operating_hours=0.0):
        """Overall energy efficiency including heat recovery credit."""
        X = self.conversion(T_set, pressure_bar, plr, operating_hours)
        S = self.selectivity_dme(T_set, plr)
        plr_arr = np.asarray(plr, dtype=float)
        num = X * S * (self.LHV_DME / 2.0) + X * self.DH * self.f_recovery
        den = (self.h2_co_ratio * self.LHV_H2) * plr_arr + 1e-12
        return np.clip(num / den, 0.0, 1.0)

    def deactivation_factor(self, operating_hours):
        hours = np.asarray(operating_hours, dtype=float)
        return 1.0 - np.clip(self.k_deact * hours / 1000.0, 0.0, 0.30)

    def compute(self, n_co_in, T_set, pressure_bar, plr=1.0, operating_hours=0.0):
        """Full computation."""
        X    = self.conversion(T_set, pressure_bar, plr, operating_hours)
        S    = self.selectivity_dme(T_set, plr)
        T_eff = self._effective_temperature(T_set, plr)
        dme  = self.dme_production_mol_s(T_set, pressure_bar, plr, operating_hours, n_co_in)
        meoh = self.meoh_slip_mol_s(T_set, pressure_bar, plr, operating_hours, n_co_in)
        Q    = self.heat_recovery_kW(T_set, pressure_bar, plr, operating_hours, n_co_in)
        eta  = self.energy_efficiency(T_set, pressure_bar, plr, operating_hours)
        deact = self.deactivation_factor(operating_hours)

        return {
            "co_conversion": X,
            "effective_temperature_C": T_eff,
            "selectivity_dme": S,
            "dme_production_mol_s": dme,
            "meoh_slip_mol_s": meoh,
            "heat_recovery_kW": Q,
            "energy_efficiency": eta,
            "deactivation_factor": deact,
        }
