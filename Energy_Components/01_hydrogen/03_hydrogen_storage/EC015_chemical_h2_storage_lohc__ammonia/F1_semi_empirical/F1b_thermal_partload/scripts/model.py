"""
EC015 -- Chemical H2 Storage (LOHC / Ammonia) -- F1b Thermal + Part-load Model

Extends F1a by adding:
1. Temperature-dependent reactor efficiency via Arrhenius correction
2. Part-load efficiency penalty: eta_pl = (F/F_nom)^n
3. Temperature-dependent heat demand (Kirchhoff correction for ΔH(T))
4. Round-trip efficiency accounting for heat integration

Temperature corrections:
    LOHC dehydrogenation:
        eta(T) = eta_nom * exp(-E_a/R * (1/T - 1/T_nom))
        At T < T_nom: eta < eta_nom (slower kinetics -> lower conversion)
        At T > T_nom: eta approaches 1 but capped by thermodynamic limit

    Part-load:
        eta_partload(F) = (F / F_nom)^n   [power-law; n~0.1-0.15]
        Combined: eta_eff = eta_T * eta_pl

Heat demand:
    Q_thermal(T) = n_H2 * ΔH_rxn / eta_eff

References:
    Preuster et al. (2017). Acc. Chem. Res. 50(1), 74-85.
    Niermann et al. (2021). Energy Environ. Sci. 14, 1928-1944.
    Reuse et al. (2004). Chem. Eng. J. 101(1-3), 133-141.
    Lamb et al. (2019). Int. J. Hydrogen Energy 44(7), 3580-3593.
"""

import numpy as np

R_UNIV = 8.314      # J/(mol K)
M_H2   = 0.002016   # kg/mol


def _arrhenius_eta(eta_nom, E_a, T_K, T_nom_K):
    """
    Temperature-corrected efficiency via Arrhenius factor (relative to nominal).

    eta(T) = eta_nom * exp(-E_a/R * (1/T - 1/T_nom))

    At T = T_nom: eta = eta_nom (reference).
    At T > T_nom: factor > 1, but efficiency is capped at min(result, 0.99).
    At T < T_nom: factor < 1, eta drops.
    """
    T   = np.asarray(T_K, dtype=float)
    T_n = float(T_nom_K)
    factor = np.exp(-E_a / R_UNIV * (1.0 / T - 1.0 / T_n))
    return np.clip(eta_nom * factor, 0.01, 0.99)


def _partload_eta(F_kg_s, F_nom, exponent):
    """Part-load efficiency factor = (F/F_nom)^n, clipped to [0, 1]."""
    F = np.asarray(F_kg_s, dtype=float)
    ratio = np.clip(F / F_nom, 0.1, 1.0)   # below 10% load not modeled
    return ratio ** exponent


class ChemicalH2StorageF1b:
    """Chemical H2 storage (LOHC-DBT and Ammonia): temperature + part-load corrections."""

    def __init__(self, params: dict):
        L  = params["lohc_dbt"]
        A  = params["ammonia"]
        th = params["thermodynamics"]

        # LOHC parameters
        self.lohc_cap   = float(L["gravimetric_capacity_wt_pct"]["value"]) / 100.0
        self.lohc_dH_h  = float(L["delta_H_hydrogenation"]["value"])     # J/mol (negative)
        self.lohc_dH_d  = float(L["delta_H_dehydrogenation"]["value"])   # J/mol (positive)
        self.lohc_T_h   = float(L["T_nom_hydrogenation_K"]["value"])
        self.lohc_T_d   = float(L["T_nom_dehydrogenation_K"]["value"])
        self.lohc_eta0  = float(L["energy_efficiency_nom"]["value"])
        self.lohc_Ea    = float(L["T_activation_K"]["value"])
        self.lohc_exp   = float(L["part_load_exponent"]["value"])
        self.lohc_Fnom  = float(L["F_nom_kg_s"]["value"])

        # NH3 parameters
        self.nh3_cap    = float(A["gravimetric_capacity_wt_pct"]["value"]) / 100.0
        self.nh3_dH_s   = float(A["delta_H_synthesis"]["value"])          # J/mol (negative)
        self.nh3_dH_c   = float(A["delta_H_cracking"]["value"])           # J/mol (positive)
        self.nh3_T_s    = float(A["T_nom_synthesis_K"]["value"])
        self.nh3_T_c    = float(A["T_nom_cracking_K"]["value"])
        self.nh3_eta0   = float(A["energy_efficiency_nom"]["value"])
        self.nh3_Ea     = float(A["T_activation_K"]["value"])
        self.nh3_exp    = float(A["part_load_exponent"]["value"])
        self.nh3_Fnom   = float(A["F_nom_kg_s"]["value"])

        self.M_H2       = float(th["M_H2"]["value"])
        self.LHV_H2     = float(th["LHV_H2"]["value"])

    # ------------------------------------------------------------------
    # LOHC methods
    # ------------------------------------------------------------------

    def lohc_efficiency(self, T_K, F_kg_s=None):
        """
        Effective LOHC dehydrogenation efficiency at temperature T and flow rate F.

        Args:
            T_K:    Reactor temperature [K]
            F_kg_s: H2 flow rate [kg/s] for part-load correction (optional)

        Returns:
            eta_eff [0, 1]
        """
        eta_T = _arrhenius_eta(self.lohc_eta0, self.lohc_Ea, T_K, self.lohc_T_d)
        if F_kg_s is not None:
            eta_pl = _partload_eta(F_kg_s, self.lohc_Fnom, self.lohc_exp)
            return eta_T * eta_pl
        return eta_T

    def lohc_thermal_energy(self, m_H2_kg, direction="dehydrogenation",
                             T_K=None, F_kg_s=None):
        """
        Thermal energy demand [MJ] at given temperature and part-load.

        Args:
            m_H2_kg:   Mass of H2 [kg]
            direction: 'hydrogenation' or 'dehydrogenation'
            T_K:       Reactor temperature [K] (optional; uses T_nom if None)
            F_kg_s:    H2 flow rate for part-load (optional)

        Returns:
            Q [MJ]; positive = heat required, negative = heat released
        """
        m = np.asarray(m_H2_kg, dtype=float)
        n_H2 = m / self.M_H2

        if direction == "dehydrogenation":
            if T_K is not None:
                eta_eff = self.lohc_efficiency(T_K, F_kg_s)
            else:
                eta_eff = self.lohc_eta0
            Q_J = n_H2 * self.lohc_dH_d / eta_eff
        else:  # hydrogenation
            Q_J = n_H2 * self.lohc_dH_h  # negative (exothermic)

        return Q_J / 1e6  # MJ

    def lohc_specific_energy(self, direction="dehydrogenation", T_K=None, F_kg_s=None):
        """Specific thermal energy [MJ/kg_H2]."""
        if direction == "dehydrogenation":
            if T_K is not None:
                eta_eff = self.lohc_efficiency(T_K, F_kg_s)
            else:
                eta_eff = self.lohc_eta0
            Q_per_mol = self.lohc_dH_d / eta_eff
        else:
            Q_per_mol = self.lohc_dH_h
        return Q_per_mol / self.M_H2 / 1e6  # MJ/kg_H2

    def lohc_carrier_mass(self, m_H2_kg):
        """Mass of H0-DBT carrier required [kg]."""
        m = np.asarray(m_H2_kg, dtype=float)
        return m * (1.0 - self.lohc_cap) / self.lohc_cap

    def lohc_roundtrip_efficiency(self, T_dehydro_K=None, F_kg_s=None):
        """Round-trip efficiency: LHV_H2 / (LHV_H2 + |Q_dehydro/kg|)."""
        q = abs(self.lohc_specific_energy("dehydrogenation", T_K=T_dehydro_K, F_kg_s=F_kg_s)) * 1e6  # J/kg
        return self.LHV_H2 / (self.LHV_H2 + q)

    # ------------------------------------------------------------------
    # Ammonia methods
    # ------------------------------------------------------------------

    def nh3_efficiency(self, T_K, F_kg_s=None):
        """Effective NH3 cracking efficiency at temperature T."""
        eta_T = _arrhenius_eta(self.nh3_eta0, self.nh3_Ea, T_K, self.nh3_T_c)
        if F_kg_s is not None:
            eta_pl = _partload_eta(F_kg_s, self.nh3_Fnom, self.nh3_exp)
            return eta_T * eta_pl
        return eta_T

    def nh3_thermal_energy(self, m_H2_kg, direction="cracking", T_K=None, F_kg_s=None):
        """Thermal energy [MJ] for NH3 synthesis or cracking."""
        m = np.asarray(m_H2_kg, dtype=float)
        n_H2 = m / self.M_H2

        if direction == "cracking":
            if T_K is not None:
                eta_eff = self.nh3_efficiency(T_K, F_kg_s)
            else:
                eta_eff = self.nh3_eta0
            Q_J = n_H2 * self.nh3_dH_c / eta_eff
        else:  # synthesis
            Q_J = n_H2 * self.nh3_dH_s  # negative (exothermic)

        return Q_J / 1e6  # MJ

    def nh3_specific_energy(self, direction="cracking", T_K=None, F_kg_s=None):
        """Specific thermal energy [MJ/kg_H2] for NH3 cracking."""
        if direction == "cracking":
            eta_eff = self.nh3_efficiency(T_K, F_kg_s) if T_K is not None else self.nh3_eta0
            Q_per_mol = self.nh3_dH_c / eta_eff
        else:
            Q_per_mol = self.nh3_dH_s
        return Q_per_mol / self.M_H2 / 1e6

    def nh3_carrier_mass(self, m_H2_kg):
        """Mass of NH3 required to store m_H2_kg [kg]."""
        m = np.asarray(m_H2_kg, dtype=float)
        return m * (1.0 - self.nh3_cap) / self.nh3_cap

    def nh3_roundtrip_efficiency(self, T_crack_K=None, F_kg_s=None):
        """Round-trip efficiency for NH3 storage."""
        q = abs(self.nh3_specific_energy("cracking", T_K=T_crack_K, F_kg_s=F_kg_s)) * 1e6
        return self.LHV_H2 / (self.LHV_H2 + q)
