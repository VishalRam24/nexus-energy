"""
EC014 -- Metal Hydride H2 Storage -- F1b van't Hoff Thermal Model

Extends F1a by adding:
1. First-order sorption kinetics with Arrhenius rate constants
2. Lumped thermal balance (reaction heat + sensible heat)

van't Hoff equilibrium (Sandrock 1999):
    Desorption convention:  ΔH_des = -ΔH_formation > 0  (endothermic, heat absorbed)
                            ΔS_des = -ΔS_formation > 0  (entropy increases)

    Correct van't Hoff sign (Sandrock 1999, Eq. 1):
        ln(P_eq / P_ref) = -ΔH_des / (R*T) + ΔS_des / R
    where P_ref = 1 atm = 1.01325 bar.
    This gives d(ln_P)/dT = +ΔH_des/(R*T^2) > 0 (pressure rises with temperature). ✓

Sorption kinetics (Mayer et al. 1987):
    Absorption:  dC/dt = Ca_abs * exp(-Ea_abs/(R*T)) * ln(P/P_eq) * (C_max - C)
    Desorption:  dC/dt = Ca_des * exp(-Ea_des/(R*T)) * (P/P_eq - 1) * C
    where C = stored H2 mass fraction [kg_H2/kg_bed], C_max = H_max_wt_pct/100

Thermal balance:
    m_bed * cp_bed * dT/dt = Q_rxn + Q_cool
    Q_rxn = -ΔH_des/M_H2 * (dm_H2/dt)   [W; positive when exothermic absorption]
    Q_cool = -UA_bed * (T - T_amb)        [W; cooling when T > T_amb]

References:
    Sandrock (1999). J. Alloys Compd. 293-295, 877-888.
    Mayer et al. (1987). Int. J. Hydrogen Energy 12(11), 753-761.
    Lototskyy et al. (2014). Prog. Nat. Sci. Mater. Int. 24(2), 97-116.
"""

import numpy as np

R_UNIVERSAL = 8.314      # J/(mol K)
M_H2        = 0.002016   # kg/mol
P_REF_BAR   = 1.01325    # standard atmosphere in bar


class MetalHydrideH2F1b:
    """Metal hydride H2 storage: van't Hoff equilibrium + kinetics + thermal."""

    def __init__(self, params: dict):
        mat   = params["material"]
        therm = params["thermodynamics"]
        kin   = params["kinetics"]
        th    = params["thermal"]

        # Thermodynamic parameters (stored as FORMATION values, negative for absorption)
        self.delta_H = float(mat["delta_H"]["value"])           # J/mol (absorption, negative)
        self.delta_S = float(mat["delta_S"]["value"])           # J/mol/K (absorption, negative)
        self.H_max   = float(mat["H_max_wt_pct"]["value"])      # wt%
        self.m_bed   = float(mat["bed_mass_kg"]["value"])       # kg
        self.V_bed   = float(mat["bed_volume_m3"]["value"])     # m3

        self.hf      = float(therm["hysteresis_factor"]["value"])
        self.slope   = float(therm["plateau_slope_factor"]["value"])

        # Kinetic parameters
        self.Ca_abs  = float(kin["Ca_abs"]["value"])            # 1/s
        self.Ca_des  = float(kin["Ca_des"]["value"])            # 1/s
        self.Ea_abs  = float(kin["Ea_abs"]["value"])            # J/mol
        self.Ea_des  = float(kin["Ea_des"]["value"])            # J/mol

        # Thermal parameters
        self.cp_bed  = float(th["cp_hydride"]["value"])         # J/kg/K
        self.UA_bed  = float(th["UA_bed"]["value"])             # W/K
        self.T_ref   = float(th["T_ref"]["value"])              # K

        # Max H2 mass
        self.C_max   = self.H_max / 100.0                       # kg_H2/kg_bed (mass fraction)
        self.m_H2_max = self.C_max * self.m_bed                 # kg

    # ------------------------------------------------------------------
    # Equilibrium plateau pressure (Sandrock 1999 sign convention)
    # ------------------------------------------------------------------

    def plateau_pressure(self, T_K, mode="desorption", soc=0.5):
        """
        Equilibrium plateau pressure [bar] from the van't Hoff equation.

        Sandrock (1999) sign convention:
            ln(P_eq / P_ref) = -ΔH_des/(R*T) + ΔS_des/R
        where ΔH_des = -ΔH_formation > 0 (endothermic desorption).
        This ensures d(ln_P)/dT > 0 (plateau pressure rises with T). ✓

        Args:
            T_K:  Temperature [K]
            mode: 'absorption' or 'desorption'
            soc:  State of charge (0-1) for sloped plateau correction

        Returns:
            P_eq [bar]
        """
        T   = np.asarray(T_K, dtype=float)
        soc = np.asarray(soc, dtype=float)

        # Convert formation to desorption values (sign flip for endothermic reaction)
        dH_des = -self.delta_H   # > 0  (J/mol_H2)
        dS_des = -self.delta_S   # > 0  (J/mol_H2/K)

        # Sandrock (1999): ln(P/P_ref) = -ΔH_des/(RT) + ΔS_des/R
        ln_P_over_Pref = -dH_des / (R_UNIVERSAL * T) + dS_des / R_UNIVERSAL

        # Absolute ln_P
        ln_P = ln_P_over_Pref + np.log(P_REF_BAR)

        # Sloped plateau
        ln_P = ln_P + self.slope * (soc - 0.5)

        # Hysteresis: absorption plateau is HIGHER (requires more driving pressure)
        if mode == "absorption":
            ln_P = ln_P + np.log(self.hf)

        return np.exp(ln_P)   # bar

    # ------------------------------------------------------------------
    # Sorption kinetics (Mayer et al. 1987)
    # ------------------------------------------------------------------

    def sorption_rate(self, T_K, P_bar, soc, mode="absorption"):
        """
        H2 sorption rate [kg_H2/s].

        Absorption (charging): rate = Ca * exp(-Ea_abs/RT) * ln(P/P_eq) * (C_max - C)
        Desorption (discharging): rate = Ca * exp(-Ea_des/RT) * (P/P_eq - 1) * C

        Sign convention: positive = absorption (H2 stored), negative = desorption (H2 released)

        Args:
            T_K:  Temperature [K]
            P_bar: Gas pressure [bar]
            soc:  State of charge (0-1)
            mode: 'absorption' or 'desorption'

        Returns:
            dm_H2/dt [kg/s]; positive = absorption
        """
        T   = np.asarray(T_K, dtype=float)
        P   = np.asarray(P_bar, dtype=float)
        soc = np.clip(np.asarray(soc, dtype=float), 1e-6, 1.0 - 1e-6)

        P_eq = self.plateau_pressure(T, mode=mode, soc=soc)
        C    = soc * self.C_max   # current H2 fraction [kg/kg]

        if mode == "absorption":
            k   = self.Ca_abs * np.exp(-self.Ea_abs / (R_UNIVERSAL * T))
            # Driving force: ln(P/P_eq) for absorption (Mayer 1987)
            df  = np.log(np.maximum(P / np.maximum(P_eq, 1e-6), 1e-10))
            df  = np.maximum(df, 0.0)          # only absorb when P > P_eq
            dC_dt = k * df * (self.C_max - C)  # [1/s * (kg_H2/kg_bed)]
        else:
            k   = self.Ca_des * np.exp(-self.Ea_des / (R_UNIVERSAL * T))
            df  = P_eq / np.maximum(P, 1e-6) - 1.0
            df  = np.maximum(df, 0.0)          # only desorb when P < P_eq
            dC_dt = -k * df * C                # negative = desorption

        # Convert from [kg_H2/kg_bed / s] to [kg_H2/s]
        return dC_dt * self.m_bed

    # ------------------------------------------------------------------
    # Reaction heat
    # ------------------------------------------------------------------

    def reaction_heat(self, dm_H2_dt_kg_s):
        """
        Heat generation rate from sorption reaction [W].

        Q_rxn = ΔH_des/M_H2 * (dm_H2/dt)
        For absorption (dm/dt > 0): ΔH_des > 0 but reaction is EXOTHERMIC (heat released).
        Sign: Q_rxn = +|ΔH_abs|/M_H2 * dm/dt  (positive when absorbing = heat release)

        Returns:
            Q_rxn [W]; positive = heat released to bed
        """
        dm = np.asarray(dm_H2_dt_kg_s, dtype=float)
        # |ΔH_abs| = |delta_H| = -delta_H (since delta_H < 0 for absorption)
        Q_per_kg = abs(self.delta_H) / M_H2   # J/kg_H2
        return dm * Q_per_kg   # W (positive when absorbing = exothermic)

    # ------------------------------------------------------------------
    # Thermal balance (lumped)
    # ------------------------------------------------------------------

    def dTdt(self, T_K, dm_H2_dt_kg_s, T_amb_K):
        """
        Temperature derivative [K/s].

        m_bed * cp_bed * dT/dt = Q_rxn - UA_bed * (T - T_amb)

        Args:
            T_K:             Bed temperature [K]
            dm_H2_dt_kg_s:  H2 sorption rate [kg/s]; positive = absorption
            T_amb_K:         Ambient/coolant temperature [K]

        Returns:
            dT/dt [K/s]
        """
        T   = np.asarray(T_K, dtype=float)
        Q_rxn  = self.reaction_heat(dm_H2_dt_kg_s)
        Q_cool = self.UA_bed * (T - np.asarray(T_amb_K, dtype=float))
        return (Q_rxn - Q_cool) / (self.m_bed * self.cp_bed)

    # ------------------------------------------------------------------
    # Stored mass and densities
    # ------------------------------------------------------------------

    def stored_mass(self, soc):
        """H2 mass stored [kg]."""
        return np.clip(np.asarray(soc, dtype=float), 0.0, 1.0) * self.m_H2_max

    def gravimetric_density(self, soc):
        """Gravimetric H2 capacity [wt%]."""
        m_h2 = self.stored_mass(soc)
        return m_h2 / (m_h2 + self.m_bed) * 100.0

    def volumetric_density(self, soc):
        """Volumetric H2 density [kg_H2/m3]."""
        return self.stored_mass(soc) / self.V_bed

    # ------------------------------------------------------------------
    # Full evaluate
    # ------------------------------------------------------------------

    def evaluate(self, T_K, P_bar, soc, mode="desorption", T_amb_K=298.15):
        """
        Full operating-point evaluation.

        Parameters
        ----------
        T_K:     Bed temperature [K]
        P_bar:   Gas pressure [bar]
        soc:     State of charge (0-1)
        mode:    'absorption' or 'desorption'
        T_amb_K: Ambient / coolant temperature [K]

        Returns
        -------
        dict with all outputs
        """
        T   = np.asarray(T_K, dtype=float)
        P   = np.asarray(P_bar, dtype=float)
        soc = np.asarray(soc, dtype=float)

        P_eq     = self.plateau_pressure(T, mode=mode, soc=soc)
        dm_dt    = self.sorption_rate(T, P, soc, mode=mode)
        Q_rxn    = self.reaction_heat(dm_dt)
        dT_dt    = self.dTdt(T, dm_dt, T_amb_K)
        m_h2     = self.stored_mass(soc)
        grav     = self.gravimetric_density(soc)
        vol_dens = self.volumetric_density(soc)

        return {
            "plateau_pressure_bar": P_eq,
            "sorption_rate_kg_s":   dm_dt,
            "reaction_heat_W":      Q_rxn,
            "dTdt_K_s":             dT_dt,
            "stored_mass_kg":       m_h2,
            "gravimetric_wt_pct":   grav,
            "volumetric_kg_per_m3": vol_dens,
        }
