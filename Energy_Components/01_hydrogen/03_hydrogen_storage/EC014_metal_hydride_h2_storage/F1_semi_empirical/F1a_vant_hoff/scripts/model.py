"""
EC014 — Metal Hydride H2 Storage — F1a van't Hoff Model

van't Hoff equation for equilibrium plateau pressure:
    ln(P_eq) = ΔH/(R*T) - ΔS/R

where:
    P_eq  = equilibrium (plateau) pressure [bar]
    ΔH    = enthalpy of hydride formation  [J/mol_H2]  (negative for absorption)
    ΔS    = entropy of hydride formation   [J/mol_H2/K] (negative; ~gas entropy)
    R     = 8.314 J/mol/K
    T     = temperature [K]

Absorbed H2 mass:
    m_H2 = soc * H_max_wt_pct/100 * bed_mass_kg   [kg]

Heat of reaction per kg H2 absorbed/desorbed:
    Q_rxn = |ΔH| / M_H2   [J/kg_H2]

Sloped plateau (optional):
    ln(P) = ln(P_eq) + slope_factor * (soc - 0.5)

Hysteresis: absorption P_abs = hysteresis_factor * P_des (in log space: +ln(hf))

References:
    Lototskyy et al. (2014). Prog. Nat. Sci. Mater. Int. 24(2), 97-116.
    Sakintuna et al. (2007). Int. J. Hydrogen Energy, 32(9), 1121-1140.
    van't Hoff (1886). Z. Phys. Chem. 1, 481.
"""

import numpy as np

R_UNIVERSAL = 8.314       # J/(mol·K)
M_H2 = 0.002016           # kg/mol


class MetalHydrideH2F1a:
    """Metal hydride H2 storage model using van't Hoff plateau pressure."""

    def __init__(self, params: dict):
        mat = params["material"]
        thermo = params["thermodynamics"]
        ht = params["heat_transfer"]

        self.delta_H = mat["delta_H"]["value"]          # J/mol_H2
        self.delta_S = mat["delta_S"]["value"]          # J/mol_H2/K
        self.H_max_wt_pct = mat["H_max_wt_pct"]["value"]   # wt%
        self.bed_mass_kg = mat["bed_mass_kg"]["value"]  # kg
        self.volumetric_capacity = mat["volumetric_capacity"]["value"]  # kg_H2/m3
        self.bed_volume_m3 = mat["bed_volume_m3"]["value"]  # m3

        self.hysteresis_factor = thermo["hysteresis_factor"]["value"]
        self.slope_factor = thermo["plateau_slope_factor"]["value"]

        self.cp_hydride = ht["cp_hydride"]["value"]     # J/kg/K

        # Max H2 mass capacity
        self.m_H2_max = self.H_max_wt_pct / 100.0 * self.bed_mass_kg  # kg

    def plateau_pressure(self, T_K, mode="desorption", soc=0.5):
        """
        Equilibrium plateau pressure from van't Hoff equation.

        Standard thermodynamic form (Sandrock 1999, Lototskyy 2014):
            ln(P_eq / P_ref) = -ΔH_des/(R*T) + ΔS_des/R
        where ΔH_des > 0 (endothermic) and ΔS_des > 0 for desorption.
        This gives d(ln_P)/dT = ΔH_des/(R*T²) > 0 (pressure rises with T). ✓

        P_ref = 1 atm = 1.01325 bar (standard reference pressure).

        Args:
            T_K:  Temperature [K]
            mode: 'absorption' or 'desorption'
            soc:  State of charge (0–1), used for slope correction

        Returns:
            P_eq [bar]
        """
        T = np.asarray(T_K, dtype=float)
        soc_arr = np.asarray(soc, dtype=float)

        # Desorption: endothermic (ΔH_des = -ΔH_absorption > 0)
        #             entropy increases (ΔS_des = -ΔS_absorption > 0)
        dH_des = -self.delta_H   # positive (J/mol_H2)
        dS_des = -self.delta_S   # positive (J/mol_H2/K)

        # Correct van't Hoff: ln(P/P_ref) = -ΔH_des/(RT) + ΔS_des/R
        # Note: d(ln_P)/dT = +ΔH_des/(RT²) > 0 → P increases with T for endothermic reaction
        P_ref_bar = 1.01325  # standard atmosphere in bar
        ln_P_over_Pref = -dH_des / (R_UNIVERSAL * T) + dS_des / R_UNIVERSAL

        ln_P = ln_P_over_Pref + np.log(P_ref_bar)

        # Sloped plateau correction
        ln_P = ln_P + self.slope_factor * (soc_arr - 0.5)

        # Hysteresis: absorption plateau is higher than desorption
        if mode == "absorption":
            ln_P = ln_P + np.log(self.hysteresis_factor)

        return np.exp(ln_P)  # bar

    def stored_mass(self, soc):
        """
        H2 mass stored [kg] given state of charge.

        Args:
            soc: State of charge (0=empty, 1=full)
        """
        soc_arr = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return soc_arr * self.m_H2_max

    def heat_of_reaction(self, delta_m_H2_kg):
        """
        Heat released (absorption) or absorbed (desorption) [kJ].

        Args:
            delta_m_H2_kg: Mass of H2 absorbed (positive) or desorbed (negative) [kg]

        Returns:
            Q_rxn [kJ]; positive = heat released to environment (absorption)
        """
        delta_m = np.asarray(delta_m_H2_kg, dtype=float)
        # Heat per mol H2: |ΔH| = |delta_H_formation|
        # Heat per kg H2: |ΔH| / M_H2
        q_per_kg = abs(self.delta_H) / M_H2  # J/kg_H2
        return delta_m * q_per_kg / 1000.0   # kJ

    def gravimetric_density(self, soc):
        """Gravimetric H2 capacity [wt%] at given SOC."""
        m_h2 = self.stored_mass(soc)
        return m_h2 / (m_h2 + self.bed_mass_kg) * 100.0

    def volumetric_density(self, soc):
        """Volumetric H2 density [kg_H2/m3] at given SOC."""
        m_h2 = self.stored_mass(soc)
        return m_h2 / self.bed_volume_m3

    def fill_fraction(self, soc):
        """Fill fraction = soc (by definition)."""
        return np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
