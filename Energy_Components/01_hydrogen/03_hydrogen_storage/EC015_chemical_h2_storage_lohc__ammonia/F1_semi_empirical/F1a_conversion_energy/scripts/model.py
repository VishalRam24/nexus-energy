"""
EC015 — Chemical H2 Storage (LOHC / Ammonia) — F1a Conversion Energy Model

LOHC (Dibenzyltoluene, DBT):
    Hydrogenation:    H0-DBT + 9 H2  →  H18-DBT     ΔH = -65 kJ/mol_H2 (exothermic)
    Dehydrogenation:  H18-DBT        →  H0-DBT + 9H2  ΔH = +65 kJ/mol_H2 (endothermic)

    Gravimetric capacity: 6.2 wt% H2

Ammonia (NH3):
    Synthesis:  N2 + 3H2 → 2NH3  ΔH = -46 kJ/mol_H2 (exothermic per H2)
    Cracking:   2NH3 → N2 + 3H2  ΔH = +46 kJ/mol_H2 (endothermic per H2)

    Gravimetric capacity: 17.6 wt% H2

Energy demand per kg H2 released (dehydrogenation/cracking):
    Q_thermal = ΔH_reaction / M_H2   [J/kg_H2]
    Accounting for reactor efficiency: Q_supplied = Q_thermal / eta

References:
    Preuster et al. (2017). Acc. Chem. Res. 50(1), 74-85.
    Niermann et al. (2021). Energy Environ. Sci. 14, 1928.
    Lamb et al. (2019). Int. J. Hydrogen Energy 44(7), 3580-3593.
"""

import numpy as np

M_H2 = 0.002016  # kg/mol


class ChemicalH2StorageF1a:
    """Chemical H2 storage conversion energy model for LOHC (DBT) and Ammonia."""

    def __init__(self, params: dict):
        lohc = params["lohc_dbt"]
        nh3 = params["ammonia"]
        thermo = params["thermodynamics"]

        # LOHC parameters
        self.lohc_cap_wt = lohc["gravimetric_capacity_wt_pct"]["value"]      # wt%
        self.lohc_dH_hydro = lohc["delta_H_hydrogenation"]["value"]           # J/mol_H2 (negative)
        self.lohc_dH_dehydro = lohc["delta_H_dehydrogenation"]["value"]       # J/mol_H2 (positive)
        self.lohc_T_hydro = lohc["T_reactor_hydrogenation_K"]["value"]        # K
        self.lohc_T_dehydro = lohc["T_reactor_dehydrogenation_K"]["value"]    # K
        self.lohc_eta_dehydro = lohc["energy_efficiency_dehydrogenation"]["value"]
        self.lohc_M_carrier = lohc["molar_mass_carrier_kg_per_mol"]["value"]  # kg/mol H0-DBT
        self.lohc_H2_per_mol = lohc["H2_mol_per_mol_carrier"]["value"]        # mol H2 / mol carrier

        # NH3 parameters
        self.nh3_cap_wt = nh3["gravimetric_capacity_wt_pct"]["value"]         # wt%
        self.nh3_dH_synth = nh3["delta_H_synthesis"]["value"]                  # J/mol_H2 (negative)
        self.nh3_dH_crack = nh3["delta_H_cracking"]["value"]                   # J/mol_H2 (positive)
        self.nh3_T_synth = nh3["T_reactor_synthesis_K"]["value"]               # K
        self.nh3_T_crack = nh3["T_reactor_cracking_K"]["value"]                # K
        self.nh3_eta_crack = nh3["energy_efficiency_cracking"]["value"]
        self.nh3_M = nh3["molar_mass_nh3_kg_per_mol"]["value"]                 # kg/mol
        self.nh3_H2_per_mol = nh3["H2_mol_per_mol_carrier"]["value"]           # mol H2 / mol NH3

        self.LHV_H2 = thermo["LHV_H2"]["value"]  # J/kg_H2

    # ─────────────────────────────────────────────
    # LOHC methods
    # ─────────────────────────────────────────────

    def lohc_carrier_mass(self, m_H2_kg):
        """
        Mass of LOHC carrier (H0-DBT) required to carry m_H2_kg of H2 [kg].

        Gravimetric capacity = m_H2 / (m_H2 + m_carrier) => m_carrier = m_H2*(1-wt)/wt
        """
        m = np.asarray(m_H2_kg, dtype=float)
        cap = self.lohc_cap_wt / 100.0
        return m * (1.0 - cap) / cap

    def lohc_thermal_energy(self, m_H2_kg, direction="dehydrogenation"):
        """
        Thermal energy demand [MJ] for LOHC hydrogenation or dehydrogenation.

        Args:
            m_H2_kg:   Mass of H2 involved [kg]
            direction: 'hydrogenation' (charging) or 'dehydrogenation' (discharging)

        Returns:
            Q_thermal [MJ]; positive = heat required (dehydrogenation)
                            negative = heat released (hydrogenation, exothermic)
        """
        m = np.asarray(m_H2_kg, dtype=float)
        n_H2 = m / M_H2   # mol H2

        if direction == "dehydrogenation":
            Q_J = n_H2 * self.lohc_dH_dehydro / self.lohc_eta_dehydro
        else:  # hydrogenation
            Q_J = n_H2 * self.lohc_dH_hydro  # negative (heat released)

        return Q_J / 1e6  # MJ

    def lohc_specific_energy(self, direction="dehydrogenation"):
        """
        Specific thermal energy demand [MJ/kg_H2] for LOHC reaction.
        """
        Q_per_mol = (self.lohc_dH_dehydro / self.lohc_eta_dehydro
                     if direction == "dehydrogenation"
                     else self.lohc_dH_hydro)
        Q_per_kg = Q_per_mol / M_H2  # J/kg_H2
        return Q_per_kg / 1e6  # MJ/kg_H2

    def lohc_reactor_temperature(self, direction="dehydrogenation"):
        """Reactor temperature [K] for LOHC reaction."""
        if direction == "dehydrogenation":
            return self.lohc_T_dehydro
        return self.lohc_T_hydro

    def lohc_roundtrip_efficiency(self):
        """
        Round-trip energy efficiency: ratio of H2 LHV out to H2 LHV in + thermal input.
        Simplified: eta_rt = LHV_H2 / (LHV_H2 + |Q_dehydro_per_kg|)
        """
        q_dehydro = abs(self.lohc_dH_dehydro) / (M_H2 * self.lohc_eta_dehydro)  # J/kg_H2
        return self.LHV_H2 / (self.LHV_H2 + q_dehydro)

    # ─────────────────────────────────────────────
    # Ammonia methods
    # ─────────────────────────────────────────────

    def nh3_carrier_mass(self, m_H2_kg):
        """Mass of NH3 required to store m_H2_kg of H2 [kg]."""
        m = np.asarray(m_H2_kg, dtype=float)
        cap = self.nh3_cap_wt / 100.0
        return m * (1.0 - cap) / cap

    def nh3_thermal_energy(self, m_H2_kg, direction="cracking"):
        """
        Thermal energy demand [MJ] for NH3 synthesis or cracking.

        Args:
            m_H2_kg:   Mass of H2 involved [kg]
            direction: 'synthesis' (charging) or 'cracking' (discharging)

        Returns:
            Q_thermal [MJ]; positive = heat required (cracking)
                            negative = heat released (synthesis, exothermic)
        """
        m = np.asarray(m_H2_kg, dtype=float)
        n_H2 = m / M_H2  # mol H2

        if direction == "cracking":
            Q_J = n_H2 * self.nh3_dH_crack / self.nh3_eta_crack
        else:  # synthesis
            Q_J = n_H2 * self.nh3_dH_synth  # negative

        return Q_J / 1e6  # MJ

    def nh3_specific_energy(self, direction="cracking"):
        """Specific thermal energy demand [MJ/kg_H2] for NH3 reaction."""
        Q_per_mol = (self.nh3_dH_crack / self.nh3_eta_crack
                     if direction == "cracking"
                     else self.nh3_dH_synth)
        Q_per_kg = Q_per_mol / M_H2
        return Q_per_kg / 1e6  # MJ/kg_H2

    def nh3_reactor_temperature(self, direction="cracking"):
        """Reactor temperature [K] for NH3 reaction."""
        if direction == "cracking":
            return self.nh3_T_crack
        return self.nh3_T_synth

    def nh3_roundtrip_efficiency(self):
        """Round-trip energy efficiency for NH3 storage."""
        q_crack = abs(self.nh3_dH_crack) / (M_H2 * self.nh3_eta_crack)
        return self.LHV_H2 / (self.LHV_H2 + q_crack)
