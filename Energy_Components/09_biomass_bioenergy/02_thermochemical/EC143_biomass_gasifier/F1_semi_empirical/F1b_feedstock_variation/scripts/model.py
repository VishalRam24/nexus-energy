"""
EC143 -- Biomass Gasifier -- F1b Feedstock Variation Model

Builds on F1a (single equilibrium model) by adding:
  - Feedstock-specific ultimate analysis (C, H, O, N, S, ash mass fractions)
  - Equivalence ratio (ER) effect on syngas composition via empirical correlations
  - Moisture content correction (higher moisture -> lower gasification efficiency)
  - Tar content estimation (higher ER -> lower tar)
  - Feedstock-specific HHV for cold gas efficiency calculation

Empirical correlations for syngas composition (dry basis, vol%):
  CO  = a_CO * C_daf / (0.5) * (1 - ER/0.6) * moisture_factor
  H2  = a_H2 * (H_daf + 0.5*moisture) * (1 - ER/0.7)
  CO2 = a_CO2 * C_daf * ER
  CH4 = a_CH4 * H_daf * (1 - ER/0.5)
  N2  = balance (from air)

where daf = dry ash-free basis.

References:
    Zainal, Z.A. et al. (2001). Energy Conversion and Management, 42(12), 1499-1515.
    Basu, P. (2010). Biomass Gasification and Pyrolysis. Academic Press.
    Li, X.T. et al. (2004). Biomass & Bioenergy, 26(2), 171-193.
"""

import numpy as np


class BiomassGasifierF1b:
    """Biomass gasifier with feedstock-specific syngas prediction."""

    def __init__(self, params: dict):
        self.feedstock_db = params["feedstock_db"]
        u = params["unit"]
        self.T_gasif = u["T_gasification_degC"]["value"]
        self.ER_design = u["ER_design"]["value"]
        self.LHV_CO = u["LHV_CO_MJ_Nm3"]["value"]
        self.LHV_H2 = u["LHV_H2_MJ_Nm3"]["value"]
        self.LHV_CH4 = u["LHV_CH4_MJ_Nm3"]["value"]

    def _get_feedstock(self, feedstock_type):
        """Get feedstock data, raise if unknown."""
        if feedstock_type not in self.feedstock_db:
            raise ValueError(f"Unknown feedstock: {feedstock_type}. "
                             f"Available: {list(self.feedstock_db.keys())}")
        return self.feedstock_db[feedstock_type]

    def syngas_composition(self, feedstock_type, equivalence_ratio,
                           moisture_content=0.0):
        """
        Compute dry syngas composition from feedstock analysis and ER.

        Uses empirical correlations calibrated against published gasification data.
        """
        fs = self._get_feedstock(feedstock_type)
        ER = np.clip(float(equivalence_ratio), 0.15, 0.45)
        MC = np.clip(float(moisture_content), 0.0, 0.5)

        # Dry ash-free basis
        ash = fs["ash"]
        daf_factor = 1.0 - ash - MC  # fraction of dry ash-free matter
        daf_factor = max(daf_factor, 0.1)

        C = fs["C"] / (1 - ash) if (1 - ash) > 0 else fs["C"]
        H = fs["H"] / (1 - ash) if (1 - ash) > 0 else fs["H"]
        O = fs["O"] / (1 - ash) if (1 - ash) > 0 else fs["O"]

        # Moisture correction: higher moisture reduces gasification temperature
        # and shifts equilibrium toward H2 and CO2
        moisture_factor_CO = max(0.3, 1.0 - 0.8 * MC)
        moisture_factor_H2 = 1.0 + 0.3 * MC  # steam reforming increases H2

        # Empirical composition correlations (volume fractions)
        # Higher ER -> more oxidation -> more CO2, less CO
        CO = 0.44 * C * (1.0 - ER / 0.6) * moisture_factor_CO
        CO = max(0.0, CO)

        H2 = 0.30 * (H * 10 + 0.5 * MC) * (1.0 - ER / 0.7) * moisture_factor_H2
        H2 = max(0.0, min(H2, 0.30))

        CO2 = 0.20 * C * (ER / 0.25)
        CO2 = max(0.0, min(CO2, 0.25))

        CH4 = 0.08 * H * 10 * max(0.0, 1.0 - ER / 0.5) * (1.0 - MC)
        CH4 = max(0.0, min(CH4, 0.08))

        # N2 as balance (from air gasification agent)
        combustibles = CO + H2 + CO2 + CH4
        N2 = max(0.0, 1.0 - combustibles)

        # Normalize
        total = CO + H2 + CO2 + CH4 + N2
        if total > 0:
            CO /= total
            H2 /= total
            CO2 /= total
            CH4 /= total
            N2 /= total

        return {"CO": CO, "H2": H2, "CO2": CO2, "CH4": CH4, "N2": N2}

    def lhv_syngas(self, composition):
        """LHV of syngas [MJ/Nm3] from composition."""
        return (self.LHV_CO * composition["CO"] +
                self.LHV_H2 * composition["H2"] +
                self.LHV_CH4 * composition["CH4"])

    def syngas_yield(self, feedstock_type, equivalence_ratio, moisture_content):
        """
        Specific syngas yield [Nm3/kg dry biomass].
        Based on stoichiometry: more air (higher ER) -> more syngas volume but diluted.
        """
        fs = self._get_feedstock(feedstock_type)
        ER = np.clip(float(equivalence_ratio), 0.15, 0.45)
        MC = np.clip(float(moisture_content), 0.0, 0.5)

        # Base yield scaled by carbon content and ER
        C = fs["C"]
        V_base = 1.5 + 3.0 * C  # Nm3/kg for typical biomass ~2.0-3.0
        V_syngas = V_base * (ER / 0.25) ** 0.4 * (1.0 - 0.5 * MC)

        return max(0.5, V_syngas)

    def tar_content(self, equivalence_ratio):
        """
        Tar content in syngas [g/Nm3].
        Decreases with ER (more oxidation breaks down tars).
        Typical: 10-100 g/Nm3 for updraft, 0.5-5 for downdraft.
        """
        ER = np.clip(float(equivalence_ratio), 0.15, 0.45)
        # Exponential decay with ER (downdraft gasifier)
        tar = 50.0 * np.exp(-6.0 * ER)
        return float(tar)

    def cold_gas_efficiency(self, feedstock_type, equivalence_ratio,
                            moisture_content, composition=None):
        """
        Cold gas efficiency = (LHV_syngas * V_syngas) / (HHV_biomass * m_biomass).
        """
        fs = self._get_feedstock(feedstock_type)
        if composition is None:
            composition = self.syngas_composition(feedstock_type, equivalence_ratio,
                                                  moisture_content)

        lhv = self.lhv_syngas(composition)
        V = self.syngas_yield(feedstock_type, equivalence_ratio, moisture_content)
        HHV = fs["HHV_MJ_kg"]

        cge = (lhv * V) / HHV if HHV > 0 else 0.0
        return np.clip(cge, 0.0, 0.95)

    def predict(self, feedstock_type, equivalence_ratio,
                moisture_content=0.1, feed_rate_kg_h=100.0):
        """
        Compute gasification outputs.

        Args:
            feedstock_type:      str, name from feedstock_db
            equivalence_ratio:   float [0.15-0.45]
            moisture_content:    float [0-0.5], wet-basis moisture fraction
            feed_rate_kg_h:      float, biomass feed rate [kg/h]

        Returns:
            dict with:
                syngas_composition : dict of mole fractions
                syngas_yield_nm3_kg : specific syngas yield [Nm3/kg]
                cold_gas_efficiency : CGE [-]
                tar_content_g_nm3   : tar in syngas [g/Nm3]
                lhv_syngas_mj_nm3   : syngas LHV [MJ/Nm3]
        """
        comp = self.syngas_composition(feedstock_type, equivalence_ratio,
                                       moisture_content)
        V = self.syngas_yield(feedstock_type, equivalence_ratio, moisture_content)
        lhv = self.lhv_syngas(comp)
        cge = self.cold_gas_efficiency(feedstock_type, equivalence_ratio,
                                       moisture_content, comp)
        tar = self.tar_content(equivalence_ratio)

        return {
            "syngas_composition": {k: float(v) for k, v in comp.items()},
            "syngas_yield_nm3_kg": float(V),
            "cold_gas_efficiency": float(cge),
            "tar_content_g_nm3": float(tar),
            "lhv_syngas_mj_nm3": float(lhv),
        }
