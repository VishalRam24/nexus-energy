"""
EC143 -- Biomass Gasifier -- F2a Chemical Equilibrium Model

Physics-lumped equilibrium model for biomass gasification.

Overall reaction:
    CH_x O_y N_z + w*H2O + m*(O2 + 3.76*N2) -> n1*CO + n2*CO2 + n3*H2 + n4*H2O + n5*CH4 + n6*N2

Equilibrium reactions with temperature-dependent equilibrium constants:
    Water-gas shift: CO + H2O <-> CO2 + H2
        K_wgs(T) = exp(4577.8/T - 4.33)
    Methanation: C + 2H2 <-> CH4
        K_meth(T) = exp(7082/T - 7.466 + 0.372*ln(T))

Constraints:
    - Atom balances: C, H, O, N
    - Equilibrium equations (WGS, methanation)
    - Total mole fraction = 1

References:
    Zainal et al. (2001) Energy Conversion & Management, 42, 1499-1515
    Li et al. (2004) Biomass & Bioenergy, 26, 171-193
    Jarungthammachote & Dutta (2007) Energy Conversion & Management, 48, 2718-2731
"""

import numpy as np
from scipy.optimize import fsolve

# Molecular weights [g/mol]
MW_C = 12.011
MW_H = 1.008
MW_O = 15.999
MW_N = 14.007
MW_H2O = 18.015
MW_CO = 28.010
MW_CO2 = 44.009
MW_H2 = 2.016
MW_CH4 = 16.043
MW_N2 = 28.014
MW_O2 = 31.998


class BiomassGasifier_F2a:
    """Biomass Gasifier -- Chemical equilibrium model."""

    R = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.C_mass = u["biomass_C"]["value"]
        self.H_mass = u["biomass_H"]["value"]
        self.O_mass = u["biomass_O"]["value"]
        self.N_mass = u["biomass_N"]["value"]
        self.moisture = u["moisture_content"]["value"]
        self.ER = u["equivalence_ratio"]["value"]
        self.T_gasif = u["gasification_temperature"]["value"]
        self.P = u["pressure"]["value"]
        self.feed_rate = u["biomass_feed_rate"]["value"]
        self.LHV_CO = u["LHV_CO"]["value"]
        self.LHV_H2 = u["LHV_H2"]["value"]
        self.LHV_CH4 = u["LHV_CH4"]["value"]

    # ------------------------------------------------------------------
    # Biomass formula on a per-mole-C basis: CH_x O_y N_z
    # ------------------------------------------------------------------
    def biomass_formula(self, C_mass=None, H_mass=None, O_mass=None, N_mass=None):
        """Convert mass fractions to molar formula per mole of C."""
        C = (C_mass if C_mass is not None else self.C_mass) / MW_C
        H = (H_mass if H_mass is not None else self.H_mass) / MW_H
        O = (O_mass if O_mass is not None else self.O_mass) / MW_O
        N = (N_mass if N_mass is not None else self.N_mass) / MW_N
        # Normalise to 1 mole C
        x = H / C  # H atoms per C atom
        y = O / C
        z = N / C
        return x, y, z

    # ------------------------------------------------------------------
    # Equilibrium constants
    # ------------------------------------------------------------------
    @staticmethod
    def K_wgs(T):
        """Water-gas shift equilibrium constant: CO + H2O <-> CO2 + H2."""
        return np.exp(4577.8 / T - 4.33)

    @staticmethod
    def K_meth(T):
        """Methanation equilibrium constant: C + 2H2 <-> CH4."""
        return np.exp(7082.0 / T - 7.466 + 0.372 * np.log(T))

    # ------------------------------------------------------------------
    # Stoichiometric air requirement
    # ------------------------------------------------------------------
    def stoichiometric_air(self, x, y, z):
        """Moles of O2 needed for complete combustion of CH_x O_y N_z per mole C."""
        # CH_x O_y + (1 + x/4 - y/2)*O2 -> CO2 + (x/2)*H2O
        return 1.0 + x / 4.0 - y / 2.0

    # ------------------------------------------------------------------
    # Solve equilibrium composition
    # ------------------------------------------------------------------
    def solve_equilibrium(self, T=None, ER=None, moisture=None,
                          C_mass=None, H_mass=None, O_mass=None, N_mass=None):
        """
        Solve nonlinear system for equilibrium gasification composition.

        Parameters
        ----------
        T : float, optional
            Gasification temperature [K]
        ER : float, optional
            Equivalence ratio
        moisture : float, optional
            Moisture content (wet basis)
        C_mass, H_mass, O_mass, N_mass : float, optional
            Biomass ultimate analysis mass fractions

        Returns
        -------
        dict with syngas composition, LHV, efficiencies, etc.
        """
        T = T if T is not None else self.T_gasif
        ER = ER if ER is not None else self.ER
        moisture = moisture if moisture is not None else self.moisture

        x, y, z = self.biomass_formula(C_mass, H_mass, O_mass, N_mass)

        # Moles of moisture per mole C in dry biomass
        # moisture = mass_water / (mass_water + mass_dry_biomass)
        # Per mole C: dry biomass MW ~ 12 + x*1 + y*16 + z*14
        MW_dry_biomass = MW_C + x * MW_H + y * MW_O + z * MW_N
        w = (moisture / (1.0 - moisture)) * MW_dry_biomass / MW_H2O

        # Stoichiometric O2 and actual air
        O2_stoich = self.stoichiometric_air(x, y, z)
        m = ER * O2_stoich  # moles O2 supplied

        # Unknowns: n1=CO, n2=CO2, n3=H2, n4=H2O, n5=CH4
        # N2 is determined from nitrogen balance: n6 = z/2 + 3.76*m

        n6 = z / 2.0 + 3.76 * m  # N2 moles (fixed from balance)

        P_atm = self.P / 101325.0

        def equations(vars):
            n1, n2, n3, n4, n5 = vars
            # Ensure positivity by working with absolute values in residuals
            n_total = n1 + n2 + n3 + n4 + n5 + n6

            # Carbon balance: 1 = n1 + n2 + n5
            eq1 = n1 + n2 + n5 - 1.0

            # Hydrogen balance: x + 2*w = 2*n3 + 2*n4 + 4*n5
            eq2 = 2.0 * n3 + 2.0 * n4 + 4.0 * n5 - (x + 2.0 * w)

            # Oxygen balance: y + w + 2*m = n1 + 2*n2 + n4
            eq3 = n1 + 2.0 * n2 + n4 - (y + w + 2.0 * m)

            # WGS equilibrium: K_wgs = (n2 * n3) / (n1 * n4)
            K_w = self.K_wgs(T)
            eq4 = n2 * n3 - K_w * n1 * n4

            # Methanation equilibrium: K_meth = (n5/n_total) / (n3/n_total)^2 * P_atm
            # K_meth = x_CH4 / (x_H2^2 * P)  where x_i = n_i/n_total, P in atm
            K_m = self.K_meth(T)
            eq5 = n5 * n_total - K_m * n3**2 * P_atm

            return [eq1, eq2, eq3, eq4, eq5]

        # Initial guess
        n0 = [0.20, 0.10, 0.15, 0.10, 0.02]

        sol = fsolve(equations, n0, full_output=True)
        n_sol = sol[0]
        info = sol[1]

        n1, n2, n3, n4, n5 = np.maximum(n_sol, 0.0)
        n_total = n1 + n2 + n3 + n4 + n5 + n6

        # Mole fractions (dry basis -- exclude H2O)
        n_dry = n1 + n2 + n3 + n5 + n6
        if n_dry < 1e-10:
            n_dry = 1e-10

        x_CO = n1 / n_dry
        x_CO2 = n2 / n_dry
        x_H2 = n3 / n_dry
        x_CH4 = n5 / n_dry
        x_N2 = n6 / n_dry

        # Wet-basis mole fractions
        x_CO_wet = n1 / n_total
        x_CO2_wet = n2 / n_total
        x_H2_wet = n3 / n_total
        x_H2O_wet = n4 / n_total
        x_CH4_wet = n5 / n_total
        x_N2_wet = n6 / n_total

        # LHV of syngas [MJ/Nm3]
        LHV_syngas = (x_CO * self.LHV_CO + x_H2 * self.LHV_H2 +
                       x_CH4 * self.LHV_CH4)

        # Biomass HHV estimation (Channiwala & Parikh correlation) [MJ/kg]
        C_pct = (C_mass if C_mass is not None else self.C_mass) * 100
        H_pct = (H_mass if H_mass is not None else self.H_mass) * 100
        O_pct = (O_mass if O_mass is not None else self.O_mass) * 100
        N_pct = (N_mass if N_mass is not None else self.N_mass) * 100
        HHV_biomass = (0.3491 * C_pct + 1.1783 * H_pct - 0.1034 * O_pct -
                        0.0151 * N_pct + 0.1005 * 0.0 - 0.0211 * 1.0)  # MJ/kg daf

        # Gas yield [Nm3/kg biomass] -- from stoichiometry
        # n_dry moles gas per mole C in biomass
        # 1 mol C in dry biomass = MW_dry_biomass grams
        gas_yield = n_dry * 22.414e-3 / (MW_dry_biomass * 1e-3)  # Nm3/kg

        # Cold gas efficiency
        CGE = (LHV_syngas * gas_yield) / HHV_biomass if HHV_biomass > 0 else 0.0

        # H2/CO ratio
        H2_CO_ratio = n3 / n1 if n1 > 1e-10 else float('inf')

        return {
            "composition_dry_mol_pct": {
                "CO": x_CO * 100,
                "CO2": x_CO2 * 100,
                "H2": x_H2 * 100,
                "CH4": x_CH4 * 100,
                "N2": x_N2 * 100,
            },
            "composition_wet_mol_pct": {
                "CO": x_CO_wet * 100,
                "CO2": x_CO2_wet * 100,
                "H2": x_H2_wet * 100,
                "H2O": x_H2O_wet * 100,
                "CH4": x_CH4_wet * 100,
                "N2": x_N2_wet * 100,
            },
            "LHV_syngas_MJ_Nm3": LHV_syngas,
            "HHV_biomass_MJ_kg": HHV_biomass,
            "gas_yield_Nm3_per_kg": gas_yield,
            "cold_gas_efficiency": CGE,
            "H2_CO_ratio": H2_CO_ratio,
            "temperature_K": T,
            "equivalence_ratio": ER,
            "moisture_content": moisture,
            "moles_per_C": {
                "CO": n1, "CO2": n2, "H2": n3, "H2O": n4, "CH4": n5, "N2": n6,
            },
        }

    # ------------------------------------------------------------------
    # Temperature sweep
    # ------------------------------------------------------------------
    def temperature_sweep(self, T_range=None, ER=None, moisture=None):
        """Compute equilibrium composition across temperature range."""
        if T_range is None:
            T_range = np.linspace(973.15, 1373.15, 50)
        results = []
        for T in T_range:
            r = self.solve_equilibrium(T=T, ER=ER, moisture=moisture)
            results.append(r)
        return T_range, results

    # ------------------------------------------------------------------
    # ER sweep
    # ------------------------------------------------------------------
    def er_sweep(self, ER_range=None, T=None, moisture=None):
        """Compute equilibrium composition across ER range."""
        if ER_range is None:
            ER_range = np.linspace(0.15, 0.50, 50)
        results = []
        for ER in ER_range:
            r = self.solve_equilibrium(T=T, ER=ER, moisture=moisture)
            results.append(r)
        return ER_range, results
