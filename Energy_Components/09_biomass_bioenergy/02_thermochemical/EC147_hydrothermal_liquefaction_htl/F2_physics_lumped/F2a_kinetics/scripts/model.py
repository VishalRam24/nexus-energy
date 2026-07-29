"""
EC147 -- Hydrothermal Liquefaction (HTL) -- F2a Physics-Lumped Kinetics

Physics-lumped HTL reactor model. Wet biomass is liquefied in hot compressed
(subcritical) water (~300-350 C, 15-20 MPa). The model couples:

  1. A LUMPED REACTION NETWORK (Valdez & Savage 2013 lumped-kinetics framework).
     Solid biomass B converts in parallel to four product lumps with first-order
     Arrhenius kinetics, and the desired biocrude lump C undergoes SECONDARY
     decomposition. This secondary loss is what makes biocrude yield peak at an
     intermediate severity (residence time x temperature), as observed
     experimentally (Valdez & Savage 2013; Faeth et al. 2013).

         B --k_BC--> C   (biocrude, primary liquefaction)
         B --k_BA--> A   (aqueous / water-soluble organics)
         B --k_BG--> G   (gas, mainly CO2 via decarboxylation)
         B --k_BS--> S   (solid residue / char)
         C --k_CG--> G   (secondary cracking of biocrude to gas)
         C --k_CA--> A   (secondary repolymerisation / dissolution)

     Each rate: k_i(T) = A_i * exp(-Ea_i / (R T))   (Arrhenius).
     Mass fractions y = [B, C, A, G, S] satisfy sum(y) = 1 exactly at all times
     (the ODE right-hand side is constructed to be mass-conserving: every term
     that leaves one lump enters another).

  2. A STIRRED-REACTOR ENERGY BALANCE for the slurry temperature:

         m cp dT/dt = UA (T_set - T)  -  dH_rxn * dm_B/dt

     where the heater drives T toward the subcritical setpoint and the
     (mildly exothermic) liquefaction enthalpy is released as biomass is
     consumed. Temperature feeds back into every Arrhenius rate, so the
     composition and energy ODEs are fully coupled and integrated together
     with scipy.integrate.solve_ivp.

Subcritical-water check: HTL keeps water as a compressed LIQUID below its
critical point (Tc = 374 C, Pc = 22.06 MPa). The operating pressure must
exceed the saturation pressure at the operating temperature so the water does
not boil. We expose is_subcritical() / water_liquid() helpers for this.

References:
    Toor, S.S., Rosendahl, L., Rudolf, A. (2011). Hydrothermal liquefaction of
        biomass: A review of subcritical water technologies. Energy 36:2328.
    Valdez, P.J., Savage, P.E. (2013). A reaction network for the hydrothermal
        liquefaction of Nannochloropsis sp. Algal Research 2:416-425.
    Valdez, P.J., Tocco, V.J., Savage, P.E. (2014). A general kinetic model for
        the HTL of microalgae. Bioresour. Technol. 163:123-127.
    Sheehan, J.D., Savage, P.E. (2017). Modeling the lipid extraction / HTL
        kinetics. Bioresour. Technol. 239:144.
    Peterson, A.A. et al. (2008). Thermochemical biofuel production in
        hydrothermal media. Energy Environ. Sci. 1:32 (water properties/energetics).
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314462618  # J/(mol.K)

# Antoine-type saturation pressure of water (MPa), valid ~250-374 C.
# Fitted to IAPWS saturation data; used only for the subcritical-water guard.
_TC_WATER_C = 374.0
_PC_WATER_MPA = 22.06


def water_saturation_pressure_MPa(T_C: float) -> float:
    """Saturation pressure of water [MPa] from a Wagner-style correlation.

    Uses the IAPWS critical point and a reduced-temperature fit accurate to a
    few percent over 200-374 C, which is all we need for the subcritical guard.
    """
    T_K = T_C + 273.15
    Tc_K = _TC_WATER_C + 273.15
    if T_K >= Tc_K:
        return _PC_WATER_MPA
    tau = 1.0 - T_K / Tc_K
    # Wagner equation coefficients for water (IAPWS-IF97 region 4 fit)
    a1, a2, a3, a4 = -7.85951783, 1.84408259, -11.7866497, 22.6807411
    a5, a6 = -15.9618719, 1.80122502
    ln_pr = (Tc_K / T_K) * (
        a1 * tau + a2 * tau**1.5 + a3 * tau**3 + a4 * tau**3.5
        + a5 * tau**4 + a6 * tau**7.5
    )
    return _PC_WATER_MPA * np.exp(ln_pr)


class HTL_F2a:
    """Lumped HTL reaction-network kinetics coupled to a reactor energy balance.

    State vector for the ODE:  s = [B, C, A, G, S, T]
        B, C, A, G, S : mass fractions of biomass / biocrude / aqueous / gas / solid
        T             : reactor (slurry) temperature [K]
    """

    LUMP_NAMES = ("biomass", "biocrude", "aqueous", "gas", "solid")

    def __init__(self, params: dict):
        u = params["unit"]

        def g(k):
            return float(u[k]["value"])

        # Arrhenius pre-exponentials [1/s] and activation energies [J/mol]
        self.A_BC, self.Ea_BC = g("A_BC"), g("Ea_BC")
        self.A_BA, self.Ea_BA = g("A_BA"), g("Ea_BA")
        self.A_BG, self.Ea_BG = g("A_BG"), g("Ea_BG")
        self.A_BS, self.Ea_BS = g("A_BS"), g("Ea_BS")
        self.A_CG, self.Ea_CG = g("A_CG"), g("Ea_CG")
        self.A_CA, self.Ea_CA = g("A_CA"), g("Ea_CA")

        self.dH_rxn = g("dH_rxn")        # J/kg biomass reacted (negative = exothermic)
        self.m_slurry = g("m_slurry")    # kg
        self.cp_slurry = g("cp_slurry")  # J/(kg.K)
        self.UA = g("UA_heater")         # W/K
        self.LHV_biocrude = g("LHV_biocrude")  # MJ/kg

        self.Tc_water_C = g("Tc_water_C")
        self.Pc_water_MPa = g("Pc_water_MPa")

    # ------------------------------------------------------------------ rates
    @staticmethod
    def arrhenius(A, Ea, T):
        return A * np.exp(-Ea / (R_GAS * T))

    def rate_constants(self, T):
        """Return all six first-order rate constants [1/s] at temperature T [K]."""
        return {
            "k_BC": self.arrhenius(self.A_BC, self.Ea_BC, T),
            "k_BA": self.arrhenius(self.A_BA, self.Ea_BA, T),
            "k_BG": self.arrhenius(self.A_BG, self.Ea_BG, T),
            "k_BS": self.arrhenius(self.A_BS, self.Ea_BS, T),
            "k_CG": self.arrhenius(self.A_CG, self.Ea_CG, T),
            "k_CA": self.arrhenius(self.A_CA, self.Ea_CA, T),
        }

    # ------------------------------------------------------------- subcritical
    def is_subcritical(self, T_C, P_MPa):
        """True if (T,P) keeps water a subcritical COMPRESSED LIQUID.

        Requires T < Tc, P < Pc, and P above the saturation pressure at T so the
        water stays liquid (does not flash to steam).
        """
        if T_C >= self.Tc_water_C or P_MPa >= self.Pc_water_MPa:
            return False
        return P_MPa > water_saturation_pressure_MPa(T_C)

    def water_liquid(self, T_C, P_MPa):
        return P_MPa > water_saturation_pressure_MPa(T_C)

    # -------------------------------------------------------------------- ODE
    def _rhs(self, t, s, T_set_K):
        B, C, A, G, S, T = s
        B = max(B, 0.0)
        C = max(C, 0.0)
        k = self.rate_constants(T)

        # Primary consumption of biomass (parallel pathways)
        r_BC = k["k_BC"] * B
        r_BA = k["k_BA"] * B
        r_BG = k["k_BG"] * B
        r_BS = k["k_BS"] * B
        # Secondary decomposition of biocrude
        r_CG = k["k_CG"] * C
        r_CA = k["k_CA"] * C

        dB = -(r_BC + r_BA + r_BG + r_BS)
        dC = r_BC - (r_CG + r_CA)
        dA = r_BA + r_CA
        dG = r_BG + r_CG
        dS = r_BS
        # Mass conservation: dB+dC+dA+dG+dS == 0 by construction.

        # Energy balance: heater drive + reaction enthalpy from biomass consumed.
        # dm_B/dt = dB * m_slurry (kg/s of biomass reacting). Exothermic release
        # = -dH_rxn * (rate of biomass consumed) [W].
        biomass_consumed_rate = -dB * self.m_slurry  # kg/s (>=0)
        q_rxn = -self.dH_rxn * biomass_consumed_rate  # W (dH<0 -> q_rxn>0)
        q_heat = self.UA * (T_set_K - T)
        dT = (q_heat + q_rxn) / (self.m_slurry * self.cp_slurry)

        return [dB, dC, dA, dG, dS, dT]

    def simulate(self, T_setpoint_C=350.0, residence_min=30.0, T0_C=200.0,
                 biomass0=1.0, P_MPa=18.0, n_out=200):
        """Integrate the coupled kinetics + energy balance.

        Parameters
        ----------
        T_setpoint_C : reactor setpoint (subcritical) [C]
        residence_min : total reaction/residence time [min]
        T0_C : initial slurry temperature [C]
        biomass0 : initial biomass mass fraction (rest assumed inert water, 0 products)
        P_MPa : operating pressure [MPa] (for subcritical check / reporting)
        n_out : number of output time points

        Returns
        -------
        dict with time series of each lump, temperature, conversion, yields,
        plus scalar final yields and a subcritical flag.
        """
        biomass0 = float(np.clip(biomass0, 0.0, 1.0))
        T_set_K = T_setpoint_C + 273.15
        T0_K = T0_C + 273.15
        t_end = residence_min * 60.0  # s

        s0 = [biomass0, 0.0, 0.0, 0.0, 0.0, T0_K]
        if t_end <= 0.0:
            t_eval = np.array([0.0])
            sol_y = np.array(s0).reshape(6, 1)
        else:
            t_eval = np.linspace(0.0, t_end, n_out)
            sol = solve_ivp(
                self._rhs, (0.0, t_end), s0, t_eval=t_eval,
                args=(T_set_K,), method="LSODA", rtol=1e-8, atol=1e-10,
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed: {sol.message}")
            sol_y = sol.y

        B, C, A, G, S, T = sol_y
        # Clip tiny negative numerical excursions then renormalise lumps to 1.
        lumps = np.vstack([B, C, A, G, S])
        lumps = np.clip(lumps, 0.0, None)
        tot = lumps.sum(axis=0)
        tot = np.where(tot <= 0, 1.0, tot)
        # Yields on initial-biomass basis (not renormalised) — physical product yields.
        denom = biomass0 if biomass0 > 0 else 1.0
        conversion = (biomass0 - lumps[0]) / denom  # fraction of biomass reacted

        biocrude_yield = lumps[1] / denom
        aqueous_yield = lumps[2] / denom
        gas_yield = lumps[3] / denom
        solid_yield = lumps[4] / denom

        biocrude_energy_MJ_per_kg_feed = biocrude_yield[-1] * self.LHV_biocrude

        return {
            "t_s": t_eval,
            "t_min": t_eval / 60.0,
            "biomass": lumps[0],
            "biocrude": lumps[1],
            "aqueous": lumps[2],
            "gas": lumps[3],
            "solid": lumps[4],
            "temperature_K": T,
            "temperature_C": T - 273.15,
            "conversion": conversion,
            "biocrude_yield": biocrude_yield,
            "aqueous_yield": aqueous_yield,
            "gas_yield": gas_yield,
            "solid_yield": solid_yield,
            "mass_total": lumps.sum(axis=0),  # should track biomass0 at all t
            "final": {
                "conversion": float(conversion[-1]),
                "biocrude_yield": float(biocrude_yield[-1]),
                "aqueous_yield": float(aqueous_yield[-1]),
                "gas_yield": float(gas_yield[-1]),
                "solid_yield": float(solid_yield[-1]),
                "biocrude_energy_MJ_per_kg_feed": float(biocrude_energy_MJ_per_kg_feed),
            },
            "P_MPa": P_MPa,
            "T_setpoint_C": T_setpoint_C,
            "subcritical": bool(self.is_subcritical(T_setpoint_C, P_MPa)),
            "water_liquid": bool(self.water_liquid(T_setpoint_C, P_MPa)),
        }

    def biocrude_yield_at(self, T_setpoint_C, residence_min, **kw):
        """Convenience: final biocrude yield for a (T, t) operating point."""
        r = self.simulate(T_setpoint_C=T_setpoint_C, residence_min=residence_min, **kw)
        return r["final"]["biocrude_yield"]
