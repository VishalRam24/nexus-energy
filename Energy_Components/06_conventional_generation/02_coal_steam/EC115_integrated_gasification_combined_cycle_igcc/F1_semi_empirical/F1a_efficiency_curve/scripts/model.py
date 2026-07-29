"""
EC115 — Integrated Gasification Combined Cycle (IGCC) — F1a Efficiency Curve

System topology: Coal feed → Entrained-flow gasifier → Syngas cleanup (desulfurization,
particulates) → CCGT block (gas turbine + HRSG + steam turbine)

Net electrical efficiency:
    eta(PLR, T_amb) = eta_iso * f_PLR(PLR) * f_amb(T_amb)
    f_PLR(PLR) = a0 + a1*PLR                   (linear; IGCC has limited part-load flex)
    f_amb(T_amb) = 1 - k_amb*(T_amb - T_ref)   (gas turbine component dominates)

Derived outputs:
    P_out         = P_rated * PLR                                    [MW_e]
    coal_rate     = P_out / (eta * LHV_coal)                         [kg/s]
    syngas_rate   = coal_rate * LHV_coal * CGE / LHV_syngas          [Nm3/s]
    CO2_rate      = coal_rate * CO2_per_kg_coal                      [kg/s]
    CO2_intensity = CO2_rate / P_out * 3600                          [g/kWh] (~700-800 g/kWh)

Carbon-capture ready: IGCC can pre-combustion shift CO to H2+CO2, enabling CCS.
Without CCS:  CO2 ~700-800 g/kWh
With CCS:     CO2 ~80-120 g/kWh (not modelled here, noted in docstring)

Syngas LHV: ~10-12 MJ/Nm3 (coal; lower than natural gas ~35 MJ/Nm3)

References:
    Cormos, C.-C. (2012). Evaluation of energy integration aspects for IGCC-based
    hydrogen production with CO2 capture. Int. J. Hydrogen Energy, 37(4), 3083-3095.
    IEA GHG R&D Programme (2003). Potential for improvements in gasification combined
    cycle power generation with CO2 capture.
    Booras, G. & Holt, N. (2004). Pulverized coal and IGCC plant cost and performance
    estimates. Gasification Technologies Conference, EPRI.
"""

import numpy as np


class IGCCF1a:
    """IGCC — efficiency curve semi-empirical model (coal-to-electricity, no CCS)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated            = u["rated_power_mw"]["value"]              # MW_e
        self.eta_iso            = u["eta_iso"]["value"]                     # dimensionless
        self.T_amb_ref          = u["T_amb_ref"]["value"]                   # degC
        self.k_amb              = u["k_amb"]["value"]                       # 1/K
        self.a0                 = u["plr_coeffs"]["a0"]["value"]
        self.a1                 = u["plr_coeffs"]["a1"]["value"]
        self.a2                 = u["plr_coeffs"]["a2"]["value"]
        self.LHV_coal           = u["LHV_coal"]["value"]                    # MJ/kg
        self.CO2_per_kg_coal    = u["CO2_per_kg_coal"]["value"]             # kg_CO2/kg_coal
        self.syngas_lhv         = u["syngas_lhv"]["value"]                  # MJ/Nm3
        self.cge                = u["gasifier_cold_gas_efficiency"]["value"] # dimensionless
        self.min_plr            = u["min_plr"]["value"]

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def f_plr(self, plr):
        """Part-load efficiency correction factor (linear for IGCC)."""
        plr = np.asarray(plr, dtype=float)
        return self.a0 + self.a1 * plr + self.a2 * plr ** 2

    def f_amb(self, T_amb):
        """Linear ambient-temperature efficiency derating (gas turbine component)."""
        T_amb = np.asarray(T_amb, dtype=float)
        return 1.0 - self.k_amb * (T_amb - self.T_amb_ref)

    # ------------------------------------------------------------------
    # Primary outputs
    # ------------------------------------------------------------------

    def efficiency(self, plr, T_amb):
        """Net LHV efficiency (dimensionless)."""
        return self.eta_iso * self.f_plr(plr) * self.f_amb(T_amb)

    def power_mw(self, plr):
        """Electrical output power [MW_e]."""
        return self.P_rated * np.asarray(plr, dtype=float)

    def coal_rate_kgs(self, plr, T_amb):
        """Coal mass flow rate to gasifier [kg/s]."""
        P_out    = self.power_mw(plr)
        eta      = self.efficiency(plr, T_amb)
        eta_safe = np.where(np.asarray(eta) > 1e-6, eta, 1e-6)
        fuel_mw  = P_out / eta_safe                      # MW_th (coal basis)
        return fuel_mw / self.LHV_coal                   # kg/s

    def syngas_rate_nm3s(self, plr, T_amb):
        """
        Syngas volumetric flow rate [Nm3/s] delivered to CCGT block.
        Syngas energy = coal energy * cold-gas efficiency.
        """
        coal_kgs   = self.coal_rate_kgs(plr, T_amb)
        coal_mw    = coal_kgs * self.LHV_coal            # MW_th (coal)
        syngas_mw  = coal_mw * self.cge                  # MW_th (syngas after gasifier)
        # Nm3/s = MW_th / (MJ/Nm3)  = (MJ/s) / (MJ/Nm3) = Nm3/s  ✓
        return syngas_mw / self.syngas_lhv

    def co2_rate_kgs(self, plr, T_amb):
        """CO2 emission rate [kg/s] (without CCS)."""
        return self.coal_rate_kgs(plr, T_amb) * self.CO2_per_kg_coal

    def co2_intensity_g_per_kwh(self, plr, T_amb):
        """
        CO2 emission intensity [g_CO2/kWh_e] without CCS.
        Typical: 700-800 g/kWh for IGCC (lower than subcritical, comparable to SC/USC).
        """
        P_out_kw = self.power_mw(plr) * 1e3
        co2_gs   = self.co2_rate_kgs(plr, T_amb) * 1e3   # g/s
        P_safe   = np.where(np.asarray(P_out_kw) > 1e-6, P_out_kw, 1e-6)
        return co2_gs / P_safe * 3600.0
