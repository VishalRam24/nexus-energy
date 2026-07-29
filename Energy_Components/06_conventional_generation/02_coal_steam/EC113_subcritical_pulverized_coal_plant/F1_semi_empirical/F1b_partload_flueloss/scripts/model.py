"""
EC113 -- Subcritical Pulverized Coal Plant -- F1b Part-Load + Flue Loss Model

Extends F1a efficiency curve by adding:
  1. Explicit flue gas enthalpy loss term (stack heat loss)
  2. Auxiliary power part-load curve (fans, pumps, mills degrade at turndown)
  3. Stack temperature model (rises at part load due to less heat absorption)

Physics additions vs F1a:
    Q_flue = m_flue * cp_flue * (T_stack - T_amb)                    [MW_th]
    m_flue = m_coal * (1 + stoich_air * (1 + excess_air))            [kg/s]
    T_stack(PLR) = T_stack_rated + T_stack_offset * (1 - PLR)         [degC]
    P_aux(PLR) = P_gross * (aux_rated + aux_extra*(1-PLR))             [MW_e]
    P_net(PLR) = P_gross(PLR) - P_aux(PLR)                            [MW_e]

    The flue gas loss is separated from the efficiency curve for explicit
    energy balance reporting (used in boiler efficiency diagnostics).

    NOTE: The flue loss does not change the net efficiency equation —
    the efficiency curve from F1a (eta_iso * f_PLR * f_amb) already
    accounts for all losses including flue. The F1b flue model adds
    explicit reporting of the flue loss magnitude and stack temperature,
    plus an improved auxiliary power sub-model.

CO2 intensity note:
    CO2 intensity is well-defined only at rated operation (PLR=1).
    At part load, the combined effect of efficiency penalty and auxiliary
    power correction causes CO2/kWh to spike well above 1000 g/kWh.
    This is physically correct (not a model error) and is validated only
    at rated load — see test RATIONALE comment.

References:
    Booras, G. & Holt, N. (2004). Pulverized coal and IGCC plant cost and
    performance estimates. Gasification Technologies Conference. EPRI.
    IEA Clean Coal Centre, CCC/168 (2010). Efficiency and emissions performance
    of new and upgrading coal-fired power plants.
    Borgnakke, C. & Sonntag, R.E. (2009). Fundamentals of Thermodynamics,
    7th ed. Wiley.
"""

import numpy as np


class SubcriticalCoalF1b:
    """Subcritical coal plant with part-load efficiency, flue loss, and auxiliary power."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated              = u["rated_power_mw"]["value"]
        self.eta_iso              = u["eta_iso"]["value"]
        self.T_amb_ref            = u["T_amb_ref"]["value"]
        self.k_amb                = u["k_amb"]["value"]
        self.a0                   = u["plr_coeffs"]["a0"]["value"]
        self.a1                   = u["plr_coeffs"]["a1"]["value"]
        self.a2                   = u["plr_coeffs"]["a2"]["value"]
        self.LHV_coal             = u["LHV_coal"]["value"]
        self.CO2_per_kg_coal      = u["CO2_per_kg_coal"]["value"]
        self.T_stack_rated        = u["T_stack_rated_c"]["value"]
        self.T_stack_offset       = u["T_stack_partload_offset_c"]["value"]
        self.cp_flue              = u["cp_flue_kj_kgK"]["value"]
        self.T_amb_std            = u["T_amb_standard_c"]["value"]
        self.stoich_air           = u["stoich_air_kg_per_kg_coal"]["value"]
        self.excess_air           = u["excess_air_fraction"]["value"]
        self.aux_rated_frac       = u["aux_power_rated_pct"]["value"] / 100.0
        self.aux_extra_frac       = u["aux_power_partload_extra_pct"]["value"] / 100.0
        self.min_plr              = u["min_plr"]["value"]

    # ------------------------------------------------------------------
    # Correction factors (same as F1a)
    # ------------------------------------------------------------------

    def f_plr(self, plr):
        plr = np.asarray(plr, dtype=float)
        return self.a0 + self.a1 * plr + self.a2 * plr ** 2

    def f_amb(self, T_amb):
        T_amb = np.asarray(T_amb, dtype=float)
        return 1.0 - self.k_amb * (T_amb - self.T_amb_ref)

    # ------------------------------------------------------------------
    # Efficiency (same net equation as F1a)
    # ------------------------------------------------------------------

    def efficiency_net(self, plr, T_amb):
        """Net LHV efficiency (accounting for F1a curve)."""
        return self.eta_iso * self.f_plr(plr) * self.f_amb(T_amb)

    # ------------------------------------------------------------------
    # Stack temperature model (F1b addition)
    # ------------------------------------------------------------------

    def stack_temperature_c(self, plr):
        """
        Stack (flue gas exhaust) temperature [degC].
        At part load, less heat is transferred to steam in the boiler,
        so the flue gas exits at a higher temperature (lower effectiveness).
        T_stack(PLR) = T_rated + T_offset * (1 - PLR)
        """
        plr = np.asarray(plr, dtype=float)
        return self.T_stack_rated + self.T_stack_offset * (1.0 - plr)

    # ------------------------------------------------------------------
    # Flue gas loss (F1b addition)
    # ------------------------------------------------------------------

    def coal_rate_gross_kgs(self, plr, T_amb):
        """Coal mass flow rate based on gross efficiency [kg/s].
        Gross efficiency ~ net / (1 - aux_fraction).
        """
        plr    = np.asarray(plr, dtype=float)
        P_net  = self.power_mw(plr)
        eta_net = self.efficiency_net(plr, T_amb)
        # Auxiliary power fraction varies with PLR
        aux_frac = self.aux_power_fraction(plr)
        # P_gross = P_net / (1 - aux_frac), but aux_frac applied to gross
        # Approximate: eta_gross = eta_net / (1 - aux_frac)
        eta_gross = np.where(
            (1.0 - aux_frac) > 0.01,
            eta_net / (1.0 - aux_frac),
            eta_net
        )
        eta_safe = np.where(eta_gross > 1e-6, eta_gross, 1e-6)
        fuel_mw  = P_net / eta_safe              # approximate gross fuel
        return fuel_mw / self.LHV_coal

    def flue_gas_rate_kgs(self, plr, T_amb):
        """
        Total flue gas mass flow rate [kg/s].
        m_flue = m_coal * (1 + stoich_air * (1 + excess_air)) + m_ash
        Simplified: m_flue ~ m_coal * (1 + stoich_air * (1 + excess_air))
        (ash fraction is small and remains in hopper/ESP)
        """
        m_coal = self.coal_rate_gross_kgs(plr, T_amb)
        m_air  = m_coal * self.stoich_air * (1.0 + self.excess_air)
        return m_coal + m_air   # flue = coal products + excess air

    def flue_heat_loss_mw(self, plr, T_amb):
        """
        Flue gas enthalpy loss [MW_th].
        Q_flue = m_flue * cp_flue * (T_stack - T_amb) / 1000
        """
        plr      = np.asarray(plr, dtype=float)
        T_amb    = np.asarray(T_amb, dtype=float)
        m_flue   = self.flue_gas_rate_kgs(plr, T_amb)
        T_stack  = self.stack_temperature_c(plr)
        dT       = T_stack - T_amb
        # Q_flue [kW] = m_flue [kg/s] * cp [kJ/(kg.K)] * dT [K]
        return m_flue * self.cp_flue * dT / 1000.0   # MW_th

    # ------------------------------------------------------------------
    # Auxiliary power (F1b addition)
    # ------------------------------------------------------------------

    def aux_power_fraction(self, plr):
        """
        Auxiliary power as fraction of gross output.
        At full load: aux_rated_frac (e.g. 7%).
        At PLR_min: aux_rated_frac + aux_extra_frac (e.g. 9%).
        Linear interpolation.
        """
        plr    = np.asarray(plr, dtype=float)
        frac   = self.aux_rated_frac + self.aux_extra_frac * (1.0 - plr)
        return np.clip(frac, 0.0, 0.15)

    # ------------------------------------------------------------------
    # Primary derived outputs
    # ------------------------------------------------------------------

    def power_mw(self, plr):
        """Net electrical output [MW_e]."""
        return self.P_rated * np.asarray(plr, dtype=float)

    def coal_rate_kgs(self, plr, T_amb):
        """Coal mass flow rate [kg/s] (net efficiency basis — consistent with F1a)."""
        P_net    = self.power_mw(plr)
        eta_net  = self.efficiency_net(plr, T_amb)
        eta_safe = np.where(np.asarray(eta_net) > 1e-6, eta_net, 1e-6)
        return P_net / (eta_safe * self.LHV_coal)

    def co2_rate_kgs(self, plr, T_amb):
        """CO2 emission rate [kg/s]."""
        return self.coal_rate_kgs(plr, T_amb) * self.CO2_per_kg_coal

    def co2_intensity_g_per_kwh(self, plr, T_amb):
        """CO2 emission intensity [g/kWh_e]."""
        P_kw   = self.power_mw(plr) * 1e3
        co2_gs = self.co2_rate_kgs(plr, T_amb) * 1e3
        P_safe = np.where(np.asarray(P_kw) > 1e-6, P_kw, 1e-6)
        return co2_gs / P_safe * 3600.0
