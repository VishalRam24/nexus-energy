"""
EC211 — Forward Osmosis (FO) — F1b Draw Solution Reconcentration + Membrane Aging Model

Extends F1a with:
  1. Draw solution reconcentration energy (dominates FO total SEC):
     For thermal draw (NH3/CO2): Q_regen = H_vap * 1/GOR_draw
     For osmotic dilution (RO regeneration): SEC_regen = SEC_RO(C_draw_concentrated)
  2. Concentration polarization — internal concentration polarization (ICP) factor:
     J_w = A * (pi_draw_eff - pi_feed_eff)
     pi_eff = pi_bulk * exp(-J_w/k_mass)   [ECP], * exp(+J_w/D_eff/tau) [ICP]
  3. Temperature correction: membrane water permeability A(T):
     A(T) = A_ref * exp(E_a_w/R * (1/T_ref - 1/T))  [E_a_w ~ 20 kJ/mol for water]
  4. Membrane fouling: A(t) = A0 * (1 - k_foul * t/8760), k_foul ~ 0.05-0.10/year.

References:
    McGinnis, R.L. & Elimelech, M. (2008). Environ. Sci. Technol., 42(23), 8625-8629.
    Cath, T.Y. et al. (2006). J. Membrane Sci., 281(1-2), 70-87.
    Zhao, S. et al. (2012). Water Research, 46(4), 1195-1204.
"""

import numpy as np

R_GAS = 8.314   # J/(mol*K)


class FOF1b:
    """Forward Osmosis — draw reconcentration + ICP + temperature + aging model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_ref       = u["A_ref"]["value"]          # L/(m2*h*bar) water permeability
        self.B           = u["B"]["value"]               # L/(m2*h) salt permeability
        self.membrane_area = u["membrane_area_m2"]["value"]
        self.k_mass      = u["k_mass"]["value"]          # m/s ECP mass transfer coeff
        self.D_eff       = u["D_eff"]["value"]           # m2/s effective diffusivity in membrane
        self.tau_mem     = u["tau_mem"]["value"]          # membrane tortuosity
        self.thickness   = u["thickness_m"]["value"]     # m support layer thickness
        self.E_a_w       = u["E_a_w_kJ_mol"]["value"]    # kJ/mol activation energy for water
        self.T_ref_K     = u["T_ref_K"]["value"]         # K
        self.k_foul      = u["k_foul"]["value"]          # 1/year fouling rate
        self.pi_draw_ref = u["pi_draw_ref"]["value"]     # bar draw osmotic pressure at design
        self.SEC_regen   = u["SEC_regen"]["value"]       # kWh/m3 draw reconcentration SEC
        self.recovery    = u["recovery"]["value"]        # design water recovery

    # ------------------------------------------------------------------ #
    #  Modifying factors
    # ------------------------------------------------------------------ #

    def _fouling_factor(self, operating_hours):
        """Permeability decline due to fouling (linear model for FO).
        A(t) = A0 * (1 - k_foul * t_years)  clipped at 0.3 (irreversible fouling floor).
        Literature: FO membranes show 5-10%/yr flux decline; linear adequate for F1b.
        """
        t_years = np.asarray(operating_hours, dtype=float) / 8760.0
        return np.clip(1.0 - self.k_foul * t_years, 0.3, 1.0)

    def _temperature_factor(self, T_degC):
        """Arrhenius temperature correction for membrane water permeability.
        A(T) = A_ref * exp(E_a_w/R * (1/T_ref - 1/T))  [E_a_w ~ 20 kJ/mol → positive coeff]
        Warmer → higher A.
        """
        T_K = np.asarray(T_degC, dtype=float) + 273.15
        exp_arg = (self.E_a_w * 1000.0 / R_GAS) * (1.0 / self.T_ref_K - 1.0 / T_K)
        return np.exp(exp_arg)

    def _icp_factor(self, J_w_lmh):
        """Internal concentration polarization attenuation factor.
        pi_eff = pi_bulk * exp(-J_w * t_mem / D_eff)
        where t_mem = thickness * tortuosity.
        Returns dimensionless factor < 1 (ICP reduces effective driving force).
        """
        J_w_ms = np.asarray(J_w_lmh, dtype=float) / 3.6e6  # L/(m2*h) → m/s
        t_mem = self.thickness * self.tau_mem
        return np.exp(-J_w_ms * t_mem / (self.D_eff + 1e-20))

    # ------------------------------------------------------------------ #
    #  Water flux
    # ------------------------------------------------------------------ #

    def water_flux_lmh(self, pi_draw_bar, pi_feed_bar, T_degC, operating_hours):
        """Water flux [L/(m2*h)].
        Iterative ICP correction (one-step approximation here):
        J_w_0 = A_eff * (pi_draw - pi_feed)  [no ICP]
        J_w   = A_eff * (pi_draw * ICP_factor - pi_feed)  [with ICP]
        """
        pi_D = np.asarray(pi_draw_bar, dtype=float)
        pi_F = np.asarray(pi_feed_bar, dtype=float)
        foul  = self._fouling_factor(operating_hours)
        T_f   = self._temperature_factor(T_degC)
        A_eff = self.A_ref * foul * T_f

        # Initial estimate (no ICP)
        J0 = A_eff * np.clip(pi_D - pi_F, 0.0, None)
        # ICP correction
        icp_f = self._icp_factor(J0)
        J_w   = A_eff * np.clip(pi_D * icp_f - pi_F, 0.0, None)
        return np.clip(J_w, 0.0, 200.0)

    def permeate_flow_m3_h(self, pi_draw_bar, pi_feed_bar, T_degC, operating_hours):
        """Permeate (product) flow rate [m3/h]."""
        J_w = self.water_flux_lmh(pi_draw_bar, pi_feed_bar, T_degC, operating_hours)
        return J_w * self.membrane_area / 1000.0  # L → m3

    # ------------------------------------------------------------------ #
    #  Energy
    # ------------------------------------------------------------------ #

    def sec_total_kwh_m3(self, pi_draw_bar, pi_feed_bar, T_degC, operating_hours):
        """Total SEC [kWh/m3 product] = draw reconcentration + pumping.
        Draw reconcentration dominates. Pumping ≈ 0.1-0.3 kWh/m3.
        SEC scales with recovery: lower recovery → more draw per m3 product.
        """
        J_w       = self.water_flux_lmh(pi_draw_bar, pi_feed_bar, T_degC, operating_hours)
        # Recovery-adjusted SEC: SEC_actual = SEC_design / recovery
        # (more diluted draw per unit volume means more regeneration work)
        actual_recovery = self.recovery * (J_w / (self.A_ref * (self.pi_draw_ref - pi_feed_bar) + 1e-9))
        actual_recovery = np.clip(actual_recovery, 0.05, 0.85)
        sec_regen = self.SEC_regen / np.clip(actual_recovery, 0.1, 1.0)
        sec_pumping = 0.15  # kWh/m3 (low-pressure pumping)
        return np.clip(sec_regen + sec_pumping, 0.5, 20.0)

    def salt_leakage_mg_l(self, pi_draw_bar, pi_feed_bar, T_degC, operating_hours):
        """Reverse salt flux [mg/L product] due to salt permeability B.
        RSF = B * (C_draw - C_feed) / J_w
        C_draw ≈ pi_draw / 0.7 * 1000 [mol/m3] * M_NaCl [g/mol] → simplified to pi/pressure_coeff.
        Returns concentration of salt in product water [mg/L].
        """
        J_w = self.water_flux_lmh(pi_draw_bar, pi_feed_bar, T_degC, operating_hours)
        J_w_ms = J_w / 3.6e6
        # Approximate concentration gradient driving reverse salt flux
        pi_D = np.asarray(pi_draw_bar, dtype=float)
        pi_F = np.asarray(pi_feed_bar, dtype=float)
        dC_approx = (pi_D - pi_F) / 0.7 * 1000.0 * 58.44  # g/m3 = mg/L
        RSF_mg_l = self.B / 3.6e6 * dC_approx / (J_w_ms + 1e-10) * 1000.0
        return np.clip(RSF_mg_l, 0.0, 1000.0)

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, pi_draw_bar, pi_feed_bar, T_feed_degC, operating_hours):
        """Full computation.

        Parameters
        ----------
        pi_draw_bar     : bar   — draw solution osmotic pressure (e.g. NaCl draw: 20-100 bar)
        pi_feed_bar     : bar   — feed water osmotic pressure (seawater: ~27 bar)
        T_feed_degC     : degC  — feed/membrane temperature
        operating_hours : hours — cumulative operating hours

        Returns
        -------
        dict with permeate_flow_m3_h, water_flux_lmh, sec_total_kwh_m3,
                  salt_leakage_mg_l, flux_decline_factor
        """
        Q_p    = self.permeate_flow_m3_h(pi_draw_bar, pi_feed_bar, T_feed_degC, operating_hours)
        J_w    = self.water_flux_lmh(pi_draw_bar, pi_feed_bar, T_feed_degC, operating_hours)
        sec    = self.sec_total_kwh_m3(pi_draw_bar, pi_feed_bar, T_feed_degC, operating_hours)
        rss    = self.salt_leakage_mg_l(pi_draw_bar, pi_feed_bar, T_feed_degC, operating_hours)
        flux_f = self._fouling_factor(operating_hours) * self._temperature_factor(T_feed_degC)

        return {
            "permeate_flow_m3_h":    Q_p,
            "water_flux_lmh":        J_w,
            "sec_total_kwh_m3":      sec,
            "salt_leakage_mg_l":     rss,
            "flux_decline_factor":   flux_f,
        }
