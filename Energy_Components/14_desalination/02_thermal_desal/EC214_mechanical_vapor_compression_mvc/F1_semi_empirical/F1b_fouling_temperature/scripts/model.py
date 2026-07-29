"""
EC214 — Mechanical Vapor Compression (MVC) — F1b SEC vs Recovery + Temperature Model

MVC uses a compressor to compress vapor from the evaporator, which provides the
heating steam — no external heat required (all-electric, high SEC but scalable).

Extends F1a with:
  1. SEC vs recovery ratio curve: higher recovery → higher brine concentration →
     higher boiling point elevation (BPE) → larger compressor work.
     SEC = SEC_0 + k_BPE * (C_brine_concentration)^1.5
  2. Temperature correction on compressor work:
     W_comp ∝ T_sat * (compression_ratio^((gamma-1)/gamma) - 1) / eta_comp
     T_sat depends on evaporation temperature.
  3. Fouling/scaling on evaporator surface: k_overall declines with scale thickness.
     U(t) = U0 / (1 + k_foul * t/8760)   [W/(m2*K)]
     Reduced U → need larger dT → higher evaporation T → more compressor work.
  4. Boiling point elevation (BPE) effect: BPE = k_BPE * X^n where X = salinity [g/kg].

References:
    Mistry, K.H. et al. (2011). J. Membr. Sci., 375(1-2), 266-277.
    Darwish, M.A. (1988). Desalination, 69(3), 275-295.
    Lara, J.R. et al. (2011). Desalination, 280(1-3), 413-420.
"""

import numpy as np


def _bpe(X_g_kg):
    """Boiling point elevation [degC] from salinity X [g/kg brine].
    BPE ≈ 0.5 * (X/1000)^2 * (X + 100) [degC] — Sourirajan & Matsuura approx.
    Simplified: BPE = 0.0162 * X  [degC per g/kg, valid ~35-200 g/kg].
    """
    X = np.asarray(X_g_kg, dtype=float)
    return 0.0162 * X  # degC


def _latent_heat(T_degC):
    T = np.asarray(T_degC, dtype=float)
    return np.clip(2501.0 - 2.37 * T, 1500.0, 2600.0)


class MVCF1b:
    """MVC desalination — SEC vs recovery + temperature + fouling model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_evap_ref    = u["T_evap_ref"]["value"]      # degC reference evaporation T
        self.S_feed        = u["S_feed"]["value"]           # g/kg feed salinity
        self.eta_comp      = u["eta_comp"]["value"]         # compressor isentropic efficiency
        self.eta_HX        = u["eta_HX"]["value"]           # heat exchanger effectiveness
        self.dT_comp       = u["dT_comp"]["value"]          # degC compressor temperature lift
        self.gamma_steam   = u["gamma_steam"]["value"]      # steam Cp/Cv
        self.U0            = u["U0"]["value"]               # W/(m2*K) clean HX overall U
        self.k_foul        = u["k_foul"]["value"]           # 1/year fouling rate for U
        self.A_HX          = u["A_HX_m2"]["value"]         # m2 evaporator area
        self.k_BPE         = u["k_BPE"]["value"]            # BPE sensitivity coeff

    # ------------------------------------------------------------------ #
    #  Modifying factors
    # ------------------------------------------------------------------ #

    def _U_fouled(self, operating_hours):
        """Fouled overall heat transfer coefficient [W/(m2*K)].
        U(t) = U0 / (1 + k_foul * t_years)
        """
        t_years = np.asarray(operating_hours, dtype=float) / 8760.0
        return self.U0 / (1.0 + self.k_foul * t_years)

    def _brine_concentration(self, feed_salinity, recovery):
        """Brine salinity [g/kg] from feed salinity and recovery."""
        S_f = np.asarray(feed_salinity, dtype=float)
        r   = np.asarray(recovery, dtype=float)
        r   = np.clip(r, 0.01, 0.99)
        return S_f / (1.0 - r)

    def _effective_dT(self, recovery, T_evap_degC, operating_hours):
        """Effective temperature lift [degC] for compressor.
        dT_eff = dT_comp_base + BPE(brine) + dT_fouling
        dT_fouling: reduced U requires higher dT to transfer same heat.
        Simplified: U_fouled → dT increases by U0/U_f - 1 fraction of dT_comp.
        """
        S_brine = self._brine_concentration(self.S_feed, recovery)
        bpe     = _bpe(S_brine)
        U_f     = self._U_fouled(operating_hours)
        # Fouling adds dT proportional to conductance ratio
        dT_foul = self.dT_comp * (self.U0 / (U_f + 1e-9) - 1.0) * 0.3
        return self.dT_comp + bpe + np.clip(dT_foul, 0.0, 20.0)

    # ------------------------------------------------------------------ #
    #  Compressor work
    # ------------------------------------------------------------------ #

    def compressor_work_kwh_m3(self, recovery, T_evap_degC, operating_hours):
        """Compressor electrical work [kWh/m3 distillate].
        W = m_v * L_v(T_evap) * [(PR)^((gamma-1)/gamma) - 1] / eta_comp
        where PR (pressure ratio) = (T_evap + dT)^n / T_evap^n (saturation pressure ratio).
        Approximated using Clausius-Clapeyron: dP/P ≈ (L_v * dT) / (R * T^2)
        Simplified to: W = L_v * dT_eff / (T_evap * eta_comp) [kJ/kg condensate]
        → kWh/m3 = W[kJ/kg] / 3.6
        """
        T_ev = np.asarray(T_evap_degC, dtype=float)
        T_K  = T_ev + 273.15
        dT   = self._effective_dT(recovery, T_ev, operating_hours)
        L_v  = _latent_heat(T_ev)

        # Mechanical work: W_mech = L_v * dT / T_K / eta_comp  [kJ/kg]
        W_kJ_kg = L_v * dT / T_K / self.eta_comp
        W_kwh_m3 = W_kJ_kg / 3.6
        return np.clip(W_kwh_m3, 2.0, 25.0)

    def sec_kwh_m3(self, recovery, T_evap_degC, operating_hours):
        """Total SEC [kWh_e/m3 distillate] = compressor + feed/brine pumping.
        Pumping ≈ 0.2-0.5 kWh/m3 (small relative to compression).
        """
        W_comp   = self.compressor_work_kwh_m3(recovery, T_evap_degC, operating_hours)
        W_pumps  = 0.35  # kWh/m3 fixed auxiliary pumping
        return np.clip(W_comp + W_pumps, 2.0, 30.0)

    def production_rate_m3_h(self, recovery, T_evap_degC, operating_hours, feed_flow_m3_h):
        """Distillate production rate [m3/h]."""
        r = np.asarray(recovery, dtype=float)
        Q_f = np.asarray(feed_flow_m3_h, dtype=float)
        return Q_f * r

    def fouling_factor(self, operating_hours):
        """HX fouling factor (U/U0)."""
        U_f = self._U_fouled(operating_hours)
        return U_f / self.U0

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, recovery, T_evap_degC, operating_hours, feed_flow_m3_h):
        """Full computation.

        Parameters
        ----------
        recovery        : 0-1      — water recovery ratio
        T_evap_degC     : degC     — evaporator temperature (50-80 degC)
        operating_hours : hours    — cumulative operating hours
        feed_flow_m3_h  : m3/h    — feed flow rate

        Returns
        -------
        dict with sec_kwh_m3, compressor_work_kwh_m3, production_rate_m3_h,
                  brine_salinity_gkg, bpe_degC, fouling_factor
        """
        r       = np.asarray(recovery, dtype=float)
        S_brine = self._brine_concentration(self.S_feed, r)
        bpe     = _bpe(S_brine)
        sec     = self.sec_kwh_m3(r, T_evap_degC, operating_hours)
        W_comp  = self.compressor_work_kwh_m3(r, T_evap_degC, operating_hours)
        Q_prod  = self.production_rate_m3_h(r, T_evap_degC, operating_hours, feed_flow_m3_h)
        foul_f  = self.fouling_factor(operating_hours)

        return {
            "sec_kwh_m3":               sec,
            "compressor_work_kwh_m3":   W_comp,
            "production_rate_m3_h":     Q_prod,
            "brine_salinity_gkg":       S_brine,
            "bpe_degC":                 bpe,
            "fouling_factor":           foul_f,
        }
