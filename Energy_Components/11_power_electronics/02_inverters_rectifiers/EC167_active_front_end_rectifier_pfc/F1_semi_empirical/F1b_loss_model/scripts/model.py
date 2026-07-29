"""
EC167 -- Active Front End (AFE) / PFC Rectifier -- F1b Detailed IGBT + Diode Loss Model

An AFE rectifier is topologically identical to a three-phase VSC inverter but
operated in rectification mode (power flows from AC to DC). Each leg has an
IGBT + freewheeling diode.  The same per-device loss formulae as EC164 apply,
but the current direction in the switching pattern is reversed.

Per IGBT (one device in a half-bridge leg):
    Conduction: P_cond_igbt = V_ce0 * I_avg_igbt + r_ce * I_rms_igbt^2
    Switching:  P_sw_igbt = (E_on + E_off) * f_sw * (V_dc / V_ref) * (I_sw / I_ref)

Per freewheeling diode:
    Conduction: P_cond_diode = V_f * I_avg_diode + r_d * I_rms_diode^2
    Reverse recovery: P_rr = E_rr * f_sw * (V_dc / V_ref) * (I_sw / I_ref)

For sinusoidal PWM with modulation index m and input power factor cos(phi):
    (In rectifier mode the active switch carries current during the interval when
    the AC phase voltage pulls current towards the DC bus; same SPWM distribution
    applies as for an inverter with the same topology.)
    I_avg_igbt  = I_peak / (2π) * (1 + m*π*cos(phi)/4)
    I_rms_igbt  = I_peak * sqrt(1/8 + m*cos(phi)/(3π))
    I_avg_diode = I_peak / (2π) * (1 - m*π*cos(phi)/4)
    I_rms_diode = I_peak * sqrt(max(1/8 - m*cos(phi)/(3π), 0))

Total (6 devices):
    P_total = 6 * (P_cond_igbt + P_sw_igbt + P_cond_diode + P_rr)

Efficiency:
    eta = P_dc_out / (P_dc_out + P_total)

PFC function (boost-mode AFE):
    V_dc_min = √2 * V_LL_rms * m_needed  => m can be computed from V_dc target
    m_needed = V_dc / (√2 * V_LL_rms)

Thermal:
    T_j = T_a + P_worst_device * R_th_ja

References:
    Mohan, Undeland & Robbins (2003). Power Electronics, 3rd ed. Wiley.
    Semikron Application Manual (2015). Power Semiconductors, section 3.4.
    Blaabjerg, F. et al. (2006). IEEE Trans. Ind. Electron., 53(2), 486-496.
"""

import numpy as np


class AFERectifierF1b:
    """Active Front End / PFC rectifier -- detailed IGBT + diode loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_dc = u["V_dc"]["value"]        # V nominal DC bus
        self.P_rated = u["P_rated"]["value"]  # W
        self.V_ce0 = u["V_ce0"]["value"]      # V
        self.r_ce = u["r_ce"]["value"]        # Ohm
        self.E_on = u["E_on"]["value"]        # J
        self.E_off = u["E_off"]["value"]      # J
        self.V_f = u["V_f"]["value"]          # V (freewheeling diode)
        self.r_d = u["r_d"]["value"]          # Ohm
        self.E_rr = u["E_rr"]["value"]        # J
        self.f_sw = u["f_sw"]["value"]        # Hz
        self.V_ref = u["V_ref"]["value"]      # V
        self.I_ref = u["I_ref"]["value"]      # A
        self.R_th_ja = u["R_th_ja"]["value"]  # K/W
        self.T_a = u["T_a"]["value"]          # degC

    def modulation_index(self, v_dc, v_ac_ll_rms):
        """
        Required modulation index m = V_dc / (√2 * V_LL_rms).
        Clamped to [0.0, 1.15] (over-modulation limit).
        """
        v_ll = np.asarray(v_ac_ll_rms, dtype=float)
        vd = np.asarray(v_dc, dtype=float)
        m = vd / np.where(v_ll > 0, np.sqrt(2.0) * v_ll, 1.0)
        return np.clip(m, 0.0, 1.15)

    def phase_peak_current(self, v_ac_ll_rms, p_input, power_factor=1.0):
        """Peak phase current [A] drawn from the AC supply."""
        p = np.asarray(p_input, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        v_ll = np.asarray(v_ac_ll_rms, dtype=float)
        # P = sqrt(3) * V_LL * I_rms * PF
        denom = np.sqrt(3.0) * v_ll * pf
        denom = np.where(denom > 0, denom, 1.0)
        i_rms = np.where(p > 0, p / denom, 0.0)
        return i_rms * np.sqrt(2.0)

    def igbt_conduction_loss_per_device(self, i_pk, m, pf):
        """IGBT conduction loss per device [W]."""
        i_pk = np.asarray(i_pk, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(pf, dtype=float)
        i_avg = i_pk / (2.0 * np.pi) * (1.0 + m * np.pi * pf / 4.0)
        i_rms_sq = i_pk ** 2 * (1.0 / 8.0 + m * pf / (3.0 * np.pi))
        return self.V_ce0 * i_avg + self.r_ce * i_rms_sq

    def igbt_switching_loss_per_device(self, i_pk, v_dc):
        """IGBT switching loss per device [W]."""
        i_pk = np.asarray(i_pk, dtype=float)
        v_dc = np.asarray(v_dc, dtype=float)
        i_sw = i_pk / np.pi
        return (self.E_on + self.E_off) * self.f_sw * (v_dc / self.V_ref) * (i_sw / self.I_ref)

    def diode_conduction_loss_per_device(self, i_pk, m, pf):
        """Freewheeling diode conduction loss per device [W]."""
        i_pk = np.asarray(i_pk, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(pf, dtype=float)
        i_avg = i_pk / (2.0 * np.pi) * (1.0 - m * np.pi * pf / 4.0)
        i_avg = np.maximum(i_avg, 0.0)
        i_rms_sq = i_pk ** 2 * np.maximum(1.0 / 8.0 - m * pf / (3.0 * np.pi), 0.0)
        return self.V_f * i_avg + self.r_d * i_rms_sq

    def diode_recovery_loss_per_device(self, i_pk, v_dc):
        """Freewheeling diode reverse recovery loss per device [W]."""
        i_pk = np.asarray(i_pk, dtype=float)
        v_dc = np.asarray(v_dc, dtype=float)
        i_sw = i_pk / np.pi
        return self.E_rr * self.f_sw * (v_dc / self.V_ref) * (i_sw / self.I_ref)

    def loss_breakdown(self, v_ac_ll_rms, v_dc, p_input, power_factor=1.0):
        """Full loss breakdown for 3-phase AFE [W]."""
        m = self.modulation_index(v_dc, v_ac_ll_rms)
        pf = np.asarray(power_factor, dtype=float)
        i_pk = self.phase_peak_current(v_ac_ll_rms, p_input, power_factor)

        p_igbt_cond = 6.0 * self.igbt_conduction_loss_per_device(i_pk, m, pf)
        p_igbt_sw = 6.0 * self.igbt_switching_loss_per_device(i_pk, v_dc)
        p_diode_cond = 6.0 * self.diode_conduction_loss_per_device(i_pk, m, pf)
        p_diode_rr = 6.0 * self.diode_recovery_loss_per_device(i_pk, v_dc)

        return {
            "p_igbt_cond_w": p_igbt_cond,
            "p_igbt_sw_w": p_igbt_sw,
            "p_diode_cond_w": p_diode_cond,
            "p_diode_rr_w": p_diode_rr,
        }

    def total_losses(self, v_ac_ll_rms, v_dc, p_input, power_factor=1.0):
        """Total AFE losses [W]."""
        bd = self.loss_breakdown(v_ac_ll_rms, v_dc, p_input, power_factor)
        return bd["p_igbt_cond_w"] + bd["p_igbt_sw_w"] + bd["p_diode_cond_w"] + bd["p_diode_rr_w"]

    def efficiency(self, v_ac_ll_rms, v_dc, p_input, power_factor=1.0):
        """Efficiency = P_dc_out / P_ac_in."""
        p_in = np.asarray(p_input, dtype=float)
        p_loss = self.total_losses(v_ac_ll_rms, v_dc, p_input, power_factor)
        p_out = p_in - p_loss
        safe = p_in > 0
        return np.where(safe, np.maximum(p_out, 0.0) / np.where(safe, p_in, 1.0), 0.0)

    def junction_temperature(self, v_ac_ll_rms, v_dc, p_input, power_factor=1.0):
        """Hottest device junction temperature [degC]."""
        m = self.modulation_index(v_dc, v_ac_ll_rms)
        pf = np.asarray(power_factor, dtype=float)
        i_pk = self.phase_peak_current(v_ac_ll_rms, p_input, power_factor)
        # IGBT is usually hottest at high PF (more igbt current)
        p_igbt = (self.igbt_conduction_loss_per_device(i_pk, m, pf) +
                  self.igbt_switching_loss_per_device(i_pk, v_dc))
        return self.T_a + p_igbt * self.R_th_ja
