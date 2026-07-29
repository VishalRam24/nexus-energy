"""
EC163 -- Single-Phase DC-AC Inverter -- F1b Detailed IGBT/Diode Loss Model

Full H-bridge: 4 IGBT+diode pairs. Per-device losses computed analytically
using sinusoidal PWM waveform analysis, then multiplied by 4.

For single-phase SPWM with modulation index m and power factor cos(phi):
    I_avg_igbt  = I_peak / (2*pi) + m*I_peak*cos(phi)/8
    I_rms_igbt  = I_peak * sqrt(1/8 + m*cos(phi)/(3*pi))
    I_avg_diode = I_peak / (2*pi) - m*I_peak*cos(phi)/8
    I_rms_diode = I_peak * sqrt(1/8 - m*cos(phi)/(3*pi))

Per IGBT:
    P_cond_igbt = V_ce0 * I_avg_igbt + r_ce * I_rms_igbt^2
    P_sw_igbt   = (E_on + E_off) * f_sw * (V_dc / V_ref) * (I_out / I_ref)

Per freewheeling diode:
    P_cond_diode = V_f * I_avg_diode + r_d * I_rms_diode^2
    P_rr         = E_rr * f_sw * (V_dc / V_ref) * (I_out / I_ref)

Total (4 devices):
    P_total = 4 * (P_cond_igbt + P_sw_igbt + P_cond_diode + P_rr)

Thermal balance:
    T_j = T_a + P_total * R_theta

Reference:
    Semikron Application Manual (2015), Power Semiconductors.
    Holmes & Lipo (2003), Pulse Width Modulation for Power Converters. Wiley-IEEE.
"""

import numpy as np


class SinglePhaseInverterF1b:
    """Single-phase H-bridge inverter -- detailed IGBT + diode loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_dc = u["V_dc"]["value"]
        self.P_rated = u["P_rated"]["value"]
        self.V_ce0 = u["V_ce0"]["value"]
        self.r_ce = u["r_ce"]["value"]
        self.E_on = u["E_on"]["value"]
        self.E_off = u["E_off"]["value"]
        self.V_f = u["V_f"]["value"]
        self.r_d = u["r_d"]["value"]
        self.E_rr = u["E_rr"]["value"]
        self.f_sw = u["f_sw"]["value"]
        self.V_ref = u["V_ref"]["value"]
        self.I_ref = u["I_ref"]["value"]
        self.T_a = u["T_a"]["value"]
        self.R_theta = u["R_theta"]["value"]

    def ac_rms_voltage(self, v_dc, m):
        """AC RMS output voltage [V] = m * V_dc / (2 * sqrt(2))."""
        return m * v_dc / (2.0 * np.sqrt(2.0))

    def phase_peak_current(self, v_dc, m, p_load, power_factor=1.0):
        """Peak phase current [A] from output power."""
        p_load = np.asarray(p_load, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        v_ac = self.ac_rms_voltage(v_dc, m)
        safe = v_ac * pf
        safe = np.where(safe > 0, safe, 1.0)
        i_rms = np.where(p_load > 0, p_load / safe, 0.0)
        return i_rms * np.sqrt(2.0)

    def igbt_conduction_loss_per_device(self, i_peak, m, power_factor=1.0):
        i_peak = np.asarray(i_peak, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        i_avg = i_peak / (2.0 * np.pi) + m * i_peak * pf / 8.0
        i_rms_sq = i_peak ** 2 * (1.0 / 8.0 + m * pf / (3.0 * np.pi))
        return self.V_ce0 * i_avg + self.r_ce * i_rms_sq

    def igbt_switching_loss_per_device(self, i_peak, v_dc):
        i_peak = np.asarray(i_peak, dtype=float)
        i_sw_avg = i_peak / np.pi
        return (self.E_on + self.E_off) * self.f_sw * (v_dc / self.V_ref) * (i_sw_avg / self.I_ref)

    def diode_conduction_loss_per_device(self, i_peak, m, power_factor=1.0):
        i_peak = np.asarray(i_peak, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        i_avg = np.maximum(i_peak / (2.0 * np.pi) - m * i_peak * pf / 8.0, 0.0)
        i_rms_sq = i_peak ** 2 * np.maximum(1.0 / 8.0 - m * pf / (3.0 * np.pi), 0.0)
        return self.V_f * i_avg + self.r_d * i_rms_sq

    def diode_recovery_loss_per_device(self, i_peak, v_dc):
        i_peak = np.asarray(i_peak, dtype=float)
        i_sw_avg = i_peak / np.pi
        return self.E_rr * self.f_sw * (v_dc / self.V_ref) * (i_sw_avg / self.I_ref)

    def _total_losses_from_ipeak(self, v_dc, p_load, m, power_factor):
        i_peak = self.phase_peak_current(v_dc, m, p_load, power_factor)
        p_ic = 4.0 * self.igbt_conduction_loss_per_device(i_peak, m, power_factor)
        p_is = 4.0 * self.igbt_switching_loss_per_device(i_peak, v_dc)
        p_dc = 4.0 * self.diode_conduction_loss_per_device(i_peak, m, power_factor)
        p_dr = 4.0 * self.diode_recovery_loss_per_device(i_peak, v_dc)
        return p_ic, p_is, p_dc, p_dr

    def loss_breakdown(self, v_dc, p_load, m, power_factor=1.0):
        p_ic, p_is, p_dc, p_dr = self._total_losses_from_ipeak(v_dc, p_load, m, power_factor)
        return {
            "p_igbt_cond_w": p_ic,
            "p_igbt_sw_w": p_is,
            "p_diode_cond_w": p_dc,
            "p_diode_rr_w": p_dr,
        }

    def total_losses(self, v_dc, p_load, m, power_factor=1.0):
        p_ic, p_is, p_dc, p_dr = self._total_losses_from_ipeak(v_dc, p_load, m, power_factor)
        return p_ic + p_is + p_dc + p_dr

    def junction_temperature(self, v_dc, p_load, m, power_factor=1.0):
        p_loss = self.total_losses(v_dc, p_load, m, power_factor)
        return self.T_a + p_loss * self.R_theta

    def efficiency(self, v_dc, p_load, m, power_factor=1.0):
        p_load = np.asarray(p_load, dtype=float)
        p_loss = self.total_losses(v_dc, p_load, m, power_factor)
        p_in = p_load + p_loss
        safe = p_in > 0
        return np.where(safe, p_load / np.where(safe, p_in, 1.0), 0.0)
