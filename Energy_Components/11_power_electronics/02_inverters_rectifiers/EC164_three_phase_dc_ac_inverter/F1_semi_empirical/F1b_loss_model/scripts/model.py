"""
EC164 -- Three-Phase DC-AC Inverter -- F1b Detailed IGBT/Diode Loss Model

Full-bridge three-phase inverter with 6 IGBT+diode pairs.
Per-device losses computed analytically, then multiplied by 6.

Per IGBT (one device in a half-bridge leg):
    Conduction: P_cond_igbt = V_ce0 * I_avg_igbt + r_ce * I_rms_igbt^2
    Switching:  P_sw_igbt = (E_on + E_off) * f_sw * (V_dc / V_ref) * (I_out / I_ref)

Per freewheeling diode:
    Conduction: P_cond_diode = V_f * I_avg_diode + r_d * I_rms_diode^2
    Recovery:   P_rr = E_rr * f_sw * (V_dc / V_ref) * (I_out / I_ref)

For sinusoidal PWM with modulation index m and power factor cos(phi):
    I_avg_igbt  = I_peak / (2*pi) * (1 + m*pi*cos(phi)/4)
    I_rms_igbt  = I_peak * sqrt(1/(8) + m*cos(phi)/(3*pi))
    I_avg_diode = I_peak / (2*pi) * (1 - m*pi*cos(phi)/4)
    I_rms_diode = I_peak * sqrt(1/(8) - m*cos(phi)/(3*pi))

Total (6 devices):
    P_total = 6 * (P_cond_igbt + P_sw_igbt + P_cond_diode + P_rr)

Reference:
    Semikron Application Manual (2015), Power Semiconductors.
    Mohan, Undeland & Robbins (2003), Power Electronics, 3rd ed. Wiley.
"""

import numpy as np


class ThreePhaseInverterF1b:
    """Three-phase inverter -- detailed IGBT + diode loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_dc = u["V_dc"]["value"]       # V
        self.P_rated = u["P_rated"]["value"]  # W
        self.V_ce0 = u["V_ce0"]["value"]      # V
        self.r_ce = u["r_ce"]["value"]        # Ohm
        self.E_on = u["E_on"]["value"]        # J
        self.E_off = u["E_off"]["value"]      # J
        self.V_f = u["V_f"]["value"]          # V
        self.r_d = u["r_d"]["value"]          # Ohm
        self.E_rr = u["E_rr"]["value"]        # J
        self.f_sw = u["f_sw"]["value"]        # Hz
        self.V_ref = u["V_ref"]["value"]      # V
        self.I_ref = u["I_ref"]["value"]      # A

    def ac_rms_voltage(self, v_dc, m):
        """Line-to-line AC RMS voltage [V] (SVPWM): V_ac = m * V_dc / sqrt(2)."""
        return m * v_dc / np.sqrt(2.0)

    def phase_peak_current(self, v_dc, m, p_load, power_factor=1.0):
        """Peak phase current [A] from output power."""
        p_load = np.asarray(p_load, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        v_ac = self.ac_rms_voltage(v_dc, m)
        # P = sqrt(3) * V_LL * I_L * PF  =>  I_L = P / (sqrt(3) * V_LL * PF)
        safe_denom = np.sqrt(3.0) * v_ac * pf
        safe_denom = np.where(safe_denom > 0, safe_denom, 1.0)
        i_rms = np.where(p_load > 0, p_load / safe_denom, 0.0)
        return i_rms * np.sqrt(2.0)  # peak = rms * sqrt(2)

    def igbt_conduction_loss_per_device(self, i_peak, m, power_factor=1.0):
        """IGBT conduction loss per device [W]."""
        i_peak = np.asarray(i_peak, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)

        # Average and RMS currents through IGBT (sinusoidal PWM)
        i_avg = i_peak / (2.0 * np.pi) * (1.0 + m * np.pi * pf / 4.0)
        i_rms_sq = i_peak ** 2 * (1.0 / 8.0 + m * pf / (3.0 * np.pi))

        return self.V_ce0 * i_avg + self.r_ce * i_rms_sq

    def igbt_switching_loss_per_device(self, i_peak, v_dc):
        """IGBT switching loss per device [W]."""
        i_peak = np.asarray(i_peak, dtype=float)
        v_dc = np.asarray(v_dc, dtype=float)
        # Switching energy scaled linearly with voltage and current
        # Average current seen during switching ~ I_peak / pi
        i_sw_avg = i_peak / np.pi
        return (self.E_on + self.E_off) * self.f_sw * (v_dc / self.V_ref) * (i_sw_avg / self.I_ref)

    def diode_conduction_loss_per_device(self, i_peak, m, power_factor=1.0):
        """Freewheeling diode conduction loss per device [W]."""
        i_peak = np.asarray(i_peak, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)

        i_avg = i_peak / (2.0 * np.pi) * (1.0 - m * np.pi * pf / 4.0)
        i_avg = np.maximum(i_avg, 0.0)  # cannot be negative
        i_rms_sq = i_peak ** 2 * np.maximum(1.0 / 8.0 - m * pf / (3.0 * np.pi), 0.0)

        return self.V_f * i_avg + self.r_d * i_rms_sq

    def diode_recovery_loss_per_device(self, i_peak, v_dc):
        """Diode reverse recovery loss per device [W]."""
        i_peak = np.asarray(i_peak, dtype=float)
        v_dc = np.asarray(v_dc, dtype=float)
        i_sw_avg = i_peak / np.pi
        return self.E_rr * self.f_sw * (v_dc / self.V_ref) * (i_sw_avg / self.I_ref)

    def loss_breakdown(self, v_dc, p_load, m, power_factor=1.0):
        """
        Full loss breakdown for the entire 6-device bridge [W].

        Returns dict with per-category losses (already multiplied by 6).
        """
        i_peak = self.phase_peak_current(v_dc, m, p_load, power_factor)

        p_igbt_cond = 6.0 * self.igbt_conduction_loss_per_device(i_peak, m, power_factor)
        p_igbt_sw = 6.0 * self.igbt_switching_loss_per_device(i_peak, v_dc)
        p_diode_cond = 6.0 * self.diode_conduction_loss_per_device(i_peak, m, power_factor)
        p_diode_rr = 6.0 * self.diode_recovery_loss_per_device(i_peak, v_dc)

        return {
            "p_igbt_cond_w": p_igbt_cond,
            "p_igbt_sw_w": p_igbt_sw,
            "p_diode_cond_w": p_diode_cond,
            "p_diode_rr_w": p_diode_rr,
        }

    def total_losses(self, v_dc, p_load, m, power_factor=1.0):
        """Total inverter losses [W]."""
        bd = self.loss_breakdown(v_dc, p_load, m, power_factor)
        return bd["p_igbt_cond_w"] + bd["p_igbt_sw_w"] + bd["p_diode_cond_w"] + bd["p_diode_rr_w"]

    def efficiency(self, v_dc, p_load, m, power_factor=1.0):
        """Efficiency = P_out / (P_out + P_loss)."""
        p_load = np.asarray(p_load, dtype=float)
        p_loss = self.total_losses(v_dc, p_load, m, power_factor)
        p_in = p_load + p_loss
        safe = p_in > 0
        return np.where(safe, p_load / np.where(safe, p_in, 1.0), 0.0)
