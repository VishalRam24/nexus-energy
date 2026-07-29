"""
EC169 -- Variable Frequency Drive (VFD) -- F1b Detailed Loss Model

A VFD consists of:
  1. Front-end rectifier (diode bridge or AFE)
  2. DC link capacitor
  3. PWM inverter (3-phase IGBT bridge)

This F1b model decomposes losses into:
  (A) Rectifier losses (diode bridge, 6 diodes)
  (B) DC-link losses (ESR of capacitor bank, small)
  (C) Inverter IGBT + freewheeling diode losses (6 devices)

The inverter section uses the same per-device formulae as EC164 (3-phase inverter).

Rectifier losses (3-phase diode bridge, same as EC166):
    P_rect = 6 * (V_f_rect * I_D_avg + r_d_rect * I_D_rms^2)
    I_D_avg = I_dc / 3,  I_D_rms = I_dc / sqrt(3)
    I_dc = P_motor / V_dc  (steady-state power balance)

Inverter losses (same as EC164 -- per IGBT/diode device × 6):
    P_inv = 6*(P_cond_igbt + P_sw_igbt + P_cond_diode + P_rr)

DC-link ESR loss (small):
    P_dc_link = I_ripple^2 * R_esr
    I_ripple ~ I_dc * k_ripple  (k_ripple ≈ 0.1–0.2 for well-designed link)

Motor load (output shaft):
    The VFD drives a motor at variable speed.  At part load the inverter output
    voltage and frequency scale roughly with the V/f ratio:
        f_out = f_base * (speed_pu)
        V_out_rms = V_rated * (speed_pu)  (constant V/f below base speed)
    The inverter modulation index: m = V_out_rms * sqrt(2) / V_dc

Part-load switching loss dip:
    At light load, switching losses dominate → efficiency dip.
    eta vs P_out has characteristic minimum at low power fraction.

Temperature:
    T_j_igbt = T_a + (P_cond_igbt + P_sw_igbt) / 6 * R_th_ja_igbt

References:
    Mohan, Undeland & Robbins (2003). Power Electronics, 3rd ed. Wiley.
    IEC 61800-9-2:2017. Adjustable speed electrical power drive systems — Part 9-2:
        Ecodesign for power drive systems (efficiency classes IE0–IE3).
    Semikron Application Manual (2015). Power Semiconductors.
"""

import numpy as np


class VFDf1b:
    """Variable Frequency Drive (VFD) -- detailed loss model across all sub-stages."""

    def __init__(self, params: dict):
        u = params["unit"]
        # System
        self.V_dc = u["V_dc"]["value"]          # V
        self.P_rated = u["P_rated"]["value"]    # W (motor rated)
        self.V_ac_ll = u["V_ac_ll"]["value"]    # V (supply L-L rms)
        self.f_base = u["f_base"]["value"]      # Hz (motor base frequency, 50 Hz)

        # Rectifier diodes
        self.V_f_rect = u["V_f_rect"]["value"]  # V
        self.r_d_rect = u["r_d_rect"]["value"]  # Ohm

        # DC link
        self.R_esr = u["R_esr"]["value"]        # Ohm
        self.k_ripple = u["k_ripple"]["value"]  # fraction

        # Inverter IGBTs
        self.V_ce0 = u["V_ce0"]["value"]        # V
        self.r_ce = u["r_ce"]["value"]          # Ohm
        self.E_on = u["E_on"]["value"]          # J
        self.E_off = u["E_off"]["value"]        # J

        # Inverter diodes
        self.V_f_inv = u["V_f_inv"]["value"]    # V
        self.r_d_inv = u["r_d_inv"]["value"]    # Ohm
        self.E_rr = u["E_rr"]["value"]          # J

        self.f_sw = u["f_sw"]["value"]          # Hz
        self.V_ref = u["V_ref"]["value"]        # V
        self.I_ref = u["I_ref"]["value"]        # A
        self.R_th_ja = u["R_th_ja"]["value"]    # K/W
        self.T_a = u["T_a"]["value"]            # degC

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def dc_current(self, p_motor):
        """DC link current [A]: I_dc = P_motor / V_dc (power balance)."""
        p = np.asarray(p_motor, dtype=float)
        return p / np.where(self.V_dc > 0, self.V_dc, 1.0)

    def modulation_index(self, speed_pu):
        """
        Modulation index at given speed p.u.
        V/f control: m proportional to speed, max 1.0 at base speed.
        """
        s = np.asarray(speed_pu, dtype=float)
        return np.clip(s, 0.0, 1.0)  # simplified: m = speed_pu

    def phase_peak_current(self, p_motor, power_factor=0.85):
        """Peak AC phase current to the motor [A]."""
        p = np.asarray(p_motor, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        # V_ll_out scales with speed, but we work with rated voltage at rated speed
        # Use rated voltage for worst-case current calculation
        v_ll = self.V_ac_ll
        denom = np.sqrt(3.0) * v_ll * pf
        i_rms = np.where(p > 0, p / denom, 0.0)
        return i_rms * np.sqrt(2.0)

    # ------------------------------------------------------------------
    # Rectifier losses (3-phase diode bridge)
    # ------------------------------------------------------------------

    def rectifier_losses(self, p_motor):
        """Rectifier diode bridge losses [W]."""
        i_dc = self.dc_current(p_motor)
        # 6-pulse bridge: I_D_avg = I_dc/3, I_D_rms = I_dc/sqrt(3)
        i_d_avg = i_dc / 3.0
        i_d_rms = i_dc / np.sqrt(3.0)
        p_per_diode = self.V_f_rect * i_d_avg + self.r_d_rect * i_d_rms ** 2
        return 6.0 * p_per_diode

    # ------------------------------------------------------------------
    # DC link losses
    # ------------------------------------------------------------------

    def dc_link_losses(self, p_motor):
        """DC link capacitor ESR losses [W]."""
        i_dc = self.dc_current(p_motor)
        i_ripple = i_dc * self.k_ripple
        return i_ripple ** 2 * self.R_esr

    # ------------------------------------------------------------------
    # Inverter losses (3-phase IGBT bridge)
    # ------------------------------------------------------------------

    def _igbt_cond_per_device(self, i_pk, m, pf):
        i_avg = i_pk / (2.0 * np.pi) * (1.0 + m * np.pi * pf / 4.0)
        i_rms_sq = i_pk ** 2 * (1.0 / 8.0 + m * pf / (3.0 * np.pi))
        return self.V_ce0 * i_avg + self.r_ce * i_rms_sq

    def _igbt_sw_per_device(self, i_pk):
        i_sw = i_pk / np.pi
        return (self.E_on + self.E_off) * self.f_sw * (self.V_dc / self.V_ref) * (i_sw / self.I_ref)

    def _diode_cond_per_device(self, i_pk, m, pf):
        i_avg = i_pk / (2.0 * np.pi) * (1.0 - m * np.pi * pf / 4.0)
        i_avg = np.maximum(i_avg, 0.0)
        i_rms_sq = i_pk ** 2 * np.maximum(1.0 / 8.0 - m * pf / (3.0 * np.pi), 0.0)
        return self.V_f_inv * i_avg + self.r_d_inv * i_rms_sq

    def _diode_rr_per_device(self, i_pk):
        i_sw = i_pk / np.pi
        return self.E_rr * self.f_sw * (self.V_dc / self.V_ref) * (i_sw / self.I_ref)

    def inverter_losses(self, p_motor, speed_pu, power_factor=0.85):
        """Inverter IGBT bridge losses [W]."""
        m = self.modulation_index(speed_pu)
        pf = np.asarray(power_factor, dtype=float)
        i_pk = self.phase_peak_current(p_motor, power_factor)

        p_igbt_cond = 6.0 * self._igbt_cond_per_device(i_pk, m, pf)
        p_igbt_sw = 6.0 * self._igbt_sw_per_device(i_pk)
        p_diode_cond = 6.0 * self._diode_cond_per_device(i_pk, m, pf)
        p_diode_rr = 6.0 * self._diode_rr_per_device(i_pk)

        return p_igbt_cond + p_igbt_sw + p_diode_cond + p_diode_rr

    def inverter_loss_breakdown(self, p_motor, speed_pu, power_factor=0.85):
        """Inverter loss breakdown dict [W]."""
        m = self.modulation_index(speed_pu)
        pf = np.asarray(power_factor, dtype=float)
        i_pk = self.phase_peak_current(p_motor, power_factor)

        return {
            "p_igbt_cond_w": 6.0 * self._igbt_cond_per_device(i_pk, m, pf),
            "p_igbt_sw_w": 6.0 * self._igbt_sw_per_device(i_pk),
            "p_diode_cond_w": 6.0 * self._diode_cond_per_device(i_pk, m, pf),
            "p_diode_rr_w": 6.0 * self._diode_rr_per_device(i_pk),
        }

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------

    def loss_breakdown(self, p_motor, speed_pu, power_factor=0.85):
        """Full VFD loss breakdown [W]."""
        inv_bd = self.inverter_loss_breakdown(p_motor, speed_pu, power_factor)
        return {
            "p_rectifier_w": self.rectifier_losses(p_motor),
            "p_dc_link_w": self.dc_link_losses(p_motor),
            **inv_bd,
        }

    def total_losses(self, p_motor, speed_pu, power_factor=0.85):
        """Total VFD losses [W]."""
        bd = self.loss_breakdown(p_motor, speed_pu, power_factor)
        return sum(bd.values())

    def efficiency(self, p_motor, speed_pu, power_factor=0.85):
        """VFD system efficiency = P_motor / (P_motor + P_loss)."""
        p = np.asarray(p_motor, dtype=float)
        p_loss = self.total_losses(p_motor, speed_pu, power_factor)
        p_in = p + p_loss
        safe = p_in > 0
        return np.where(safe, p / np.where(safe, p_in, 1.0), 0.0)

    def junction_temperature(self, p_motor, speed_pu, power_factor=0.85):
        """IGBT junction temperature [degC]."""
        m = self.modulation_index(speed_pu)
        pf = np.asarray(power_factor, dtype=float)
        i_pk = self.phase_peak_current(p_motor, power_factor)
        p_igbt = (self._igbt_cond_per_device(i_pk, m, pf) +
                  self._igbt_sw_per_device(i_pk))
        return self.T_a + p_igbt * self.R_th_ja
