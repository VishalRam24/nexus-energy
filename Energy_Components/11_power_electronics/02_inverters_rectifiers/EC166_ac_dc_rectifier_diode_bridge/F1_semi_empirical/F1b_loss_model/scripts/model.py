"""
EC166 -- AC-DC Rectifier (Diode Bridge) -- F1b Detailed Diode Loss Model

Extends F1a (ideal rectification + fixed efficiency) with per-device physics
for a three-phase uncontrolled diode bridge (6-pulse) or single-phase bridge (2-pulse).

Three-phase full-wave bridge (6-pulse):
    6 diodes; each diode conducts for exactly 1/3 of the line cycle.
    Average current per diode:   I_D_avg = I_dc / 3
    RMS current per diode:       I_D_rms = I_dc / sqrt(3)

Single-phase full-wave bridge (4-pulse):
    4 diodes; each diode conducts for exactly 1/2 of the line cycle.
    Average current per diode:   I_D_avg = I_dc / 2
    RMS current per diode:       I_D_rms = I_dc / sqrt(2)

Diode conduction loss per device:
    P_cond = V_f * I_D_avg + r_d * I_D_rms^2

Diode reverse recovery loss (relevant at line frequency, small but non-zero):
    P_rr = E_rr * f_line * (V_peak / V_ref) * (I_D_avg / I_ref)
    (E_rr: stored charge × peak voltage, recovered once per cycle per device)

Total losses (n_diodes devices):
    P_total = n_diodes * (P_cond + P_rr)

DC output voltage (ideal, neglecting commutation overlap):
    Three-phase: V_dc = 3*sqrt(3)/π * V_phase_peak = 1.3505 * V_L_rms
    Single-phase: V_dc = 2/π * V_peak = 0.9003 * V_rms

Thermal:
    T_j = T_a + P_device * R_th_ja

References:
    Mohan, Undeland & Robbins (2003). Power Electronics, 3rd ed. Wiley, ch. 3.
    Rashid, M.H. (2011). Power Electronics Handbook, 3rd ed. Elsevier.
    ON Semiconductor (2016). MBRF20200CT Application Note.
"""

import numpy as np


class DiodeBridgeRectifierF1b:
    """AC-DC diode bridge rectifier -- detailed per-diode loss model."""

    # Rectification constants
    _VOUT_COEFF = {
        3: 3.0 * np.sqrt(3.0) / np.pi,   # ≈ 1.6546 × V_phase_rms_peak / sqrt(2) -- see below
        1: 2.0 / np.pi,
    }
    # For 3-phase: V_dc_ideal = (3√3/π) * V_phase_peak = 1.3505 * V_L-L_rms
    # For 1-phase: V_dc_ideal = (2/π) * V_peak = 0.9003 * V_rms

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_phases = int(u["n_phases"]["value"])    # 1 or 3
        self.V_f = u["V_f"]["value"]                   # V  (threshold voltage)
        self.r_d = u["r_d"]["value"]                   # Ohm (slope resistance)
        self.E_rr = u["E_rr"]["value"]                 # J  (reverse recovery energy)
        self.f_line = u["f_line"]["value"]             # Hz (50 or 60)
        self.V_ref = u["V_ref"]["value"]               # V  (ref voltage for E_rr scaling)
        self.I_ref = u["I_ref"]["value"]               # A
        self.R_th_ja = u["R_th_ja"]["value"]           # K/W
        self.T_a = u["T_a"]["value"]                   # degC

        # Number of diodes in the bridge
        self.n_diodes = 6 if self.n_phases == 3 else 4

        # Current sharing: fraction of I_dc flowing through one diode (avg and rms)
        if self.n_phases == 3:
            self._k_avg = 1.0 / 3.0     # I_D_avg = I_dc / 3
            self._k_rms = 1.0 / np.sqrt(3.0)   # I_D_rms = I_dc / sqrt(3)
        else:
            self._k_avg = 0.5
            self._k_rms = 1.0 / np.sqrt(2.0)

    def ideal_dc_voltage(self, v_ac_rms):
        """
        Ideal no-load DC output voltage [V].
        Three-phase: V_dc = 1.3505 * V_L-L_rms
        Single-phase: V_dc = 0.9003 * V_rms
        """
        v = np.asarray(v_ac_rms, dtype=float)
        if self.n_phases == 3:
            # V_L-L_rms × (3√3/π) / √2 = 1.3505
            return v * (3.0 * np.sqrt(3.0) / np.pi) / np.sqrt(2.0) * np.sqrt(2.0)
            # Simplification: V_dc = (3√3/π) * V_phase_peak
            # V_phase_peak = V_L-L_rms * sqrt(2/3)
        else:
            return v * 2.0 * np.sqrt(2.0) / np.pi   # = 0.9003 * V_rms

    def _dc_voltage_3phase(self, v_ll_rms):
        """Three-phase: V_dc = (3√3/π) * V_LL_rms / sqrt(3) * sqrt(2) = 1.3505 * V_LL_rms."""
        return np.asarray(v_ll_rms, dtype=float) * 1.3505

    def _dc_voltage_1phase(self, v_rms):
        return np.asarray(v_rms, dtype=float) * 2.0 * np.sqrt(2.0) / np.pi

    def dc_voltage(self, v_ac_rms):
        """Ideal DC output voltage [V]."""
        if self.n_phases == 3:
            return self._dc_voltage_3phase(v_ac_rms)
        return self._dc_voltage_1phase(v_ac_rms)

    def diode_currents(self, i_dc):
        """Average and RMS current per diode [A]."""
        i_dc = np.asarray(i_dc, dtype=float)
        return i_dc * self._k_avg, i_dc * self._k_rms

    def diode_conduction_loss(self, i_dc):
        """
        Conduction loss per diode [W]: V_f * I_avg + r_d * I_rms^2.
        Total = n_diodes × per-diode.
        """
        i_avg, i_rms = self.diode_currents(i_dc)
        p_per_diode = self.V_f * i_avg + self.r_d * i_rms ** 2
        return p_per_diode, self.n_diodes * p_per_diode

    def diode_recovery_loss(self, i_dc, v_ac_rms):
        """
        Reverse recovery loss per diode [W]: E_rr * f_line * (V_peak/V_ref) * (I_avg/I_ref).
        Total = n_diodes × per-diode.
        """
        i_avg, _ = self.diode_currents(i_dc)
        v_ac = np.asarray(v_ac_rms, dtype=float)
        v_peak = v_ac * np.sqrt(2.0)
        v_ratio = v_peak / self.V_ref
        i_ratio = i_avg / self.I_ref
        p_per_diode = self.E_rr * self.f_line * v_ratio * i_ratio
        return p_per_diode, self.n_diodes * p_per_diode

    def total_losses(self, v_ac_rms, i_dc):
        """Total rectifier losses [W]."""
        _, p_cond = self.diode_conduction_loss(i_dc)
        _, p_rr = self.diode_recovery_loss(i_dc, v_ac_rms)
        return p_cond + p_rr

    def loss_breakdown(self, v_ac_rms, i_dc):
        """Loss breakdown dict [W]."""
        _, p_cond = self.diode_conduction_loss(i_dc)
        _, p_rr = self.diode_recovery_loss(i_dc, v_ac_rms)
        return {
            "p_conduction_w": p_cond,
            "p_recovery_w": p_rr,
        }

    def efficiency(self, v_ac_rms, i_dc):
        """Rectifier efficiency = P_dc / (P_dc + P_loss)."""
        v_dc = self.dc_voltage(v_ac_rms)
        i_dc_ = np.asarray(i_dc, dtype=float)
        p_out = v_dc * i_dc_
        p_loss = self.total_losses(v_ac_rms, i_dc)
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)

    def junction_temperature(self, i_dc, v_ac_rms):
        """Junction temperature of the hottest diode [degC]."""
        p_per_diode, _ = self.diode_conduction_loss(i_dc)
        p_rr_per_diode, _ = self.diode_recovery_loss(i_dc, v_ac_rms)
        p_device = p_per_diode + p_rr_per_diode
        return self.T_a + p_device * self.R_th_ja
