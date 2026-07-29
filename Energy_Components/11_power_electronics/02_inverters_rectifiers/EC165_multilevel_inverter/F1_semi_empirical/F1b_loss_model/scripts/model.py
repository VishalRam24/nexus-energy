"""
EC165 -- Multilevel Inverter (3-level NPC/T-type) -- F1b Detailed Loss Model

Extends F1a (ideal gain + efficiency map) with per-device loss physics for a
three-level Neutral-Point Clamped (NPC) or T-type topology.

Each phase leg contains 4 active switches + 2 clamping diodes (NPC) or
4 switches with inner pair rated at half the DC bus (T-type).  The clamping
devices carry only the inner voltage stress (V_dc / 2).

Loss model per OUTPUT phase (×3 for three-phase total):
──────────────────────────────────────────────────────
Outer IGBT pair (S1/S4, rated V_dc):
    Conduction (S1): P_cond_outer = V_ce0_out * I_avg_S1 + r_ce_out * I_rms_S1^2
    Switching (S1):  P_sw_outer = (E_on + E_off) * f_sw * (V_dc/V_ref) * (I_sw/I_ref)

    For sinusoidal PWM (modulation index m, power factor cos φ):
        I_avg_S1  = I_pk/(2π) * (1 + π*m*cosφ/4)
        I_rms_S1  = I_pk * sqrt(1/8 + m*cosφ/(3π))

Inner IGBT pair (S2/S3, rated V_dc/2):
    Same current expressions as outer pair (for NPC both outer and inner conduct
    simultaneously when the phase voltage is at the upper/lower rail).
    Voltage stress is halved → switching energy scales with V_dc/2:
        P_sw_inner = (E_on_inner + E_off_inner) * f_sw * (0.5*V_dc/V_ref) * (I_sw/I_ref)
    Inner conduction same formula as outer.

Clamping diode (D_clamp, each rated V_dc/2):
    Conduction: P_cond_clamp = V_f_clamp * I_avg_clamp + r_d_clamp * I_rms_clamp^2
    Reverse recovery: P_rr_clamp = E_rr_clamp * f_sw * (0.5*V_dc/V_ref) * (I_sw/I_ref)

    I_avg_clamp  = I_pk/(2π) * (1 - π*m*cosφ/4)   (clamp only conducts in freewheeling)
    I_rms_clamp  = I_pk * sqrt(max(1/8 - m*cosφ/(3π), 0))

Total (3 phases × per-phase total):
    P_total = 3 * (P_outer_sw + P_inner_sw + P_cond_outer + P_cond_inner +
                   P_cond_clamp + P_rr_clamp) * devices_per_element

Temperature:
    T_j = T_a + P_device * R_th_ja
    where R_th_ja is junction-to-ambient thermal resistance of the module.

References:
    Nabae, A., Takahashi, I., Akagi, H. (1981). IEEE Trans. Ind. Appl., IA-17(5).
    Semikron Application Manual (2015). Power Semiconductors, section on NPC inverters.
    Holmes, D.G. & Lipo, T.A. (2003). Pulse Width Modulation for Power Converters. Wiley.
"""

import numpy as np


class MultilevelInverterF1b:
    """3-level NPC/T-type multilevel inverter -- detailed per-device loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_dc = u["V_dc"]["value"]          # V  (total DC bus)
        self.P_rated = u["P_rated"]["value"]     # W
        self.n_levels = u["n_levels"]["value"]   # number of output voltage levels (3)

        # Outer IGBT (rated V_dc)
        self.V_ce0_out = u["V_ce0_outer"]["value"]   # V
        self.r_ce_out = u["r_ce_outer"]["value"]     # Ohm
        self.E_on_out = u["E_on_outer"]["value"]     # J
        self.E_off_out = u["E_off_outer"]["value"]   # J

        # Inner IGBT (rated V_dc/2)
        self.V_ce0_in = u["V_ce0_inner"]["value"]    # V
        self.r_ce_in = u["r_ce_inner"]["value"]      # Ohm
        self.E_on_in = u["E_on_inner"]["value"]      # J
        self.E_off_in = u["E_off_inner"]["value"]    # J

        # Clamping diode
        self.V_f_clamp = u["V_f_clamp"]["value"]     # V
        self.r_d_clamp = u["r_d_clamp"]["value"]     # Ohm
        self.E_rr_clamp = u["E_rr_clamp"]["value"]  # J

        self.f_sw = u["f_sw"]["value"]          # Hz
        self.V_ref = u["V_ref"]["value"]         # V (reference bus for loss scaling)
        self.I_ref = u["I_ref"]["value"]         # A
        self.R_th_ja = u["R_th_ja"]["value"]     # K/W (module junction-to-ambient)
        self.T_a = u["T_a"]["value"]             # degC ambient

    # ------------------------------------------------------------------
    # AC output quantities
    # ------------------------------------------------------------------

    def ac_rms_voltage(self, v_dc, m):
        """Line-to-neutral RMS voltage [V]: V_an = m * V_dc / (2*sqrt(2))."""
        return np.asarray(m, dtype=float) * np.asarray(v_dc, dtype=float) / (2.0 * np.sqrt(2.0))

    def phase_peak_current(self, v_dc, m, p_load, power_factor=0.9):
        """Peak phase current [A] from output active power."""
        p = np.asarray(p_load, dtype=float)
        m_ = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        v_ln = self.ac_rms_voltage(v_dc, m_)
        # P = 3 * V_ln_rms * I_rms * pf  =>  I_rms = P / (3 * V_ln * pf)
        denom = 3.0 * v_ln * pf
        denom = np.where(denom > 1e-6, denom, 1e-6)
        i_rms = np.where(p > 0, p / denom, 0.0)
        return i_rms * np.sqrt(2.0)

    # ------------------------------------------------------------------
    # Per-device current stress (sinusoidal PWM, per phase)
    # ------------------------------------------------------------------

    def _outer_igbt_i_avg(self, i_pk, m, pf):
        return i_pk / (2.0 * np.pi) * (1.0 + np.pi * m * pf / 4.0)

    def _outer_igbt_i_rms_sq(self, i_pk, m, pf):
        return i_pk ** 2 * (1.0 / 8.0 + m * pf / (3.0 * np.pi))

    def _clamp_diode_i_avg(self, i_pk, m, pf):
        val = i_pk / (2.0 * np.pi) * (1.0 - np.pi * m * pf / 4.0)
        return np.maximum(val, 0.0)

    def _clamp_diode_i_rms_sq(self, i_pk, m, pf):
        return i_pk ** 2 * np.maximum(1.0 / 8.0 - m * pf / (3.0 * np.pi), 0.0)

    # ------------------------------------------------------------------
    # Loss components (per phase, per pair)
    # ------------------------------------------------------------------

    def outer_igbt_conduction_per_phase(self, i_pk, m, pf):
        """Outer IGBT conduction loss per phase [W] (two outer devices S1+S4)."""
        i_pk = np.asarray(i_pk, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(pf, dtype=float)
        i_avg = self._outer_igbt_i_avg(i_pk, m, pf)
        i_rms_sq = self._outer_igbt_i_rms_sq(i_pk, m, pf)
        # Two outer devices per phase (S1 and S4 conduct symmetrically)
        return 2.0 * (self.V_ce0_out * i_avg + self.r_ce_out * i_rms_sq)

    def inner_igbt_conduction_per_phase(self, i_pk, m, pf):
        """Inner IGBT conduction loss per phase [W] (two inner devices S2+S3)."""
        i_pk = np.asarray(i_pk, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(pf, dtype=float)
        i_avg = self._outer_igbt_i_avg(i_pk, m, pf)
        i_rms_sq = self._outer_igbt_i_rms_sq(i_pk, m, pf)
        return 2.0 * (self.V_ce0_in * i_avg + self.r_ce_in * i_rms_sq)

    def outer_igbt_switching_per_phase(self, i_pk, v_dc):
        """Outer IGBT switching loss per phase [W] (S1+S4, voltage = V_dc)."""
        i_pk = np.asarray(i_pk, dtype=float)
        v_dc = np.asarray(v_dc, dtype=float)
        i_sw = i_pk / np.pi
        v_ratio = v_dc / self.V_ref
        # Two outer devices
        return 2.0 * (self.E_on_out + self.E_off_out) * self.f_sw * v_ratio * (i_sw / self.I_ref)

    def inner_igbt_switching_per_phase(self, i_pk, v_dc):
        """Inner IGBT switching loss per phase [W] (S2+S3, voltage = V_dc/2)."""
        i_pk = np.asarray(i_pk, dtype=float)
        v_dc = np.asarray(v_dc, dtype=float)
        i_sw = i_pk / np.pi
        v_ratio = (v_dc / 2.0) / self.V_ref
        return 2.0 * (self.E_on_in + self.E_off_in) * self.f_sw * v_ratio * (i_sw / self.I_ref)

    def clamping_diode_conduction_per_phase(self, i_pk, m, pf):
        """Clamping diode conduction loss per phase [W] (2 clamp diodes)."""
        i_pk = np.asarray(i_pk, dtype=float)
        m = np.asarray(m, dtype=float)
        pf = np.asarray(pf, dtype=float)
        i_avg = self._clamp_diode_i_avg(i_pk, m, pf)
        i_rms_sq = self._clamp_diode_i_rms_sq(i_pk, m, pf)
        return 2.0 * (self.V_f_clamp * i_avg + self.r_d_clamp * i_rms_sq)

    def clamping_diode_recovery_per_phase(self, i_pk, v_dc):
        """Clamping diode reverse recovery loss per phase [W]."""
        i_pk = np.asarray(i_pk, dtype=float)
        v_dc = np.asarray(v_dc, dtype=float)
        i_sw = i_pk / np.pi
        v_ratio = (v_dc / 2.0) / self.V_ref
        return 2.0 * self.E_rr_clamp * self.f_sw * v_ratio * (i_sw / self.I_ref)

    # ------------------------------------------------------------------
    # Total losses and efficiency
    # ------------------------------------------------------------------

    def loss_breakdown(self, v_dc, p_load, m, power_factor=0.9):
        """Full loss breakdown for 3-phase NPC inverter [W]."""
        m = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        i_pk = self.phase_peak_current(v_dc, m, p_load, power_factor)

        p_outer_cond = 3.0 * self.outer_igbt_conduction_per_phase(i_pk, m, pf)
        p_inner_cond = 3.0 * self.inner_igbt_conduction_per_phase(i_pk, m, pf)
        p_outer_sw = 3.0 * self.outer_igbt_switching_per_phase(i_pk, v_dc)
        p_inner_sw = 3.0 * self.inner_igbt_switching_per_phase(i_pk, v_dc)
        p_clamp_cond = 3.0 * self.clamping_diode_conduction_per_phase(i_pk, m, pf)
        p_clamp_rr = 3.0 * self.clamping_diode_recovery_per_phase(i_pk, v_dc)

        return {
            "p_outer_igbt_cond_w": p_outer_cond,
            "p_inner_igbt_cond_w": p_inner_cond,
            "p_outer_igbt_sw_w": p_outer_sw,
            "p_inner_igbt_sw_w": p_inner_sw,
            "p_clamp_diode_cond_w": p_clamp_cond,
            "p_clamp_diode_rr_w": p_clamp_rr,
        }

    def total_losses(self, v_dc, p_load, m, power_factor=0.9):
        """Total inverter losses [W]."""
        bd = self.loss_breakdown(v_dc, p_load, m, power_factor)
        return sum(bd.values())

    def efficiency(self, v_dc, p_load, m, power_factor=0.9):
        """Efficiency = P_out / (P_out + P_loss)."""
        p_load = np.asarray(p_load, dtype=float)
        p_loss = self.total_losses(v_dc, p_load, m, power_factor)
        p_in = p_load + p_loss
        safe = p_in > 0
        return np.where(safe, p_load / np.where(safe, p_in, 1.0), 0.0)

    def junction_temperature(self, v_dc, p_load, m, power_factor=0.9):
        """
        Junction temperature [degC] of the hottest device (outer IGBT + clamping diode).
        T_j = T_a + P_device_per_switch * R_th_ja
        """
        p_load = np.asarray(p_load, dtype=float)
        m_ = np.asarray(m, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        i_pk = self.phase_peak_current(v_dc, m_, p_load, power_factor)

        # Worst-case device: one outer IGBT in one phase
        p_one_outer = (self.outer_igbt_conduction_per_phase(i_pk, m_, pf) / 2.0 +
                       self.outer_igbt_switching_per_phase(i_pk, v_dc) / 2.0)
        return self.T_a + p_one_outer * self.R_th_ja
