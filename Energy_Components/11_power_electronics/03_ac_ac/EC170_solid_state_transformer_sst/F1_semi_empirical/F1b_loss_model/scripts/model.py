"""
EC170 -- Solid State Transformer (SST) -- F1b Detailed Loss Model

An SST replaces a conventional line-frequency transformer with three power-electronic
conversion stages:

  Stage 1: AC-DC rectifier (front-end, high voltage side)
  Stage 2: Isolated DC-DC converter (DAB or LLC, provides galvanic isolation)
  Stage 3: DC-AC inverter (output, low voltage side)

Each stage contributes losses that can be modelled analytically.

────────────────────────────────────────────────────────────────────
Stage 1 — Front-End AC-DC (IGBT H-bridge or NPC):
    Same per-device formulae as AFE/inverter (EC164/EC167):
    P_s1 = 4 * (P_cond_igbt + P_sw_igbt + P_cond_diode + P_rr)
    (4 devices for single-phase H-bridge, use 6 for 3-phase)

Stage 2 — Isolated DC-DC (Dual Active Bridge):
    Both primary and secondary H-bridges:
        P_sw_dab = 2 * n_dev * E_sw * f_sw_dab * V_hv/V_ref * I_hv/I_ref
    Transformer copper loss:
        P_xfmr_cu = I_rms_xfmr^2 * R_winding
    Transformer core loss (Steinmetz):
        P_xfmr_core = k_fe * f_sw_dab^alpha * B_peak^beta * V_core
    For a DAB at phase-shift phi:
        I_rms_xfmr ~ I_rated * sqrt(phi/pi * (1 - phi/pi))  (simplified)
        At maximum power (phi = pi/2): I_rms_xfmr ~ I_rated / sqrt(2)

Stage 3 — Output DC-AC Inverter (same as EC164):
    P_s3 = 6 * (P_cond_igbt + P_sw_igbt + P_cond_diode + P_rr)

Total:
    P_total = P_s1 + P_s2 + P_s3
    eta = P_out / (P_out + P_total)

Temperature:
    T_j_worst = T_a + P_per_device * R_th_ja  (stage 1 or 3 IGBT)

References:
    Krismer, F. & Kolar, J.W. (2012). Efficiency-Optimized High-Current Dual
    Active Bridge Converter for Automotive Applications. IEEE Trans. Ind. Electron.,
    59(7), 2745-2760.
    She, X., Huang, A.Q., Burgos, R. (2013). Review of Solid-State Transformer
    Technologies. IEEE J. Emerg. Sel. Topics Power Electron., 1(3), 186-198.
"""

import numpy as np


class SolidStateTransformerF1b:
    """Solid State Transformer -- three-stage per-device loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated"]["value"]        # W
        self.V_hv = u["V_hv"]["value"]              # V  HV-side DC link
        self.V_lv = u["V_lv"]["value"]              # V  LV-side DC link
        self.turns_ratio = u["turns_ratio"]["value"] # n (HV/LV)

        # Stage 1 -- front-end (single-phase H-bridge, 4 IGBTs)
        self.V_ce0_s1 = u["V_ce0_s1"]["value"]
        self.r_ce_s1 = u["r_ce_s1"]["value"]
        self.E_on_s1 = u["E_on_s1"]["value"]
        self.E_off_s1 = u["E_off_s1"]["value"]
        self.V_f_s1 = u["V_f_s1"]["value"]
        self.r_d_s1 = u["r_d_s1"]["value"]
        self.E_rr_s1 = u["E_rr_s1"]["value"]
        self.f_sw_s1 = u["f_sw_s1"]["value"]        # Hz
        self.V_ref_s1 = u["V_ref_s1"]["value"]
        self.I_ref_s1 = u["I_ref_s1"]["value"]

        # Stage 2 -- DAB isolated DC-DC
        self.R_winding = u["R_winding"]["value"]     # Ohm (referred to HV side)
        self.P_core_rated = u["P_core_rated"]["value"] # W (core loss at rated flux)
        self.f_sw_dab = u["f_sw_dab"]["value"]       # Hz
        self.E_sw_dab = u["E_sw_dab"]["value"]       # J (per device per event)
        self.V_ref_dab = u["V_ref_dab"]["value"]
        self.I_ref_dab = u["I_ref_dab"]["value"]
        self.n_dev_dab = 8  # two H-bridges (4 + 4 devices total)

        # Stage 3 -- output inverter (single-phase H-bridge or 3-phase)
        self.V_ce0_s3 = u["V_ce0_s3"]["value"]
        self.r_ce_s3 = u["r_ce_s3"]["value"]
        self.E_on_s3 = u["E_on_s3"]["value"]
        self.E_off_s3 = u["E_off_s3"]["value"]
        self.V_f_s3 = u["V_f_s3"]["value"]
        self.r_d_s3 = u["r_d_s3"]["value"]
        self.E_rr_s3 = u["E_rr_s3"]["value"]
        self.f_sw_s3 = u["f_sw_s3"]["value"]
        self.V_ref_s3 = u["V_ref_s3"]["value"]
        self.I_ref_s3 = u["I_ref_s3"]["value"]

        self.R_th_ja = u["R_th_ja"]["value"]         # K/W (worst-case device)
        self.T_a = u["T_a"]["value"]

    # ------------------------------------------------------------------
    # Current stress helpers
    # ------------------------------------------------------------------

    def _hv_peak_current(self, p_out, power_factor=1.0):
        """Peak current on HV AC side [A]."""
        p = np.asarray(p_out, dtype=float)
        v_hv_rms = self.V_hv / np.sqrt(2.0)  # assume V_hv is DC ≈ V_peak_hv_ac
        denom = v_hv_rms * np.asarray(power_factor, dtype=float)
        denom = np.where(denom > 0, denom, 1.0)
        i_rms = np.where(p > 0, p / denom, 0.0)
        return i_rms * np.sqrt(2.0)

    def _lv_peak_current(self, p_out, power_factor=1.0):
        """Peak current on LV AC side [A]."""
        p = np.asarray(p_out, dtype=float)
        v_lv_rms = self.V_lv / np.sqrt(2.0)
        denom = v_lv_rms * np.asarray(power_factor, dtype=float)
        denom = np.where(denom > 0, denom, 1.0)
        i_rms = np.where(p > 0, p / denom, 0.0)
        return i_rms * np.sqrt(2.0)

    def _dab_rms_current(self, p_out):
        """DAB transformer RMS current [A] (simplified, referred to HV side)."""
        p = np.asarray(p_out, dtype=float)
        p_pu = p / np.where(self.P_rated > 0, self.P_rated, 1.0)
        # At rated power (phi=pi/2): I_rms = I_rated
        # At part power: I_rms ≈ I_rated * sqrt(p_pu) (simplified sinusoidal approx)
        i_rated = self.P_rated / self.V_hv
        return i_rated * np.sqrt(np.clip(p_pu, 0.0, 1.2))

    # ------------------------------------------------------------------
    # Stage 1: Front-end AC-DC (single-phase H-bridge, 4 devices)
    # ------------------------------------------------------------------

    def _stage1_cond_per_device(self, i_pk, m=0.9, pf=1.0):
        # SPWM H-bridge (m~0.9 near unity PF):
        i_avg = i_pk / (2.0 * np.pi) * (1.0 + m * np.pi * pf / 4.0)
        i_rms_sq = i_pk ** 2 * (1.0 / 8.0 + m * pf / (3.0 * np.pi))
        return self.V_ce0_s1 * i_avg + self.r_ce_s1 * i_rms_sq

    def _stage1_sw_per_device(self, i_pk):
        i_sw = i_pk / np.pi
        return (self.E_on_s1 + self.E_off_s1) * self.f_sw_s1 * (self.V_hv / self.V_ref_s1) * (i_sw / self.I_ref_s1)

    def _stage1_diode_cond_per_device(self, i_pk, m=0.9, pf=1.0):
        i_avg = i_pk / (2.0 * np.pi) * (1.0 - m * np.pi * pf / 4.0)
        i_avg = np.maximum(i_avg, 0.0)
        i_rms_sq = i_pk ** 2 * np.maximum(1.0 / 8.0 - m * pf / (3.0 * np.pi), 0.0)
        return self.V_f_s1 * i_avg + self.r_d_s1 * i_rms_sq

    def _stage1_rr_per_device(self, i_pk):
        i_sw = i_pk / np.pi
        return self.E_rr_s1 * self.f_sw_s1 * (self.V_hv / self.V_ref_s1) * (i_sw / self.I_ref_s1)

    def stage1_losses(self, p_out, power_factor=1.0):
        """Stage 1 (front-end) total losses [W]."""
        i_pk = self._hv_peak_current(p_out, power_factor)
        pf = np.asarray(power_factor, dtype=float)
        n = 4  # single-phase H-bridge
        return n * (self._stage1_cond_per_device(i_pk, pf=pf) +
                    self._stage1_sw_per_device(i_pk) +
                    self._stage1_diode_cond_per_device(i_pk, pf=pf) +
                    self._stage1_rr_per_device(i_pk))

    # ------------------------------------------------------------------
    # Stage 2: Isolated DC-DC (DAB)
    # ------------------------------------------------------------------

    def stage2_losses(self, p_out):
        """Stage 2 (DAB isolated DC-DC) total losses [W]."""
        i_rms = self._dab_rms_current(p_out)
        i_pk = i_rms * np.sqrt(2.0)

        # Copper loss
        P_cu = i_rms ** 2 * self.R_winding

        # Core loss (approximately constant at given flux)
        P_core = np.where(np.asarray(p_out, dtype=float) > 0,
                          self.P_core_rated, 0.0)

        # Switching loss (8 devices in two H-bridges)
        i_sw = i_pk / np.pi
        P_sw = self.n_dev_dab * self.E_sw_dab * self.f_sw_dab * (self.V_hv / self.V_ref_dab) * (i_sw / self.I_ref_dab)

        return P_cu + P_core + P_sw

    # ------------------------------------------------------------------
    # Stage 3: Output DC-AC (single-phase H-bridge, 4 devices)
    # ------------------------------------------------------------------

    def _stage3_cond_per_device(self, i_pk, m=0.9, pf=1.0):
        i_avg = i_pk / (2.0 * np.pi) * (1.0 + m * np.pi * pf / 4.0)
        i_rms_sq = i_pk ** 2 * (1.0 / 8.0 + m * pf / (3.0 * np.pi))
        return self.V_ce0_s3 * i_avg + self.r_ce_s3 * i_rms_sq

    def _stage3_sw_per_device(self, i_pk):
        i_sw = i_pk / np.pi
        return (self.E_on_s3 + self.E_off_s3) * self.f_sw_s3 * (self.V_lv / self.V_ref_s3) * (i_sw / self.I_ref_s3)

    def _stage3_diode_cond_per_device(self, i_pk, m=0.9, pf=1.0):
        i_avg = i_pk / (2.0 * np.pi) * (1.0 - m * np.pi * pf / 4.0)
        i_avg = np.maximum(i_avg, 0.0)
        i_rms_sq = i_pk ** 2 * np.maximum(1.0 / 8.0 - m * pf / (3.0 * np.pi), 0.0)
        return self.V_f_s3 * i_avg + self.r_d_s3 * i_rms_sq

    def _stage3_rr_per_device(self, i_pk):
        i_sw = i_pk / np.pi
        return self.E_rr_s3 * self.f_sw_s3 * (self.V_lv / self.V_ref_s3) * (i_sw / self.I_ref_s3)

    def stage3_losses(self, p_out, power_factor=1.0):
        """Stage 3 (output inverter) total losses [W]."""
        i_pk = self._lv_peak_current(p_out, power_factor)
        pf = np.asarray(power_factor, dtype=float)
        n = 4
        return n * (self._stage3_cond_per_device(i_pk, pf=pf) +
                    self._stage3_sw_per_device(i_pk) +
                    self._stage3_diode_cond_per_device(i_pk, pf=pf) +
                    self._stage3_rr_per_device(i_pk))

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------

    def loss_breakdown(self, p_out, power_factor=1.0):
        """Full SST loss breakdown [W]."""
        return {
            "p_stage1_w": self.stage1_losses(p_out, power_factor),
            "p_stage2_w": self.stage2_losses(p_out),
            "p_stage3_w": self.stage3_losses(p_out, power_factor),
        }

    def total_losses(self, p_out, power_factor=1.0):
        bd = self.loss_breakdown(p_out, power_factor)
        return bd["p_stage1_w"] + bd["p_stage2_w"] + bd["p_stage3_w"]

    def efficiency(self, p_out, power_factor=1.0):
        """SST efficiency = P_out / (P_out + P_loss)."""
        p = np.asarray(p_out, dtype=float)
        p_loss = self.total_losses(p_out, power_factor)
        p_in = p + p_loss
        safe = p_in > 0
        return np.where(safe, p / np.where(safe, p_in, 1.0), 0.0)

    def junction_temperature(self, p_out, power_factor=1.0):
        """Worst-case IGBT junction temperature [degC] (HV-side devices in Stage 1)."""
        i_pk = self._hv_peak_current(p_out, power_factor)
        pf = np.asarray(power_factor, dtype=float)
        p_device = (self._stage1_cond_per_device(i_pk, pf=pf) +
                    self._stage1_sw_per_device(i_pk))
        return self.T_a + p_device * self.R_th_ja
